"""离线dev运行器：强制模拟预算检查、串行执行、持久化尝试标记。"""

import copy
import fcntl
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import retry_resume
import budget_guard as budget_module
from budget_guard import BudgetError, BudgetGuard, amount, tokens
from build_prompts import build_messages
from deepseek_client import build_request, validate_config
from experiment_log import (
    ERROR_KINDS,
    append_record,
    build_record,
    create_log,
    hash_json,
    json_text,
    read_records,
)
from retry_resume import POLICY_VERSION, decision, plan_next_attempt
from select_random_demos import select_for_condition


RUNNER_VERSION = "offline_dev_runner_v2_budgeted"


class SimulatedAPIError(Exception):
    """模拟已确认的接口错误，不保存原始异常文本。"""

    def __init__(self, kind, *, token_usage=None):
        if kind not in ERROR_KINDS:
            raise ValueError("未知模拟错误类别。")
        self.kind = kind
        # 仅用于明确提供收费证据的模拟场景；默认不假定零费用。
        self.token_usage = copy.deepcopy(token_usage)
        super().__init__(kind)


class SimulatedResponder:
    """只返回预先提供的模拟结果，没有网络调用代码。"""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0
        self.inputs = []

    def send(self, request):
        self.calls += 1
        self.inputs.append(copy.deepcopy(request))

        if not self.outcomes:
            raise RuntimeError("模拟响应已用完。")

        outcome = self.outcomes.pop(0)
        if isinstance(outcome, SimulatedAPIError):
            raise outcome
        if not isinstance(outcome, dict):
            raise ValueError("模拟响应必须是字典。")

        return copy.deepcopy(outcome)


def write_new_json(path, payload):
    """排他创建；文件已存在时停止，绝不覆盖。"""
    with Path(path).open("x", encoding="utf-8") as file:
        file.write(json_text(payload) + "\n")
        file.flush()
        os.fsync(file.fileno())


class DevRunner:
    def __init__(
        self,
        run_dir,
        *,
        run_id,
        config,
        artifact_hashes,
        budget_guard=None,
        max_input_tokens=None,
        create=False,
        is_simulated=True,
    ):
        if is_simulated is not True:
            raise ValueError("本版只允许模拟运行。")

        validate_config(config)
        runtime = config["runtime"]
        retries = runtime["transient_error_retries"]

        if (
            runtime["concurrency"] != 1
            or type(retries) is not int
            or not 0 <= retries <= 2
        ):
            raise ValueError("必须串行运行，额外重试次数只能为0到2。")

        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id必须是非空字符串。")
        if type(budget_guard) is not BudgetGuard:
            raise ValueError("必须提供已存在的模拟项目预算账本。")
        max_input_tokens = tokens(max_input_tokens)
        max_output_tokens = tokens(config["api"]["max_tokens"])
        if max_input_tokens == 0 or max_output_tokens == 0:
            raise ValueError("输入及输出token预留上界必须大于零。")
        budget_config = config.get("budget", {})
        budget_binding = budget_guard.inspect()["binding"]
        if (
            budget_config.get("currency") != "CNY"
            or amount(budget_config.get("project_total_cap"), positive=True)
            != amount(budget_binding["context"]["cap_cny"], positive=True)
        ):
            raise ValueError("项目配置与预算账本的币种或上限不一致。")

        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.config = copy.deepcopy(config)
        self.artifact_hashes = copy.deepcopy(artifact_hashes)
        self.budget_guard = budget_guard
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens

        self.manifest_path = self.run_dir / "run_manifest.json"
        self.result_path = self.run_dir / "results.jsonl"
        self.journal_path = self.run_dir / "attempts.jsonl"
        self.lock_path = self.run_dir / "run.lock"

        # 原日志指纹之外，额外固定运行器和重试策略源码。
        source_paths = {
            "dev_runner.py": Path(__file__),
            "retry_resume.py": Path(retry_resume.__file__),
            "budget_guard.py": Path(budget_module.__file__),
        }
        self.source_paths = source_paths
        self.context = {
            "runner_version": RUNNER_VERSION,
            "policy_version": POLICY_VERSION,
            "run_id": self.run_id,
            "is_simulated": True,
            "config_hash": hash_json(self.config),
            "artifact_hashes": self.artifact_hashes,
            "budget_binding": budget_binding,
            "token_bounds": {
                "input": self.max_input_tokens,
                "output": self.max_output_tokens,
            },
            "source_hashes": {
                name: hashlib.sha256(path.read_bytes()).hexdigest()
                for name, path in source_paths.items()
            },
        }
        self.context_hash = hash_json(self.context)

        if create:
            self.run_dir.mkdir(exist_ok=False)
            write_new_json(self.manifest_path, self.context)
            create_log(self.result_path)
            create_log(self.journal_path)
            with self.lock_path.open("x"):
                pass

        self.check_manifest()

    def check_manifest(self):
        for path in (
            self.manifest_path,
            self.result_path,
            self.journal_path,
            self.lock_path,
        ):
            if not path.is_file():
                raise ValueError("运行目录不完整，不能自动补建或续跑。")

        saved = json.loads(
            self.manifest_path.read_text(encoding="utf-8")
        )
        if saved != self.context:
            raise ValueError("运行身份、配置或文件指纹发生变化。")

        if (
            hash_json(self.config) != self.context["config_hash"]
            or self.artifact_hashes != self.context["artifact_hashes"]
        ):
            raise ValueError("运行期间输入配置发生变化。")

        current_sources = {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in self.source_paths.items()
        }
        if current_sources != self.context["source_hashes"]:
            raise ValueError("运行器或重试策略源码发生变化。")
        if (
            self.budget_guard.inspect()["binding"]
            != self.context["budget_binding"]
            or self.context["token_bounds"] != {
                "input": self.max_input_tokens,
                "output": self.max_output_tokens,
            }
        ):
            raise ValueError("项目预算账本或token预留上界发生变化。")

    def read_events(self):
        text = self.journal_path.read_text(encoding="utf-8")
        if text and not text.endswith("\n"):
            raise ValueError("尝试标记尾行不完整，请停止检查。")

        events = []
        previous_hash = None
        fields = {
            "state", "key", "result_record_hash",
            "context_hash", "previous_event_hash",
            "timestamp", "event_hash",
        }

        for line in text.splitlines():
            event = json.loads(line)
            if set(event) != fields:
                raise ValueError("尝试标记字段不正确。")

            payload = dict(event)
            saved_hash = payload.pop("event_hash")
            if saved_hash != hash_json(payload):
                raise ValueError("尝试标记指纹不一致。")
            if event["context_hash"] != self.context_hash:
                raise ValueError("尝试标记属于另一套运行配置。")
            if event["previous_event_hash"] != previous_hash:
                raise ValueError("尝试标记顺序或链条不一致。")

            previous_hash = saved_hash
            events.append(event)

        return events

    def append_event(self, state, key, result_hash=None):
        events = self.read_events()
        event = {
            "state": state,
            "key": list(key),
            "result_record_hash": result_hash,
            "context_hash": self.context_hash,
            "previous_event_hash": (
                events[-1]["event_hash"] if events else None
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        event["event_hash"] = hash_json(event)

        with self.journal_path.open("r+", encoding="utf-8") as file:
            file.seek(0, os.SEEK_END)
            file.write(json_text(event) + "\n")
            file.flush()
            os.fsync(file.fileno())

    def reconcile(self, records, *, repair=True):
        """核对标记与结果；只补确认标记，不重新模拟作答。"""
        starts = set()
        finished = {}
        counts = {}
        pending = None

        for event in self.read_events():
            raw_key = event["key"]
            if not isinstance(raw_key, list) or len(raw_key) != 2:
                raise ValueError("尝试标记编号不正确。")

            logical_id, attempt_no = raw_key
            if (
                not isinstance(logical_id, str)
                or len(logical_id) != 64
                or any(c not in "0123456789abcdef" for c in logical_id)
                or type(attempt_no) is not int
                or not 1 <= attempt_no <= 3
            ):
                raise ValueError("尝试标记编号不合法。")

            key = (logical_id, attempt_no)
            if event["state"] == "started":
                if pending is not None or key in starts:
                    raise ValueError("存在重叠或重复的开始标记。")
                if event["result_record_hash"] is not None:
                    raise ValueError("开始标记不能包含结果指纹。")
                if attempt_no != counts.get(logical_id, 0) + 1:
                    raise ValueError("尝试标记编号不连续。")

                counts[logical_id] = attempt_no
                starts.add(key)
                pending = key

            elif event["state"] == "recorded":
                if pending != key or key in finished:
                    raise ValueError("完成标记没有匹配的开始标记。")
                finished[key] = event["result_record_hash"]
                pending = None

            else:
                raise ValueError("未知尝试标记状态。")

        result_map = {}
        for record in records:
            key = (
                record["logical_request_id"],
                record["attempt_no"],
            )
            if key in result_map or key not in starts:
                raise ValueError("结果缺少开始标记或重复记录。")
            result_map[key] = record

        for key, digest in finished.items():
            if (
                key not in result_map
                or result_map[key]["record_hash"] != digest
            ):
                raise ValueError("完成标记与结果记录不一致。")

        if pending is not None and pending in result_map:
            if repair:
                self.append_event(
                    "recorded",
                    pending,
                    result_map[pending]["record_hash"],
                )
            return False

        return pending is not None

    @staticmethod
    def budget_attempt_id(key):
        return f"{key[0]}:{key[1]}"

    @staticmethod
    def confirmed_usage(record):
        usage = record["token_usage"]
        if not isinstance(usage, dict):
            return None
        try:
            input_count = tokens(usage.get("prompt_tokens"))
            output_count = tokens(usage.get("completion_tokens"))
            if "total_tokens" in usage and (
                tokens(usage["total_tokens"]) != input_count + output_count
            ):
                return None
        except BudgetError:
            return None
        return input_count, output_count

    def reconcile_budget(self, records):
        """仅凭已校验的结果用量恢复费用，证据不全时保留预留。"""
        inspection = self.budget_guard.inspect()
        attempts = inspection["attempts"]
        started_ids = {
            self.budget_attempt_id(event["key"])
            for event in self.read_events()
            if event["state"] == "started"
        }
        for attempt_id in started_ids:
            item = attempts.get(attempt_id)
            if item is None or (
                item["max_input_tokens"] != self.max_input_tokens
                or item["max_output_tokens"] != self.max_output_tokens
            ):
                raise ValueError("尝试标记缺少匹配的预算预留。")

        # 全部交叉校验通过后，才允许补记费用。
        repairs = []
        for record in records:
            attempt_id = self.budget_attempt_id((
                record["logical_request_id"], record["attempt_no"],
            ))
            item = attempts.get(attempt_id)
            if attempt_id not in started_ids or item is None:
                raise ValueError("结果记录缺少匹配的预算预留。")
            usage = self.confirmed_usage(record)
            if item["status"] in ("settled", "violated"):
                if (
                    usage is None
                    or item["evidence_hash"] != record["record_hash"]
                    or usage != (item["input_tokens"], item["output_tokens"])
                ):
                    raise ValueError("费用证据与结果记录不一致。")
            else:
                repairs.append((attempt_id, item["status"], usage, record))

        for attempt_id, status, usage, record in repairs:
            if usage is None:
                if status == "reserved":
                    self.budget_guard.mark_unknown(attempt_id)
            else:
                self.budget_guard.settle(
                    attempt_id,
                    input_tokens=usage[0],
                    output_tokens=usage[1],
                    evidence_hash=record["record_hash"],
                )

        view = self.budget_guard.summary()
        if view["halted"]:
            return "budget_halted"
        if view["pending_attempt_ids"]:
            if any(usage is None for _, _, usage, _ in repairs):
                return "budget_usage_unknown"
            return "budget_unconfirmed"
        return None

    def run_one(
        self,
        target,
        condition,
        demo_pool,
        *,
        responder,
        checkpoint=None,
    ):
        if type(responder) is not SimulatedResponder:
            raise TypeError("本版只接受内置SimulatedResponder。")

        demos = select_for_condition(
            demo_pool,
            target,
            self.config["demo_selection"]["seed"],
            condition,
        )
        messages = build_messages(target, condition, demos)

        common = {
            "run_id": self.run_id,
            "config": self.config,
            "target": target,
            "condition": condition,
            "demos": demos,
            "messages": messages,
            "artifact_hashes": self.artifact_hashes,
            "is_simulated": True,
        }

        # 只建立身份参照，不把这条占位错误写入结果日志。
        expected = build_record(
            **common,
            error_kind="unknown_error",
        )

        new_attempts = 0
        planned_waits = []

        def finish(plan):
            return {
                **plan,
                "new_attempts": new_attempts,
                "planned_wait_seconds": planned_waits,
                "is_simulated": True,
                "budget": self.budget_guard.summary(),
            }

        with self.lock_path.open("r+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("另一运行器正在使用这个目录。") from None

            try:
                while True:
                    self.check_manifest()
                    records = read_records(self.result_path)

                    # 先校验全部结果，再允许恢复确认标记。
                    plan_next_attempt(
                        records,
                        expected,
                        pending_attempt=False,
                    )
                    pending = self.reconcile(records, repair=False)
                    budget_reason = self.reconcile_budget(records)
                    self.reconcile(records)
                    plan = plan_next_attempt(
                        records,
                        expected,
                        pending_attempt=pending,
                    )

                    if plan["action"] in ("block", "stop"):
                        return finish(plan)
                    if budget_reason:
                        return finish(decision("block", budget_reason))
                    if plan["action"] == "skip":
                        return finish(plan)

                    attempt_no = plan["next_attempt_no"]
                    key = (expected["logical_request_id"], attempt_no)
                    attempt_id = self.budget_attempt_id(key)
                    needed = self.budget_guard.cost(
                        self.max_input_tokens, self.max_output_tokens,
                    )
                    if needed > amount(
                        self.budget_guard.summary()["available_cny"]
                    ):
                        return finish(decision("block", "budget_insufficient"))
                    try:
                        self.budget_guard.reserve(
                            attempt_id,
                            max_input_tokens=self.max_input_tokens,
                            max_output_tokens=self.max_output_tokens,
                        )
                    except BudgetError:
                        return finish(decision(
                            "block", "budget_reservation_blocked",
                        ))
                    if checkpoint:
                        checkpoint("after_budget_reserve")

                    if plan["wait_seconds"]:
                        planned_waits.append(plan["wait_seconds"])
                        # 离线测试只记录等待计划，不实际等待。

                    self.append_event("started", key)
                    if checkpoint:
                        checkpoint("after_start")

                    response = None
                    error_kind = None
                    error_token_usage = None
                    new_attempts += 1

                    try:
                        response = responder.send(
                            build_request(self.config, messages)
                        )
                    except SimulatedAPIError as error:
                        error_kind = error.kind
                        error_token_usage = error.token_usage
                    except Exception:
                        self.budget_guard.mark_unknown(attempt_id)
                        raise

                    if checkpoint:
                        checkpoint("after_response")

                    record = build_record(
                        **common,
                        response=response,
                        error_kind=error_kind,
                        error_token_usage=error_token_usage,
                        attempt_no=attempt_no,
                    )
                    append_record(self.result_path, record)

                    if checkpoint:
                        checkpoint("after_log")

                    self.reconcile_budget(read_records(self.result_path))
                    if checkpoint:
                        checkpoint("after_budget_settle")

                    self.append_event(
                        "recorded",
                        key,
                        record["record_hash"],
                    )
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
