"""Day 3 dev runner. Default: read-only preview. --live explicitly sends requests."""

import argparse
import copy
import fcntl
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get(
    "ICL_REASONING_PROJECT_ROOT", str(Path(__file__).resolve().parents[1])
)).resolve()
sys.path.insert(0, str(ROOT / "scripts"))

from build_prompts import build_messages
from deepseek_client import request_once, validate_config
from experiment_log import (
    append_record, build_record, collect_artifact_hashes,
    create_log, hash_json, json_text, read_records,
)
from retry_resume import plan_next_attempt
from select_random_demos import load_inputs, select_for_condition

VERSION = "live_dev_runner_v1"
CONDITIONS = ("C0", "C1", "C2")
UNCERTAIN_ERRORS = {"timeout", "connection", "server_error", "unknown_error"}


class LiveAPIError(Exception):
    def __init__(self, kind, status_code=None, request_id=None):
        self.kind = kind
        self.status_code = status_code
        self.request_id = request_id
        super().__init__(kind)


def send_real(config, messages):
    # Imported only after the user explicitly selected --live.
    from openai import OpenAI, APIStatusError, APITimeoutError, APIConnectionError
    try:
        return request_once(config, messages, client_factory=OpenAI)
    except APITimeoutError:
        raise LiveAPIError("timeout") from None
    except APIConnectionError:
        raise LiveAPIError("connection") from None
    except APIStatusError as exc:
        status = exc.status_code
        kind = (
            "rate_limit" if status == 429 else
            "authentication" if status == 401 else
            "permission" if status == 403 else
            "server_error" if status >= 500 else "invalid_request"
        )
        raise LiveAPIError(kind, status, exc.request_id) from None


def schedule(config, demos, targets):
    validate_config(config)
    if config["conditions"] != list(CONDITIONS):
        raise ValueError("必须使用 C0、C1、C2 三个条件。")
    if len(targets) != 20 or len(demos) != 120:
        raise ValueError("本版要求 demo=120、dev=20。")
    if config["runtime"]["concurrency"] != 1:
        raise ValueError("必须串行运行。")
    retries = config["runtime"]["transient_error_retries"]
    if type(retries) is not int or not 0 <= retries <= 2:
        raise ValueError("额外重试次数只能为 0 到 2。")
    result = []
    for target in sorted(targets, key=lambda item: item["id"]):
        for condition in CONDITIONS:
            selected = select_for_condition(
                demos, target, config["demo_selection"]["seed"], condition
            )
            result.append((target, condition, selected,
                           build_messages(target, condition, selected)))
    return result


def write_new(path, value):
    with path.open("x", encoding="utf-8") as file:
        file.write(json_text(value) + "\n")
        file.flush()
        os.fsync(file.fileno())


class LiveRunner:
    def __init__(self, run_dir, run_id, config, demos, targets, *, create=False):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id):
            raise ValueError("run-id 只能使用字母、数字、下划线和短横线。")
        self.config = copy.deepcopy(config)
        self.items = schedule(self.config, demos, targets)
        self.artifacts = collect_artifact_hashes()
        self.source_paths = {
            "run_dev_live.py": Path(__file__).resolve(),
            "retry_resume.py": ROOT / "scripts/retry_resume.py",
        }
        self.expected = {}
        for target, condition, selected, messages in self.items:
            record = build_record(
                run_id=run_id, config=self.config, target=target,
                condition=condition, demos=selected, messages=messages,
                artifact_hashes=self.artifacts, error_kind="unknown_error",
                is_simulated=False,
            )
            self.expected[(target["id"], condition)] = record
        self.context = {
            "runner_version": VERSION, "run_id": run_id,
            "is_simulated": False, "config_hash": hash_json(self.config),
            "artifact_hashes": self.artifacts,
            "source_hashes": self.source_hashes(),
            "schedule": [{
                "question_id": record["question_id"],
                "condition": record["condition"],
                "logical_request_id": record["logical_request_id"],
                "prompt_hash": record["prompt_hash"],
                "demo_ids": record["demo_ids"],
            } for record in self.expected.values()],
        }
        self.context_hash = hash_json(self.context)
        self.run_dir = Path(run_dir)
        self.manifest = self.run_dir / "run_manifest.json"
        self.results = self.run_dir / "results.jsonl"
        self.journal = self.run_dir / "attempts.jsonl"
        self.lock = self.run_dir / "run.lock"
        if create:
            self.run_dir.mkdir(exist_ok=False)
            write_new(self.manifest, self.context)
            create_log(self.results)
            create_log(self.journal)
            with self.lock.open("x"):
                pass
        self.check_context()

    def source_hashes(self):
        return {name: hashlib.sha256(path.read_bytes()).hexdigest()
                for name, path in self.source_paths.items()}

    def check_context(self):
        if not all(path.is_file() for path in (
            self.manifest, self.results, self.journal, self.lock
        )):
            raise ValueError("运行目录不完整，不能自动补建。")
        saved = json.loads(self.manifest.read_text(encoding="utf-8"))
        if saved != self.context:
            raise ValueError("运行身份、配置、数据或源码指纹已改变。")
        if (collect_artifact_hashes() != self.artifacts
                or self.source_hashes() != self.context["source_hashes"]
                or hash_json(self.config) != self.context["config_hash"]):
            raise ValueError("运行期间配置、数据或源码发生变化。")

    def read_events(self):
        text = self.journal.read_text(encoding="utf-8")
        if text and not text.endswith("\n"):
            raise ValueError("尝试日志尾行不完整，请保留文件并停止。")
        previous_hash = None
        events = []
        fields = {"state", "key", "result_record_hash", "context_hash",
                  "previous_event_hash", "timestamp", "event_hash"}
        for line in text.splitlines():
            event = json.loads(line)
            payload = dict(event)
            digest = payload.pop("event_hash", None)
            if (set(event) != fields or digest != hash_json(payload)
                    or event["context_hash"] != self.context_hash
                    or event["previous_event_hash"] != previous_hash):
                raise ValueError("尝试日志指纹、上下文或顺序不一致。")
            previous_hash = digest
            events.append(event)
        return events

    def event(self, state, key, result_hash=None):
        events = self.read_events()
        event = {
            "state": state, "key": list(key),
            "result_record_hash": result_hash,
            "context_hash": self.context_hash,
            "previous_event_hash": events[-1]["event_hash"] if events else None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        event["event_hash"] = hash_json(event)
        with self.journal.open("r+", encoding="utf-8") as file:
            file.seek(0, os.SEEK_END)
            file.write(json_text(event) + "\n")
            file.flush()
            os.fsync(file.fileno())

    def reconcile(self, records):
        starts, finished, counts = set(), {}, {}
        pending = None
        allowed = {record["logical_request_id"] for record in self.expected.values()}
        for event in self.read_events():
            raw = event["key"]
            if (not isinstance(raw, list) or len(raw) != 2
                    or raw[0] not in allowed or type(raw[1]) is not int
                    or not 1 <= raw[1] <= 3):
                raise ValueError("尝试日志的请求编号不合法。")
            key = tuple(raw)
            if event["state"] == "started":
                if (pending is not None or key in starts
                        or event["result_record_hash"] is not None
                        or key[1] != counts.get(key[0], 0) + 1):
                    raise ValueError("请求开始标记重复、重叠或编号不连续。")
                starts.add(key)
                counts[key[0]] = key[1]
                pending = key
            elif event["state"] == "recorded":
                if pending != key or key in finished:
                    raise ValueError("完成标记没有匹配的开始标记。")
                finished[key] = event["result_record_hash"]
                pending = None
            else:
                raise ValueError("未知尝试标记。")
        result_map = {}
        for record in records:
            key = (record["logical_request_id"], record["attempt_no"])
            if key not in starts or key in result_map:
                raise ValueError("结果缺少开始标记或重复保存。")
            result_map[key] = record
        if any(key not in result_map or result_map[key]["record_hash"] != digest
               for key, digest in finished.items()):
            raise ValueError("完成标记与结果不一致。")
        if pending is not None:
            if pending not in result_map:
                raise ValueError("存在已开始但结果不明的请求；停止，不自动重发。")
            self.event("recorded", pending, result_map[pending]["record_hash"])

    def history(self):
        self.check_context()
        records = read_records(self.results)
        for record in records:
            if (record["question_id"], record["condition"]) not in self.expected:
                raise ValueError("日志包含本次 dev 计划之外的记录。")
        # Validate every target's history before repairing any marker or calling API.
        for expected in self.expected.values():
            plan_next_attempt(records, expected, pending_attempt=False)
        self.reconcile(records)
        for record in records:
            if record["status"] == "api_error":
                if record["error_kind"] in UNCERTAIN_ERRORS:
                    raise ValueError("历史请求的结果不确定；保留记录，停止自动续跑。")
                continue
            usage = record["token_usage"]
            if (record["response_model"] != self.config["api"]["model"]
                    or not isinstance(usage, dict)
                    or any(type(usage.get(name)) is not int or usage[name] < 0
                           for name in ("prompt_tokens", "completion_tokens", "total_tokens"))
                    or usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]
                    or usage["prompt_tokens"] > 20000
                    or usage["completion_tokens"] > self.config["api"]["max_tokens"]):
                raise ValueError("已保存响应的模型名或用量异常；停止进一步请求。")
        return records

    def run(self, send, *, max_new_requests=60, checkpoint=None):
        if type(max_new_requests) is not int or not 1 <= max_new_requests <= 180:
            raise ValueError("max-new-requests 必须为 1 到 180 的整数。")
        sent = 0
        with self.lock.open("r+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("同一运行目录已有另一个进程在使用。") from None
            try:
                self.history()
                for target, condition, demos, messages in self.items:
                    expected = self.expected[(target["id"], condition)]
                    while True:
                        records = self.history()
                        plan = plan_next_attempt(records, expected, pending_attempt=False)
                        if plan["action"] == "skip":
                            break
                        if plan["action"] in ("block", "stop"):
                            raise ValueError("请求停止：" + plan["reason"])
                        if sent >= max_new_requests:
                            return self.summary(sent)
                        if plan["wait_seconds"]:
                            time.sleep(plan["wait_seconds"])
                        self.check_context()
                        attempt = plan["next_attempt_no"]
                        key = (expected["logical_request_id"], attempt)
                        self.event("started", key)
                        if checkpoint:
                            checkpoint("after_start")
                        sent += 1
                        response, error = None, None
                        try:
                            response = send(self.config, messages)
                        except LiveAPIError as exc:
                            error = exc.kind
                            print(f"API错误：{error}；HTTP={exc.status_code}；请求ID={exc.request_id}", flush=True)
                        if checkpoint:
                            checkpoint("after_response")
                        record = build_record(
                            run_id=self.context["run_id"], config=self.config,
                            target=target, condition=condition, demos=demos,
                            messages=messages, artifact_hashes=self.artifacts,
                            response=response, error_kind=error,
                            attempt_no=attempt, is_simulated=False,
                        )
                        append_record(self.results, record)
                        if checkpoint:
                            checkpoint("after_log")
                        self.event("recorded", key, record["record_hash"])
                        self.history()
                        if response is not None:
                            score = record["score"]
                            usage = record["token_usage"]
                            print(f"已保存：{target['id']} {condition} | strict_correct={score['strict_correct']} | tokens={usage['total_tokens']}", flush=True)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
        return self.summary(sent)

    def summary(self, sent=0):
        records = read_records(self.results)
        responses = [record for record in records if record["status"] != "api_error"]
        output = {"new_api_attempts": sent, "recorded_responses": len(responses),
                  "planned_responses": 60, "all_responses_recorded": len(responses) == 60,
                  "api_error_attempts": len(records) - len(responses), "conditions": {}}
        for condition in CONDITIONS:
            rows = [row for row in responses if row["condition"] == condition]
            output["conditions"][condition] = {
                "planned": 20, "recorded": len(rows),
                "strict_correct": sum(row["score"]["strict_correct"] for row in rows),
                "lenient_correct": sum(row["score"]["lenient_correct"] for row in rows),
            }
        return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--live", action="store_true")
    parser.add_argument("--run-id", default="day3_dev_live_v1_20261008")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-requests", type=int, default=60)
    args = parser.parse_args(argv)
    config, demos, targets = load_inputs()
    items = schedule(config, demos, targets)
    if not args.live:
        print("预览模式：dev=20；条件=C0/C1/C2；计划=60组。")
        for target, condition, selected, _ in items[:3]:
            print(target["id"], condition, [demo["id"] for demo in selected])
        print("未读取密钥、未写入运行文件、未读取test、未调用API。")
        return 0
    if not 1 <= args.max_new_requests <= 180:
        raise ValueError("max-new-requests 必须在 1 到 180 之间。")
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key or any(char.isspace() for char in key):
        raise ValueError("当前终端缺少有效的 DEEPSEEK_API_KEY。")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.run_id):
        raise ValueError("run-id 格式不合法。")
    parent = ROOT / "runs"
    parent.mkdir(exist_ok=True)
    runner = LiveRunner(parent / args.run_id, args.run_id, config,
                        demos, targets, create=not args.resume)
    print("真实dev运行；目录：", runner.run_dir, flush=True)
    print("本次最多新增API尝试：", args.max_new_requests, flush=True)
    result = runner.run(send_real, max_new_requests=args.max_new_requests)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("运行已中断；保留运行目录，下次使用相同run-id和--resume核查。")
        raise SystemExit(130)
    except (ValueError, FileExistsError) as exc:
        print("停止：", str(exc))
        raise SystemExit(1)
    except Exception as exc:
        print("停止：异常类型", type(exc).__name__, "；请保留日志并核查，勿新建运行ID重发。")
        raise SystemExit(1)
