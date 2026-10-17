"""Independent C3 dev run. Default is read-only; --live explicitly enables API."""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_c3_prompts as cp
from answer_parser import score_answer
from build_prompts import hash_messages
from deepseek_client import build_request, request_once, validate_config

VERSION = "c3_live_runner_v1"
CONFIG_PATH = "configs/inference_dev_c3_v1.json"
PROMPT_MANIFEST_HASH = "c7bc07702456200c430918dd74ac2a8d235b27dc615e9328e974686645c8db00"
ERRORS = {"rate_limit", "timeout", "connection", "authentication", "permission",
          "server_error", "invalid_request", "unknown_error"}
SDK_PACKAGES = ("openai", "httpx2", "httpcore2", "jiter", "pydantic",
                "pydantic-core", "anyio", "sniffio", "truststore", "typing-extensions")


def digest(value):
    return cp.be.digest_text(cp.be.canonical(value))


def now():
    return datetime.now(timezone.utc).isoformat()


def signed(payload, field):
    result = copy.deepcopy(payload)
    result[field] = digest(payload)
    return result


def verify_signed(value, field):
    payload = dict(value)
    saved = payload.pop(field, None)
    if saved != digest(payload):
        raise ValueError("记录指纹不一致。")


def read_jsonl(path):
    text = Path(path).read_text(encoding="utf-8")
    if text and not text.endswith("\n"):
        raise ValueError("日志尾行不完整；停止，不自动修复。")
    try:
        rows = [json.loads(line) for line in text.splitlines()]
    except (ValueError, TypeError):
        raise ValueError("日志JSON损坏；停止，不自动修复。") from None
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("日志记录必须是对象。")
    return rows


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as file:
        file.write(cp.be.canonical(value) + "\n")
        file.flush()
        os.fsync(file.fileno())


def append(path, value):
    # Never creates a missing log or overwrites earlier rows.
    with Path(path).open("r+", encoding="utf-8") as file:
        file.seek(0, os.SEEK_END)
        file.write(cp.be.canonical(value) + "\n")
        file.flush()
        os.fsync(file.fileno())


def environment():
    packages = {name: version(name) for name in (*cp.be.PACKAGES, *SDK_PACKAGES)}
    if any(packages[name] != expected for name, expected in cp.be.PACKAGES.items()):
        raise ValueError("检索环境依赖版本改变。")
    if packages["openai"] != "3.26.0":
        raise ValueError("要求已验收的openai==3.26.0。")
    if Path(sys.prefix).name != "icl-reasoning-retrieval":
        raise ValueError("请激活icl-reasoning-retrieval环境。")
    if os.environ.get("KMP_DUPLICATE_LIB_OK"):
        raise ValueError("不能启用OpenMP重复加载绕过开关。")
    return {"python": platform.python_version(), "executable": sys.executable,
            "platform": platform.platform(), "packages": packages}


def load_plan(root):
    root = Path(root)
    config = json.loads((root / CONFIG_PATH).read_text(encoding="utf-8"))
    baseline, groups, selections, c2, paths, captured = cp.load_bundle(root)
    validate_config(config)
    expected = {
        "config_schema_version": 1, "config_name": "day4_dev_c3_v1",
        "stage": "dev", "conditions": ["C3"],
        "api": baseline["api"], "runtime": baseline["runtime"],
        "demo_selection": {
            "applies_to_conditions": ["C3"], "method": cp.sr.VERSION,
            "shots": 2, "true_count": 1, "false_count": 1,
            "label_order": ["True", "False"],
            "selection_cache_path": cp.sr.OUTPUT_PATH,
            "selection_manifest_hash": cp.SELECTION_HASH},
        "data": baseline["data"],
        "prompt_cache": {"path": cp.OUTPUT_PATH,
                         "manifest_hash": PROMPT_MANIFEST_HASH, "expected_count": 20},
        "baseline": {"config_path": cp.INFERENCE_PATH, "results_path": cp.DAY3_PATH,
                     "results_sha256": cp.DAY3_HASH},
        "budget": {"currency": "CNY", "project_total_cap": 100},
        "credentials": baseline["credentials"],
        "provenance": {"baseline_condition": "C2",
                       "comparison": "semantic_vs_random_demo_selection"},
    }
    if config != expected:
        raise ValueError("C3配置不符合已验收设置或与C2参数不同。")
    folder = root / cp.OUTPUT_PATH
    manifest_path, prompts_path = folder / "prompt_manifest.json", folder / "prompts.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_signed(manifest, "manifest_hash")
    if (manifest["manifest_hash"] != PROMPT_MANIFEST_HASH
            or manifest["status"] != "complete" or manifest["question_count"] != 20
            or manifest["provenance"]["input_hashes"] != captured
            or manifest["prompts_sha256"] != cp.be.file_hash(prompts_path)):
        raise ValueError("C3提示词缓存或来源指纹改变。")
    prompts = read_jsonl(prompts_path)
    targets = {row["id"]: row for row in groups["dev"]}
    demos = {row["id"]: row for row in groups["demo"]}
    selection_map = {row["question_id"]: row for row in selections}
    if len(prompts) != 20 or [row["question_id"] for row in prompts] != sorted(targets):
        raise ValueError("C3提示词数量、题目集合或顺序错误。")
    items = []
    for cached in prompts:
        cp.validate_prompt_record(cached)
        target = targets[cached["question_id"]]
        selection = selection_map[target["id"]]
        chosen = [demos[qid] for qid in selection["demo_ids"]]
        # Token counts are already in the immutable, pinned cache; no tokenizer/model load.
        messages = cp.build_c3_messages(target, chosen)
        def count(msg):
            if msg == messages:
                return cached["input_tokens_offline"]
            if msg == c2[target["id"]]["prompt"]:
                return cached["baseline_c2"]["input_tokens_offline"]
            raise ValueError("未知提示词。")
        rebuilt = cp.audit_case(target, chosen, selection, c2[target["id"]], baseline, count)
        if rebuilt != cached or build_request(config, messages) != build_request(baseline, messages):
            raise ValueError("C3提示词或API请求参数重建不一致。")
        items.append({"cache": cached, "gold_label": target["label"]})
    paths = dict(paths)
    paths.update({"c3_config": root / CONFIG_PATH, "c3_prompt_manifest": manifest_path,
                  "c3_prompts": prompts_path, "c3_runner_source": Path(__file__).resolve()})
    return config, items, paths, cp.be.snapshot(paths)


class LiveAPIError(Exception):
    def __init__(self, kind):
        if kind not in ERRORS:
            raise ValueError("未知API错误类别。")
        self.kind = kind
        super().__init__(kind)


def send_real(config, messages):
    from openai import OpenAI, APIStatusError, APITimeoutError, APIConnectionError
    try:
        return request_once(config, messages, client_factory=OpenAI)
    except APITimeoutError:
        raise LiveAPIError("timeout") from None
    except APIConnectionError:
        raise LiveAPIError("connection") from None
    except APIStatusError as exc:
        status = exc.status_code
        kind = ("rate_limit" if status == 429 else "authentication" if status == 401
                else "permission" if status == 403 else "server_error" if status >= 500
                else "invalid_request")
        # Do not persist SDK exception bodies, headers or credentials.
        raise LiveAPIError(kind) from None


class C3Runner:
    def __init__(self, run_dir, run_id, config, items, paths, fingerprints, env,
                 *, create=False, simulated=False):
        if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id)
                or run_id.startswith("day3") or Path(run_dir).name != run_id):
            raise ValueError("运行ID不合法，或试图使用Day3目录。")
        validate_config(config)
        if config["conditions"] != ["C3"] or len(items) != 20:
            raise ValueError("要求20组独立C3请求。")
        self.config, self.items = copy.deepcopy(config), copy.deepcopy(items)
        self.paths, self.fingerprints = dict(paths), dict(fingerprints)
        self.run_dir = Path(run_dir)
        self.manifest = self.run_dir / "run_manifest.json"
        self.results = self.run_dir / "results.jsonl"
        self.journal = self.run_dir / "attempts.jsonl"
        self.lock = self.run_dir / "run.lock"
        self.context = {"runner_version": VERSION, "run_id": run_id,
                        "is_simulated": simulated, "config": self.config,
                        "artifact_hashes": self.fingerprints, "environment": copy.deepcopy(env),
                        "schedule": self.items}
        self.context_hash = digest(self.context)
        self.expected = {}
        for item in self.items:
            cache = item["cache"]
            request_id = digest({"context_hash": self.context_hash,
                                 "question_id": cache["question_id"], "condition": "C3"})
            if cache["question_id"] in self.expected:
                raise ValueError("重复目标题。")
            self.expected[cache["question_id"]] = (request_id, item)
        if create:
            self.check_inputs()
            self.run_dir.mkdir(exist_ok=False)
            write_new(self.manifest, self.context)
            for path in (self.results, self.journal, self.lock):
                with path.open("x", encoding="utf-8"):
                    pass
        self.check_context()

    def check_inputs(self):
        cp.be.verify_snapshot(self.paths, self.fingerprints)
        if self.config != self.context["config"] or self.items != self.context["schedule"]:
            raise ValueError("运行期间内存配置或提示词改变。")

    def check_context(self):
        if self.run_dir.is_symlink() or any(path.is_symlink() or not path.is_file()
                for path in (self.manifest, self.results, self.journal, self.lock)):
            raise ValueError("运行目录不完整或含符号链接，停止。")
        if json.loads(self.manifest.read_text(encoding="utf-8")) != self.context:
            raise ValueError("运行身份、配置、数据、源码或环境改变。")
        self.check_inputs()

    def record(self, item, attempt, response=None, error=None):
        cache = item["cache"]
        result = {"record_schema_version": "c3_live_record_v1", "timestamp": now(),
                  "context_hash": self.context_hash, "run_id": self.context["run_id"],
                  "is_simulated": self.context["is_simulated"], "stage": "dev", "condition": "C3",
                  "question_id": cache["question_id"],
                  "logical_request_id": self.expected[cache["question_id"]][0],
                  "attempt_no": attempt, "config_hash": digest(self.config),
                  "prompt_cache_record_hash": cache["record_hash"],
                  "prompt_hash": cache["prompt_hash"], "prompt": cache["prompt"],
                  "demo_ids": cache["demo_ids"], "inference_parameters": cache["request_parameters"],
                  "input_tokens_offline": cache["input_tokens_offline"],
                  "gold_label": item["gold_label"],
                  "status": "api_error" if error else "response", "error_kind": error,
                  "response": None, "score": None}
        if response is not None:
            fields = ("raw_output", "finish_reason", "model", "response_id",
                      "token_usage", "latency_seconds")
            result["response"] = {name: copy.deepcopy(response.get(name)) for name in fields}
            result["score"] = score_answer(response.get("raw_output"), item["gold_label"],
                                           finish_reason=response.get("finish_reason"))
        return signed(result, "record_hash")

    def read_events(self):
        events, previous = read_jsonl(self.journal), None
        for event in events:
            verify_signed(event, "event_hash")
            if (set(event) != {"state", "key", "result_record_hash", "context_hash",
                               "previous_event_hash", "timestamp", "event_hash"}
                    or event["context_hash"] != self.context_hash
                    or event["previous_event_hash"] != previous):
                raise ValueError("尝试日志上下文或顺序错误。")
            previous = event["event_hash"]
        return events

    def event(self, state, key, result_hash=None):
        events = self.read_events()
        append(self.journal, signed({"state": state, "key": list(key),
               "result_record_hash": result_hash, "context_hash": self.context_hash,
               "previous_event_hash": events[-1]["event_hash"] if events else None,
               "timestamp": now()}, "event_hash"))

    def response_ok(self, response, cache):
        usage = response["token_usage"]
        latency = response["latency_seconds"]
        return (response["model"] == self.config["api"]["model"]
                and isinstance(response["response_id"], str) and bool(response["response_id"])
                and type(latency) in (int, float) and math.isfinite(latency) and latency >= 0
                and isinstance(usage, dict)
                and all(type(usage.get(k)) is int and usage[k] >= 0
                        for k in ("prompt_tokens", "completion_tokens", "total_tokens"))
                and usage["prompt_tokens"] == cache["input_tokens_offline"]
                and usage["completion_tokens"] <= self.config["api"]["max_tokens"]
                and usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"])

    def history(self):
        self.check_context()
        rows = read_jsonl(self.results)
        grouped = {qid: [] for qid in self.expected}
        results, previous_index = {}, -1
        order = list(self.expected)
        for row in rows:
            verify_signed(row, "record_hash")
            qid = row.get("question_id")
            if qid not in self.expected or type(row.get("attempt_no")) is not int:
                raise ValueError("结果题目或尝试编号错误。")
            rid, item = self.expected[qid]
            attempt = row["attempt_no"]
            expected = self.record(item, attempt, row.get("response"), row.get("error_kind"))
            expected["timestamp"] = row["timestamp"]
            expected["record_hash"] = digest({k: v for k, v in expected.items() if k != "record_hash"})
            if row != expected:
                raise ValueError("结果与固定提示词、评分或上下文不一致。")
            index = order.index(qid)
            if index < previous_index or index > previous_index + 1:
                raise ValueError("结果不符合固定串行顺序。")
            if index > previous_index and previous_index >= 0:
                if grouped[order[previous_index]][-1]["status"] != "response":
                    raise ValueError("前一题没有响应却跳到下一题。")
            previous_index = index
            prior = grouped[qid]
            if (attempt != len(prior) + 1 or attempt > 3
                    or (prior and prior[-1]["error_kind"] != "rate_limit")):
                raise ValueError("重复请求、非法重试或编号不连续。")
            if row["status"] == "api_error":
                if row["error_kind"] not in ERRORS or row["response"] is not None:
                    raise ValueError("API错误记录结构错误。")
            elif row["status"] != "response" or row["response"] is None or row["error_kind"] is not None:
                raise ValueError("响应记录结构错误。")
            prior.append(row)
            results[(rid, attempt)] = row
        counts, pending, finished, starts = {}, None, {}, []
        allowed = {rid for rid, _ in self.expected.values()}
        for event in self.read_events():
            key = event["key"]
            if (not isinstance(key, list) or len(key) != 2 or key[0] not in allowed
                    or type(key[1]) is not int or not 1 <= key[1] <= 3):
                raise ValueError("尝试日志请求编号错误。")
            key = tuple(key)
            if event["state"] == "started":
                if (pending is not None or key[1] != counts.get(key[0], 0) + 1
                        or event["result_record_hash"] is not None):
                    raise ValueError("尝试日志开始标记错误。")
                counts[key[0]] = key[1]
                pending = key
                starts.append(key)
            elif event["state"] == "recorded":
                if pending != key or key in finished or key not in results:
                    raise ValueError("尝试日志缺少对应结果。")
                if event["result_record_hash"] != results[key]["record_hash"]:
                    raise ValueError("完成标记与结果指纹不同。")
                finished[key], pending = event["result_record_hash"], None
            else:
                raise ValueError("未知尝试状态。")
        if set(results) != set(finished) | ({pending} if pending in results else set()):
            raise ValueError("结果缺少对应开始标记。")
        if pending is not None and pending not in results:
            raise ValueError("存在已开始但结果不明的请求；不自动重发。")
        if starts != list(results):
            raise ValueError("尝试日志与结果的串行顺序不同。")
        # Validate all records before performing the only permitted reconciliation.
        for row in rows:
            if row["status"] == "api_error" and row["error_kind"] != "rate_limit":
                raise ValueError("历史API错误不自动重试；结果可能不确定，请保留日志核查。")
            if row["status"] == "response" and not self.response_ok(
                    row["response"], self.expected[row["question_id"]][1]["cache"]):
                raise ValueError("已保存响应的模型或token用量异常；停止进一步请求。")
        if pending is not None:
            self.event("recorded", pending, results[pending]["record_hash"])
        return rows, grouped

    def summary(self, rows, sent):
        responses = [r for r in rows if r["status"] == "response"]
        return {"new_api_attempts": sent, "recorded_responses": len(responses),
                "planned_responses": 20, "all_responses_recorded": len(responses) == 20,
                "api_error_attempts": len(rows) - len(responses), "conditions": {"C3": {
                    "planned": 20, "recorded": len(responses),
                    "strict_correct": sum(r["score"]["strict_correct"] for r in responses),
                    "lenient_correct": sum(r["score"]["lenient_correct"] for r in responses)}}}

    def run(self, send, *, max_new_requests=1, checkpoint=None, sleep=time.sleep):
        if type(max_new_requests) is not int or not 1 <= max_new_requests <= 60:
            raise ValueError("max-new-requests必须为1到60。")
        sent = 0
        with self.lock.open("r+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("同一运行目录已有进程使用。") from None
            try:
                self.history()
                for item in self.items:
                    qid = item["cache"]["question_id"]
                    while True:
                        rows, grouped = self.history()
                        prior = grouped[qid]
                        if prior and prior[-1]["status"] == "response":
                            break  # Wrong/ambiguous/incomplete answers are observations, never retries.
                        if len(prior) >= 3:
                            raise ValueError("429重试上限已达；停止。")
                        if sent >= max_new_requests:
                            return self.summary(rows, sent)
                        if prior:
                            sleep(2 ** len(prior))
                        self.check_context()
                        key = (self.expected[qid][0], len(prior) + 1)
                        self.event("started", key)
                        if checkpoint:
                            checkpoint("after_start")
                        sent += 1
                        response, error = None, None
                        try:
                            response = send(copy.deepcopy(self.config), copy.deepcopy(item["cache"]["prompt"]))
                        except LiveAPIError as exc:
                            error = exc.kind
                        if response is None and error is None:
                            # Unknown return value: leave the start unresolved; never resend.
                            raise ValueError("调用未返回可保存响应；保留日志，不自动重发。")
                        if checkpoint:
                            checkpoint("after_response")
                        record = self.record(item, key[1], response, error)
                        append(self.results, record)
                        if checkpoint:
                            checkpoint("after_log")
                        self.event("recorded", key, record["record_hash"])
                        self.history()
                        if response is not None:
                            print(f"已保存：{qid} C3 | strict_correct={record['score']['strict_correct']}"
                                  f" | tokens={record['response']['token_usage']['total_tokens']}", flush=True)
                        else:
                            print("已保存API错误类别：", error, flush=True)
                return self.summary(self.history()[0], sent)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--live", action="store_true")
    parser.add_argument("--run-id", default="day4_c3_dev_live_v1_20261012")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-requests", type=int, default=1)
    args = parser.parse_args(argv)
    config, items, paths, fingerprints = load_plan(ROOT)
    env = environment()
    if not args.live:
        print("只读预览：C3；dev=20；计划20组；固定缓存重建及C2参数核对通过。")
        for item in items[:3]:
            cache = item["cache"]
            print(cache["question_id"], "C3", cache["demo_ids"],
                  "input_tokens_offline=" + str(cache["input_tokens_offline"]))
        print("SDK版本：", env["packages"]["openai"], "；Pydantic：", env["packages"]["pydantic"])
        print("未读取密钥、未写运行文件、未读取test、未加载模型权重、未调用API。")
        return 0
    if not 1 <= args.max_new_requests <= 60:
        raise ValueError("max-new-requests必须为1到60。")
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key or any(c.isspace() for c in key):
        raise ValueError("当前终端缺少有效的DEEPSEEK_API_KEY。")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.run_id) or args.run_id.startswith("day3"):
        raise ValueError("run-id不合法或属于Day3。")
    parent = ROOT / "runs"
    parent.mkdir(exist_ok=True)
    runner = C3Runner(parent / args.run_id, args.run_id, config, items, paths, fingerprints,
                      env, create=not args.resume)
    print("真实C3 dev运行；目录：", runner.run_dir, flush=True)
    print("本次最多新增API尝试：", args.max_new_requests, flush=True)
    result = runner.run(send_real, max_new_requests=args.max_new_requests)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("已中断；保留运行目录，使用相同run-id及--resume核查，不要换ID重发。")
        raise SystemExit(130)
    except (ValueError, FileExistsError) as exc:
        print("停止：", str(exc))
        raise SystemExit(1)
    except Exception as exc:
        # Do not print arbitrary exception text that might contain secrets.
        print("停止：异常类型", type(exc).__name__, "；保留日志核查，不要换ID重发。")
        raise SystemExit(1)
