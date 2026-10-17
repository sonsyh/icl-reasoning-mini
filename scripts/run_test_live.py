"""Independent formal test runner: 60 questions x C0-C3. Default read-only."""
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
import build_test_prompts as tp

VERSION = "formal_test_runner_v1"
CONDITIONS = ("C0", "C1", "C2", "C3")
EXPECTED_CONFIG_HASH = "4962f0e6ccb1006ddaede673ca16338e8a0cfaf1c91224b9ad116375d47de00c"
CONFIG_PATH = "configs/inference_test_v1.json"
PROMPT_MANIFEST_HASH = "39c7807876a0375ad13463139504e1580b5ba334ce3e2d05a40cfcb70c36a78c"
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


def validate_config(config):
    if digest(config) != EXPECTED_CONFIG_HASH:
        raise ValueError("正式test配置与已验收设置不同。")


def request_once(config, messages, *, client_factory=None):
    if client_factory is None:
        raise RuntimeError("真实请求必须显式传入客户端工厂。")
    validate_config(config)
    api = config["api"]
    # Exact same request fields as the fixed dev client; stage is genuinely test.
    request = {"model": api["model"], "messages": copy.deepcopy(messages),
               "temperature": api["temperature"], "max_tokens": api["max_tokens"],
               "stream": api["stream"], "response_format": copy.deepcopy(api["response_format"]),
               "extra_body": {"thinking": copy.deepcopy(api["thinking"])}}
    key = os.environ.get(config["credentials"]["api_key_env"], "")
    if not key or any(c.isspace() for c in key):
        raise ValueError("密钥缺失或含空白字符。")
    begin = time.perf_counter()
    with client_factory(api_key=key, base_url=api["base_url"],
                        timeout=config["runtime"]["timeout_seconds"], max_retries=0) as client:
        response = client.chat.completions.create(**request)
    choice = response.choices[0]
    return {"raw_output": choice.message.content, "finish_reason": choice.finish_reason,
            "model": response.model, "response_id": response.id,
            "token_usage": response.usage.model_dump() if response.usage is not None else None,
            "latency_seconds": time.perf_counter() - begin}


def load_plan(root):
    root = Path(root)
    config, baseline, demos, targets, chosen, paths, captured = tp.load_bundle(root)
    validate_config(config)
    folder = root / tp.OUTPUT_PATH
    manifest_path, prompts_path = folder / "prompt_manifest.json", folder / "prompts.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_signed(manifest, "manifest_hash")
    expected_provenance = {
        "config": config, "input_hashes": captured,
        "selection_manifest_hash": tp.SELECTION_HASH,
        "counting_mode": "DeepSeek-V4 chat; drop_thinking=True; BOS=True; tokenizer special_tokens=False"}
    if (manifest["manifest_hash"] != PROMPT_MANIFEST_HASH
            or manifest["status"] != "complete" or manifest["stage"] != "test"
            or manifest["question_count"] != 60 or manifest["prompt_count"] != 240
            or manifest["builder_version"] != tp.VERSION
            or manifest["provenance"] != expected_provenance
            or manifest["prompts_sha256"] != cp.be.file_hash(prompts_path)):
        raise ValueError("正式提示词缓存或来源指纹改变。")
    prompts = read_jsonl(prompts_path)
    for cached in prompts:
        tp.validate_record(cached)
    counts = {}
    for cached in prompts:
        h, n = cached["prompt_hash"], cached["input_tokens_offline"]
        if h in counts and counts[h] != n:
            raise ValueError("同一消息的保存token计数冲突。")
        counts[h] = n
    def count(messages):
        h = hash_messages(messages)
        if h not in counts:
            raise ValueError("重建出现未缓存消息。")
        return counts[h]
    if prompts != tp.make_records(config, baseline, demos, targets, chosen, count):
        raise ValueError("正式提示词与固定方法重建不一致。")
    if len(prompts) != 240 or manifest["token_report"] != tp.token_report(prompts):
        raise ValueError("提示词数量或长度报告不一致。")
    # Gold is isolated in the local scorer context, never sent to client/selector.
    raw = tp.read_jsonl(root / config["data"]["eval_path"])
    gold = {row["id"]: row["label"] for row in raw}
    if len(gold) != 60 or any(label not in ("True", "False") for label in gold.values()):
        raise ValueError("本地评分标签不合法。")
    items = [{"cache": cached, "gold_label": gold[cached["question_id"]]} for cached in prompts]
    paths = dict(paths)
    paths.update({"test_prompt_manifest": manifest_path, "test_prompts": prompts_path,
                  "test_runner_source": Path(__file__).resolve()})
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


class TestRunner:
    def __init__(self, run_dir, run_id, config, items, paths, fingerprints, env,
                 *, create=False, simulated=False):
        if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id)
                or not run_id.startswith("day5_test_") or Path(run_dir).name != run_id):
            raise ValueError("test运行ID须以day5_test_开头，并使用独立目录。")
        validate_config(config)
        if config["conditions"] != list(CONDITIONS) or len(items) != 240:
            raise ValueError("要求60题×四条件的240组正式请求。")
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
            if item["gold_label"] not in ("True", "False"):
                raise ValueError("本地评分标签不合法。")
            request_id = digest({"context_hash": self.context_hash,
                                 "question_id": cache["question_id"], "condition": cache["condition"]})
            logical_key = (cache["question_id"], cache["condition"])
            if logical_key in self.expected:
                raise ValueError("重复目标题。")
            self.expected[logical_key] = (request_id, item)
        question_ids = sorted({key[0] for key in self.expected})
        if len(question_ids) != 60 or list(self.expected) != [
                (qid, condition) for qid in question_ids for condition in CONDITIONS]:
            raise ValueError("计划不是固定顺序的60题×四条件。")
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
        result = {"record_schema_version": "formal_test_record_v1", "timestamp": now(),
                  "context_hash": self.context_hash, "run_id": self.context["run_id"],
                  "is_simulated": self.context["is_simulated"], "stage": "test", "condition": cache["condition"],
                  "question_id": cache["question_id"],
                  "logical_request_id": self.expected[(cache["question_id"], cache["condition"])][0],
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
            qid = (row.get("question_id"), row.get("condition"))
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
                    row["response"], self.expected[(row["question_id"], row["condition"])][1]["cache"]):
                raise ValueError("已保存响应的模型或token用量异常；停止进一步请求。")
        if pending is not None:
            self.event("recorded", pending, results[pending]["record_hash"])
        return rows, grouped

    def summary(self, rows, sent):
        responses = [r for r in rows if r["status"] == "response"]
        conditions = {}
        for condition in CONDITIONS:
            subset = [r for r in responses if r["condition"] == condition]
            conditions[condition] = {
                "planned": 60, "recorded": len(subset),
                "strict_correct": sum(r["score"]["strict_correct"] for r in subset),
                "lenient_correct": sum(r["score"]["lenient_correct"] for r in subset)}
        return {"new_api_attempts": sent, "recorded_responses": len(responses),
                "planned_responses": 240, "all_responses_recorded": len(responses) == 240,
                "api_error_attempts": len(rows) - len(responses), "conditions": conditions}

    def run(self, send, *, max_new_requests=1, checkpoint=None, sleep=time.sleep):
        if type(max_new_requests) is not int or not 1 <= max_new_requests <= 720:
            raise ValueError("max-new-requests必须为1到720。")
        sent = 0
        with self.lock.open("r+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("同一运行目录已有进程使用。") from None
            try:
                self.history()
                for item in self.items:
                    qid = (item["cache"]["question_id"], item["cache"]["condition"])
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
                            print(f"已保存：{qid[0]} {qid[1]} | strict_correct={record['score']['strict_correct']}"
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
    parser.add_argument("--run-id", default="day5_test_live_v1_20261013")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-requests", type=int, default=1)
    args = parser.parse_args(argv)
    config, items, paths, fingerprints = load_plan(ROOT)
    env = environment()
    if not args.live:
        print("只读预览：test=60；条件=C0/C1/C2/C3；计划240组；固定缓存重建及参数核对通过。")
        for item in items[:4]:
            cache = item["cache"]
            print(cache["question_id"], cache["condition"], cache["demo_ids"],
                  "input_tokens_offline=" + str(cache["input_tokens_offline"]))
        print("SDK版本：", env["packages"]["openai"], "；Pydantic：", env["packages"]["pydantic"])
        print("未读取密钥、未写运行文件、未展示test题面/答案、未加载模型权重、未调用API。")
        return 0
    if not 1 <= args.max_new_requests <= 720:
        raise ValueError("max-new-requests必须为1到720。")
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key or any(c.isspace() for c in key):
        raise ValueError("当前终端缺少有效的DEEPSEEK_API_KEY。")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.run_id) or not args.run_id.startswith("day5_test_"):
        raise ValueError("run-id须以day5_test_开头，不能使用旧实验目录。")
    parent = ROOT / "runs"
    parent.mkdir(exist_ok=True)
    runner = TestRunner(parent / args.run_id, args.run_id, config, items, paths, fingerprints,
                      env, create=not args.resume)
    print("真实test运行；目录：", runner.run_dir, flush=True)
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
