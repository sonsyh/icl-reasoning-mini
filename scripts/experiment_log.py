"""串行dev实验日志；不自动调用API或创建正式日志。"""

import copy
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from answer_parser import PARSER_VERSION, score_answer
from build_prompts import PROMPT_VERSION, build_messages, hash_messages
from deepseek_client import build_request
from select_random_demos import ALGORITHM_VERSION, ROOT


ARTIFACT_PATHS = (
    "configs/inference_dev_v1.json",
    "data/splits/seed20261007/demo.jsonl",
    "data/splits/seed20261007/dev.jsonl",
    "data/splits/seed20261007/freeze_record.json",
    "scripts/select_random_demos.py",
    "scripts/build_prompts.py",
    "scripts/answer_parser.py",
    "scripts/deepseek_client.py",
    "scripts/experiment_log.py",
)

ERROR_KINDS = {
    "timeout", "connection", "rate_limit", "server_error",
    "authentication", "permission", "invalid_request", "unknown_error",
}

USAGE_FIELDS = (
    "prompt_tokens", "completion_tokens", "total_tokens",
    "prompt_cache_hit_tokens", "prompt_cache_miss_tokens",
)


def json_text(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def hash_json(value):
    return hashlib.sha256(json_text(value).encode("utf-8")).hexdigest()


def collect_artifact_hashes():
    """未来运行前调用；只计算列出的文件指纹，不读取test文件。"""
    return {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ARTIFACT_PATHS
    }


def build_record(
    *,
    run_id,
    config,
    target,
    condition,
    demos,
    messages,
    artifact_hashes,
    response=None,
    error_kind=None,
    error_token_usage=None,
    attempt_no=1,
    is_simulated=True,
):
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id必须是非空字符串。")

    if condition not in config["conditions"]:
        raise ValueError("实验条件不在配置中。")

    if config["runtime"]["concurrency"] != 1:
        raise ValueError("本版日志模块只支持串行运行。")

    max_attempts = 1 + config["runtime"]["transient_error_retries"]
    if type(attempt_no) is not int or not 1 <= attempt_no <= max_attempts:
        raise ValueError("尝试次数超出配置范围。")

    if type(is_simulated) is not bool:
        raise ValueError("is_simulated必须是布尔值。")

    if set(artifact_hashes) != set(ARTIFACT_PATHS):
        raise ValueError("文件指纹列表不完整或包含额外文件。")

    for digest in artifact_hashes.values():
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("文件指纹必须是小写SHA-256。")

    expected_messages = build_messages(target, condition, demos)
    if messages != expected_messages:
        raise ValueError("提示词与本版模板或示例不一致。")

    request_parameters = build_request(config, messages)
    request_parameters.pop("messages")

    if response is None:
        if error_kind not in ERROR_KINDS:
            raise ValueError("没有响应时，必须提供允许的错误类别。")
    elif not isinstance(response, dict) or error_kind is not None:
        raise ValueError("响应与错误类别不能同时提供。")
    if error_token_usage is not None and (
        response is not None or is_simulated is not True
    ):
        raise ValueError("错误用量只允许作为已确认的模拟错误证据提供。")

    prompt_hash = hash_messages(messages)
    config_hash = hash_json(config)

    logical_request_id = hash_json([
        run_id, target["id"], condition,
        config_hash, prompt_hash, hash_json(artifact_hashes),
    ])

    score = None
    usage = None

    original_usage = error_token_usage
    if response is not None:
        score = score_answer(
            response["raw_output"],
            target["label"],
            finish_reason=response["finish_reason"],
        )
        original_usage = response.get("token_usage")
    if original_usage is not None:
        if not isinstance(original_usage, dict):
            raise ValueError("token_usage必须是字典或None。")
        usage = {
            key: original_usage[key]
            for key in USAGE_FIELDS
            if key in original_usage
        }
        for value in usage.values():
            if value is not None and (
                type(value) is not int or value < 0
            ):
                raise ValueError("token用量必须是非负整数或None。")

    record = {
        "log_schema_version": 1,
        "run_id": run_id,
        "logical_request_id": logical_request_id,
        "attempt_no": attempt_no,
        "is_simulated": is_simulated,
        "stage": "dev",
        "question_id": target["id"],
        "condition": condition,
        "demo_ids": [demo["id"] for demo in demos],
        "seed": config["demo_selection"]["seed"],
        "depth": target["depth"],
        "gold_label": target["label"],
        "model_id": config["api"]["model"],
        "documented_model_version": config.get("provenance", {}).get(
            "provider_documented_version"
        ),
        "base_url": config["api"]["base_url"],
        "endpoint": config["api"]["endpoint"],
        "inference_parameters": request_parameters,
        "runtime": {
            key: config["runtime"][key]
            for key in (
                "timeout_seconds", "concurrency",
                "transient_error_retries", "sdk_max_retries",
            )
        },
        "prompt_version": PROMPT_VERSION,
        "selection_version": ALGORITHM_VERSION,
        "parser_version": PARSER_VERSION,
        "prompt": copy.deepcopy(messages),
        "prompt_hash": prompt_hash,
        "config_hash": config_hash,
        "artifact_hashes": copy.deepcopy(artifact_hashes),
        "status": (
            "api_error" if response is None
            else "completed" if score["completion_ok"]
            else "incomplete"
        ),
        "error_kind": error_kind,
        "raw_output": response.get("raw_output") if response is not None else None,
        "finish_reason": (
            response.get("finish_reason") if response is not None else None
        ),
        "response_model": response.get("model") if response is not None else None,
        "response_id": (
            response.get("response_id") if response is not None else None
        ),
        "latency_seconds": (
            response.get("latency_seconds") if response is not None else None
        ),
        "token_usage": usage,
        "cost_estimate_cny": None,
        "score": score,
        "parsed_answer": score["lenient_label"] if score else None,
        "strict_format_ok": score["strict_format_ok"] if score else None,
        "correct": score["lenient_correct"] if score else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    record["record_hash"] = hash_json(record)
    return record


def validate_record(record):
    if record.get("log_schema_version") != 1:
        raise ValueError("不支持的日志版本。")

    payload = dict(record)
    saved_hash = payload.pop("record_hash", None)

    if saved_hash != hash_json(payload):
        raise ValueError("记录指纹不一致。")

    if record["prompt_hash"] != hash_messages(record["prompt"]):
        raise ValueError("提示词指纹不一致。")


def create_log(path):
    path = Path(path)
    if path.suffix != ".jsonl":
        raise ValueError("日志文件必须使用.jsonl扩展名。")

    # x模式：文件已存在就报错，绝不覆盖。
    with path.open("x", encoding="utf-8"):
        pass

    return path


def read_records(path):
    text = Path(path).read_text(encoding="utf-8")
    if text and not text.endswith("\n"):
        raise ValueError("日志尾行不完整，请停止，不要自动修复。")

    records = []
    for line in text.splitlines():
        record = json.loads(line)
        validate_record(record)
        records.append(record)

    return records


def append_record(path, record):
    validate_record(record)
    existing = read_records(path)

    for previous in existing:
        for key in (
            "run_id", "config_hash", "artifact_hashes", "is_simulated",
        ):
            if previous[key] != record[key]:
                raise ValueError("不能混写不同运行、配置、文件或模拟状态。")

        if previous["logical_request_id"] == record["logical_request_id"]:
            if previous["attempt_no"] >= record["attempt_no"]:
                raise ValueError("尝试编号重复或倒退。")
            if previous["status"] != "api_error":
                raise ValueError("已收到回答，不允许为改善内容再次作答。")

    line = json_text(record) + "\n"

    # r+要求文件已经存在；定位到末尾追加，不截断旧内容。
    with Path(path).open("r+", encoding="utf-8") as file:
        file.seek(0, os.SEEK_END)
        file.write(line)
        file.flush()
        os.fsync(file.fileno())
