"""重试与续跑判断v1；只返回计划，不发送请求或等待。"""

from answer_parser import score_answer
from build_prompts import hash_messages
from experiment_log import ERROR_KINDS, hash_json, validate_record


POLICY_VERSION = "retry_resume_policy_v1"

TRANSIENT_ERRORS = {
    "timeout", "connection", "rate_limit", "server_error",
}

GLOBAL_KEYS = (
    "run_id", "stage", "config_hash", "artifact_hashes",
    "is_simulated", "model_id", "documented_model_version",
    "base_url", "endpoint", "inference_parameters", "runtime",
    "prompt_version", "selection_version", "parser_version", "seed",
)

TARGET_KEYS = (
    "logical_request_id", "prompt_hash", "prompt",
    "demo_ids", "gold_label", "depth",
)


def check_identity(record):
    validate_record(record)

    expected_id = hash_json([
        record["run_id"],
        record["question_id"],
        record["condition"],
        record["config_hash"],
        record["prompt_hash"],
        hash_json(record["artifact_hashes"]),
    ])

    if record["logical_request_id"] != expected_id:
        raise ValueError("逻辑请求编号与记录内容不一致。")

    if record["prompt_hash"] != hash_messages(record["prompt"]):
        raise ValueError("提示词指纹不一致。")


def check_outcome(record):
    status = record["status"]

    if status == "api_error":
        if record["error_kind"] not in ERROR_KINDS:
            raise ValueError("未知错误类别。")
        if record["raw_output"] is not None or record["score"] is not None:
            raise ValueError("API错误记录不能同时包含已收到的回答。")
        return

    if status not in ("completed", "incomplete"):
        raise ValueError("未知日志状态。")

    if record["error_kind"] is not None:
        raise ValueError("回答记录不能同时包含API错误。")

    expected_score = score_answer(
        record["raw_output"],
        record["gold_label"],
        finish_reason=record["finish_reason"],
    )
    expected_status = (
        "completed" if expected_score["completion_ok"] else "incomplete"
    )

    if record["score"] != expected_score or status != expected_status:
        raise ValueError("日志状态或评分与原始回答不一致。")

    if (
        record["correct"] != expected_score["lenient_correct"]
        or record["parsed_answer"] != expected_score["lenient_label"]
        or record["strict_format_ok"] != expected_score["strict_format_ok"]
    ):
        raise ValueError("日志中的评分摘要不一致。")


def decision(action, reason, next_attempt_no=None, wait_seconds=0):
    return {
        "policy_version": POLICY_VERSION,
        "action": action,
        "reason": reason,
        "next_attempt_no": next_attempt_no,
        "wait_seconds": wait_seconds,
    }


def plan_next_attempt(records, expected_record, *, pending_attempt):
    """expected_record提供当前输入身份；不会写入日志或用它猜答案。

    records应来自已校验的完整日志。pending_attempt必须由后续运行器
    的持久化请求标记确定，不能仅因没有回答记录就假定为False。
    """
    if type(pending_attempt) is not bool:
        raise ValueError("必须明确提供是否存在未确认的请求。")

    check_identity(expected_record)

    if expected_record["stage"] != "dev":
        raise ValueError("当前仅支持dev阶段。")

    if expected_record["condition"] not in ("C0", "C1", "C2"):
        raise ValueError("当前仅支持C0、C1、C2。")

    runtime = expected_record["runtime"]
    retries = runtime["transient_error_retries"]

    if type(retries) is not int or not 0 <= retries <= 2:
        raise ValueError("重试次数必须是0到2之间的整数。")

    if runtime["sdk_max_retries"] != 0 or runtime["concurrency"] != 1:
        raise ValueError("必须关闭SDK自动重试，并使用串行模式。")

    max_attempts = 1 + retries
    histories = {}

    # 验证整份历史，不能只检查当前题的最后一条记录。
    for record in records:
        check_identity(record)
        check_outcome(record)

        for key in GLOBAL_KEYS:
            if record[key] != expected_record[key]:
                raise ValueError(f"续跑上下文不一致：{key}")

        key = (record["question_id"], record["condition"])
        history = histories.setdefault(key, [])

        if type(record["attempt_no"]) is not int:
            raise ValueError("尝试编号必须是整数。")

        if record["attempt_no"] != len(history) + 1:
            raise ValueError("尝试记录缺失、重复或顺序不正确。")

        if record["attempt_no"] > max_attempts:
            raise ValueError("历史尝试次数超过配置上限。")

        if history:
            previous = history[-1]
            if previous["logical_request_id"] != record["logical_request_id"]:
                raise ValueError("同一题同一条件的输入身份发生变化。")

            if (
                previous["status"] != "api_error"
                or previous["error_kind"] not in TRANSIENT_ERRORS
            ):
                raise ValueError("历史中出现了不允许的额外尝试。")

        histories[key].append(record)

    target_key = (
        expected_record["question_id"],
        expected_record["condition"],
    )
    history = histories.get(target_key, [])

    for record in history:
        for key in TARGET_KEYS:
            if record[key] != expected_record[key]:
                raise ValueError(f"目标题续跑输入不一致：{key}")

    # 请求是否已经发生不确定时，绝不能根据“没有结果”自动重发。
    if pending_attempt:
        return decision("block", "unresolved_attempt")

    if not history:
        return decision("start", "no_previous_attempt", next_attempt_no=1)

    latest = history[-1]

    if latest["status"] in ("completed", "incomplete"):
        return decision("skip", "response_already_recorded")

    if latest["error_kind"] not in TRANSIENT_ERRORS:
        return decision("block", "non_retryable_error")

    if latest["attempt_no"] >= max_attempts:
        return decision("stop", "retry_limit_reached")

    return decision(
        "retry",
        "transient_error",
        next_attempt_no=latest["attempt_no"] + 1,
        wait_seconds=2 ** latest["attempt_no"],
    )
