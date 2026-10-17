"""答案解析和评分v1；不读取数据、不调用API。"""

import re


PARSER_VERSION = "answer_parser_v1"

STRICT_LINES = {
    "Answer: True": "True",
    "Answer: False": "False",
}

LENIENT_PATTERNS = (
    r"(?:final\s+)?answer\s*:\s*(true|false)[.!]?",
    r"(true|false)[.!]?",
    r"the\s+(?:final\s+)?answer\s+is\s+(true|false)[.!]?",
)

COMPILED_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE | re.ASCII)
    for pattern in LENIENT_PATTERNS
)


def label_from_line(line):
    """只接受完整匹配的答案行，不搜索普通句子中的标签。"""
    for pattern in COMPILED_PATTERNS:
        match = pattern.fullmatch(line.strip())
        if match:
            return match.group(1).capitalize()
    return None


def parse_answer(raw_output):
    result = {
        "parser_version": PARSER_VERSION,
        "last_nonempty_line": None,
        "strict_format_ok": False,
        "strict_label": None,
        "lenient_label": None,
        "parse_status": "unparseable",
    }

    if not isinstance(raw_output, str):
        result["parse_status"] = "non_text"
        return result

    # 只排除空白行；保留答案行原有空格供严格格式检查。
    lines = [
        line
        for line in raw_output.splitlines()
        if line.strip()
    ]

    if not lines:
        result["parse_status"] = "empty"
        return result

    last_line = lines[-1]
    strict_label = STRICT_LINES.get(last_line)

    result["last_nonempty_line"] = last_line
    result["strict_format_ok"] = strict_label is not None
    result["strict_label"] = strict_label

    final_label = label_from_line(last_line)
    if final_label is None:
        return result

    # 若多个完整答案行互相冲突，拒绝选择其中一个来抬高分数。
    recognized_labels = {
        label
        for line in lines
        if (label := label_from_line(line)) is not None
    }

    if len(recognized_labels) > 1:
        result["parse_status"] = "ambiguous"
        return result

    result["lenient_label"] = final_label
    result["parse_status"] = "ok"
    return result


def score_answer(raw_output, gold_label, *, finish_reason):
    """评分需要显式提供响应结束原因，不默认假定请求成功。"""
    if gold_label not in ("True", "False"):
        raise ValueError("标准答案必须是字符串True或False。")

    parsed = parse_answer(raw_output)
    completion_ok = finish_reason == "stop"
    eligible = completion_ok and parsed["parse_status"] == "ok"

    return {
        **parsed,
        "gold_label": gold_label,
        "finish_reason": finish_reason,
        "completion_ok": completion_ok,
        "strict_correct": (
            eligible
            and parsed["strict_format_ok"]
            and parsed["strict_label"] == gold_label
        ),
        "lenient_correct": (
            eligible
            and parsed["lenient_label"] == gold_label
        ),
        "score_status": (
            parsed["parse_status"] if completion_ok else "incomplete"
        ),
    }
