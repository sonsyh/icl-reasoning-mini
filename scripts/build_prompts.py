"""构建C0/C1/C2提示词；只读预览，不调用API。"""

import hashlib
import json

from select_random_demos import load_inputs, select_for_condition


PROMPT_VERSION = "logic_prompt_v1"

COMMON_INSTRUCTIONS = (
    "You are solving a deductive logic task.\n"
    "Use only the premises of the current problem. "
    "Treat category names as arbitrary symbols.\n"
    "Each problem is independent. Do not transfer facts between problems.\n"
    "Rules are directional: all A are B does not imply all B are A.\n"
    "Either the query or its explicit negation is derivable from the premises.\n"
    "Answer True if the query follows from the premises. "
    "Answer False if its explicit negation follows.\n"
)

ANSWER_ONLY_INSTRUCTIONS = (
    "Output only one line, exactly Answer: True or Answer: False. "
    "Do not include an explanation."
)

COT_INSTRUCTIONS = (
    "Give a short derivation before your final answer.\n"
    "Your last non-empty line must be exactly Answer: True or Answer: False. "
    "Do not write anything after that line."
)


def render_problem(item):
    """白名单取字段：只读取前提和命题。"""
    premises = item["premises"]
    query = item["query"]

    if (
        not isinstance(premises, list)
        or not premises
        or any(
            not isinstance(sentence, str) or not sentence.strip()
            for sentence in premises
        )
    ):
        raise ValueError("premises必须是非空字符串列表。")

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query必须是非空字符串。")

    lines = [
        f"{index}. {sentence}"
        for index, sentence in enumerate(premises, start=1)
    ]

    return "Premises:\n" + "\n".join(lines) + "\n\nQuery:\n" + query


def render_demo_answer(demo, include_cot):
    label = demo["label"]

    if label not in ("True", "False"):
        raise ValueError("示例标签必须是True或False。")

    answer = f"Answer: {label}"

    if not include_cot:
        return answer

    chain = demo["gold_cot"]
    if (
        not isinstance(chain, list)
        or not chain
        or any(
            not isinstance(sentence, str) or not sentence.strip()
            for sentence in chain
        )
    ):
        raise ValueError("C2示例必须包含有效参考推理链。")

    return "Reasoning:\n" + "\n".join(chain) + "\n\n" + answer


def build_messages(target, condition, demos=None):
    if condition not in ("C0", "C1", "C2"):
        raise ValueError("当前只支持C0、C1、C2。")

    selected = [] if demos is None else list(demos)

    if condition == "C0":
        if selected:
            raise ValueError("C0不能包含示例。")
    else:
        if len(selected) != 2:
            raise ValueError("C1/C2必须包含两条示例。")

        if [demo["label"] for demo in selected] != ["True", "False"]:
            raise ValueError("示例顺序必须是True在前、False在后。")

        if len({demo["id"] for demo in selected}) != 2:
            raise ValueError("不能重复使用同一条示例。")

        if len({demo["world_fingerprint"] for demo in selected}) != 2:
            raise ValueError("两条示例不能来自同一世界。")

        for demo in selected:
            if (
                demo["id"] == target["id"]
                or demo["world_fingerprint"] == target["world_fingerprint"]
            ):
                raise ValueError("示例不能是目标题或来自目标题的世界。")

    include_cot = condition == "C2"
    output_instructions = (
        COT_INSTRUCTIONS if include_cot else ANSWER_ONLY_INSTRUCTIONS
    )

    messages = [
        {
            "role": "system",
            "content": COMMON_INSTRUCTIONS + output_instructions,
        }
    ]

    for demo in selected:
        messages.append({
            "role": "user",
            "content": render_problem(demo),
        })
        messages.append({
            "role": "assistant",
            "content": render_demo_answer(demo, include_cot),
        })

    messages.append({
        "role": "user",
        "content": render_problem(target),
    })

    return messages


def hash_messages(messages):
    payload = json.dumps(
        messages,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    config, demo_pool, targets = load_inputs()
    seed = config["demo_selection"]["seed"]
    target = sorted(targets, key=lambda item: item["id"])[0]

    print("提示词版本：", PROMPT_VERSION)
    print("预览目标题：", target["id"])

    for condition in ("C0", "C1", "C2"):
        selected = select_for_condition(
            demo_pool, target, seed, condition
        )
        messages = build_messages(target, condition, selected)

        print(f"\n===== {condition} =====")
        print("示例ID：", [demo["id"] for demo in selected])
        print("messages SHA-256：", hash_messages(messages))

        for index, message in enumerate(messages, start=1):
            print(f"\n[{index}: {message['role']}]")
            print(message["content"])

    print("\n只读预览：未保存提示词文件、未读取test、未调用API。")
