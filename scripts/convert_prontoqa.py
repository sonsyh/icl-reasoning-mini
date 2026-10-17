import hashlib
import json
import re
from pathlib import Path


def normalize_whitespace(text):
    """统一空白，不改变词语、大小写、否定词或标点。"""
    return " ".join(text.split())


def fingerprint(value):
    """用确定的 JSON 编码方式计算 SHA-256 指纹。"""
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def convert_sample(record, source_file, example_id, depth, seed):
    """将一道原始题转换为项目统一格式。"""
    item = record["test_example"]

    question = item["question"]
    raw_query = item["query"]
    label = item["answer"]
    gold_cot = item["chain_of_thought"]

    if not isinstance(question, str) or not question.strip():
        raise ValueError("前提必须是非空字符串")

    if not question.strip().endswith("."):
        raise ValueError("前提格式异常：最后一句没有句号")

    # 本批数据使用简单英文句子，以句号后的空白分隔。
    premises = [
        normalize_whitespace(sentence)
        for sentence in re.split(r"(?<=\.)\s+", question.strip())
    ]

    prefix = "True or false: "
    if not isinstance(raw_query, str) or not raw_query.startswith(prefix):
        raise ValueError("问题缺少预期的固定前缀")

    query = normalize_whitespace(raw_query[len(prefix):])
    if not query:
        raise ValueError("待判断命题不能为空")

    if label not in ("True", "False"):
        raise ValueError(f"非法标签：{label!r}")

    if (
        not isinstance(gold_cot, list)
        or not gold_cot
        or not all(isinstance(s, str) and s.strip() for s in gold_cot)
    ):
        raise ValueError("参考推理链必须是非空字符串列表")

    if depth not in (2, 3, 4):
        raise ValueError(f"本批数据的深度不支持：{depth}")

    # 世界指纹忽略前提顺序及重复句子。
    canonical_world = sorted(set(premises))

    # 输入指纹保留前提顺序，不包含答案或参考链。
    canonical_prompt = {
        "premises": premises,
        "query": query,
    }

    return {
        "id": f"prontoqa_v1_seed{seed}_h{depth}_{example_id}",
        "source_file": source_file,
        "source_example_id": example_id,
        "source_seed": seed,
        "depth": depth,
        "premises": premises,
        "query": query,
        "label": label,
        "gold_cot": gold_cot.copy(),
        "world_fingerprint": fingerprint(canonical_world),
        "prompt_fingerprint": fingerprint(canonical_prompt),
    }


def main():
    project_root = Path(__file__).resolve().parents[1]
    source_path = (
        project_root
        / "data"
        / "raw"
        / "step2_samples_20261005"
        / "2hop_random_seed20261005.json"
    )

    with source_path.open(encoding="utf-8") as f:
        raw_data = json.load(f)

    converted = convert_sample(
        record=raw_data["example1"],
        source_file=source_path.name,
        example_id="example1",
        depth=2,
        seed=20261005,
    )

    print(json.dumps(converted, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

