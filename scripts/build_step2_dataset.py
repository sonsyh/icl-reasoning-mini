import json
from collections import Counter
from pathlib import Path

from convert_prontoqa import convert_sample


def main():
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw" / "step2_samples_20261005"
    output_path = project_root / "data" / "processed" / "step2_candidates.jsonl"

    records = []
    summaries = []

    for depth in (2, 3, 4):
        source_path = raw_dir / f"{depth}hop_random_seed20261005.json"

        with source_path.open(encoding="utf-8") as f:
            raw_data = json.load(f)

        if not isinstance(raw_data, dict) or len(raw_data) != 8:
            raise ValueError(f"{source_path.name} 应包含8道题，请检查原始文件")

        labels = Counter()

        for example_id, raw_record in raw_data.items():
            converted = convert_sample(
                record=raw_record,
                source_file=source_path.name,
                example_id=example_id,
                depth=depth,
                seed=20261005,
            )
            records.append(converted)
            labels[converted["label"]] += 1

        summaries.append((depth, len(raw_data), dict(labels)))

    if len(records) != 24:
        raise ValueError(f"预期24道题，实际转换了{len(records)}道")

    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("发现重复题目编号，停止写出")

    # 每道题编码成一行；所有转换成功后才开始写文件。
    content = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # x 模式只允许创建新文件，防止覆盖已有数据。
    with output_path.open("x", encoding="utf-8", newline="\n") as f:
        f.write(content)

    for depth, count, labels in summaries:
        print(f"{depth}跳：{count}道，标签分布：{labels}")

    print(f"\n共写出{len(records)}道题")
    print(f"输出文件：{output_path}")


if __name__ == "__main__":
    main()
