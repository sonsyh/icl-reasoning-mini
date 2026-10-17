import argparse
import json
from collections import Counter
from pathlib import Path

from convert_prontoqa import convert_sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--expected-per-depth", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.expected_per_depth <= 0:
        parser.error("--expected-per-depth 必须大于0")

    raw_dir = args.raw_dir.resolve()
    output_path = args.output.resolve()

    if output_path.exists():
        raise FileExistsError(f"输出已存在，不允许覆盖：{output_path}")

    records = []
    summaries = []

    for depth in (2, 3, 4):
        source_path = raw_dir / f"{depth}hop_random_seed{args.seed}.json"

        with source_path.open(encoding="utf-8") as f:
            raw_data = json.load(f)

        if not isinstance(raw_data, dict):
            raise ValueError(f"{source_path.name} 的顶层格式不是字典")

        if len(raw_data) != args.expected_per_depth:
            raise ValueError(
                f"{source_path.name} 预期{args.expected_per_depth}道，"
                f"实际{len(raw_data)}道"
            )

        labels = Counter()

        for example_id, raw_record in raw_data.items():
            converted = convert_sample(
                record=raw_record,
                source_file=source_path.name,
                example_id=example_id,
                depth=depth,
                seed=args.seed,
            )
            records.append(converted)
            labels[converted["label"]] += 1

        summaries.append((depth, len(raw_data), dict(labels)))

    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("发现重复编号，停止写出")

    content = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("x", encoding="utf-8", newline="\n") as f:
        f.write(content)

    for depth, count, labels in summaries:
        print(f"{depth}跳：{count}道，标签分布：{labels}")

    print(f"\n共写出{len(records)}道题")
    print(f"输出文件：{output_path}")


if __name__ == "__main__":
    main()
