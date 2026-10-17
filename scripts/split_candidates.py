"""Deterministic stratified splitting for the one-record-per-world candidate batch.

This script does not silently deduplicate inputs. Repeated IDs, prompts, or
worlds must be resolved before allocation. Existing output directories are
never reused or overwritten. Test selection is not test freezing.
"""

import argparse
import hashlib
import itertools
import json
import platform
import random
from collections import Counter
from pathlib import Path


SPLIT_NAMES = ("demo", "dev", "test", "unused")
LABELS = ("True", "False")
QUOTAS = {
    2: {"demo": 20, "dev": 4, "test": 10},
    3: {"demo": 20, "dev": 3, "test": 10},
    4: {"demo": 20, "dev": 3, "test": 10},
}
FIELDS = {
    "id", "source_file", "source_example_id", "source_seed", "depth",
    "premises", "query", "label", "gold_cot", "world_fingerprint",
    "prompt_fingerprint",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(blob):
    return hashlib.sha256(blob).hexdigest()


def canonical_digest(value):
    return sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8"))


def read_records(path):
    blob = path.read_bytes()
    records = []
    for line_number, line in enumerate(blob.decode("utf-8").splitlines(), 1):
        require(bool(line.strip()), f"{path.name}:{line_number} 是空行")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path.name}:{line_number} 不是有效JSON") from error
        require(isinstance(record, dict), f"{path.name}:{line_number} 不是对象")
        require(set(record) == FIELDS, f"{path.name}:{line_number} 字段不符合约定")
        for key in ("id", "source_file", "source_example_id", "query"):
            require(isinstance(record[key], str) and bool(record[key].strip()),
                    f"{path.name}:{line_number} 的{key}不是非空字符串")
        require(type(record["source_seed"]) is int, "source_seed必须是整数")
        require(type(record["depth"]) is int and record["depth"] in QUOTAS,
                "depth必须是2、3或4")
        require(record["label"] in LABELS, "标签必须是True或False字符串")
        for key in ("premises", "gold_cot"):
            value = record[key]
            require(isinstance(value, list) and bool(value)
                    and all(isinstance(s, str) and bool(s.strip()) for s in value),
                    f"{key}必须是非空字符串列表")
        premises = [" ".join(s.split()) for s in record["premises"]]
        query = " ".join(record["query"].split())
        require(premises == record["premises"] and query == record["query"],
                f"{record['id']}的前提或问题尚未规范化空白")
        require(record["world_fingerprint"] == canonical_digest(sorted(set(premises))),
                f"{record['id']}的世界指纹不匹配")
        require(record["prompt_fingerprint"] == canonical_digest(
            {"premises": premises, "query": query}),
            f"{record['id']}的输入指纹不匹配")
        records.append(record)
    return blob, records


def allocate(records, inspected_worlds, seed):
    for field in ("id", "prompt_fingerprint", "world_fingerprint"):
        require(len({r[field] for r in records}) == len(records),
                f"候选数据存在重复{field}；本脚本要求先解决重复记录")

    rng = random.Random(seed)
    splits = {name: [] for name in SPLIT_NAMES}
    for depth in (2, 3, 4):
        quota = QUOTAS[depth]
        for label in LABELS:
            group = sorted(
                (r for r in records if r["depth"] == depth and r["label"] == label),
                key=lambda r: r["id"],
            )
            rng.shuffle(group)
            eligible = [r for r in group
                        if r["world_fingerprint"] not in inspected_worlds]
            require(len(eligible) >= quota["test"],
                    f"{depth}跳{label}的未查看世界不足test配额")
            test = eligible[:quota["test"]]
            test_ids = {r["id"] for r in test}
            remaining = [r for r in group if r["id"] not in test_ids]
            require(len(remaining) >= quota["dev"] + quota["demo"],
                    f"{depth}跳{label}不足demo和dev配额")
            dev_end = quota["dev"]
            demo_end = dev_end + quota["demo"]
            splits["test"].extend(test)
            splits["dev"].extend(remaining[:dev_end])
            splits["demo"].extend(remaining[dev_end:demo_end])
            splits["unused"].extend(remaining[demo_end:])

    # Fix file order as well as membership; avoid grouping output by label/depth.
    for name in SPLIT_NAMES:
        rng.shuffle(splits[name])

    id_sets = {name: {r["id"] for r in group} for name, group in splits.items()}
    world_sets = {
        name: {r["world_fingerprint"] for r in group}
        for name, group in splits.items()
    }
    for name in SPLIT_NAMES:
        require(len(id_sets[name]) == len(splits[name]), f"{name}内部编号重复")
    for first, second in itertools.combinations(SPLIT_NAMES, 2):
        require(not (id_sets[first] & id_sets[second]), f"{first}/{second}编号交叉")
        require(not (world_sets[first] & world_sets[second]), f"{first}/{second}世界交叉")
    require(set().union(*id_sets.values()) == {r["id"] for r in records},
            "存在未分配记录或额外记录")
    require(sum(map(len, splits.values())) == len(records), "分组总数量不一致")
    require(not (world_sets["test"] & inspected_worlds), "test包含已查看世界")

    for name in ("demo", "dev", "test"):
        actual = Counter((r["depth"], r["label"]) for r in splits[name])
        expected = Counter({(depth, label): QUOTAS[depth][name]
                            for depth in QUOTAS for label in LABELS})
        require(actual == expected, f"{name}的深度和标签配额不匹配")
    return splits


def jsonl_bytes(records):
    return "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n"
                   for r in records).encode("utf-8")


def build_outputs(input_path, inspected_path, seed, expected_count):
    input_blob, records = read_records(input_path)
    inspected_blob, inspected = read_records(inspected_path)
    require(len(records) == expected_count,
            f"预期{expected_count}条候选记录，实际{len(records)}条")
    require(bool(inspected), "已查看样本清单不能为空")
    inspected_worlds = {r["world_fingerprint"] for r in inspected}
    splits = allocate(records, inspected_worlds, seed)
    payloads = {f"{name}.jsonl": jsonl_bytes(splits[name]) for name in SPLIT_NAMES}
    counts = {
        name: {
            "total": len(group),
            "by_depth_and_label": {
                str(depth): {label: sum(r["depth"] == depth and r["label"] == label
                                       for r in group) for label in LABELS}
                for depth in QUOTAS
            },
        } for name, group in splits.items()
    }
    report = {
        "report_schema_version": 1,
        "status": "split_generated_not_frozen",
        "test_frozen": False,
        "manual_acceptance": "pending",
        "split_seed": seed,
        "python_version": platform.python_version(),
        "rng": "Python random.Random; sorted IDs before each stratum shuffle",
        "stratum_order": [[d, label] for d in QUOTAS for label in LABELS],
        "selection_order": ["test", "dev", "demo", "unused"],
        "final_file_shuffle_order": list(SPLIT_NAMES),
        "quotas_per_label": QUOTAS,
        "candidate_input": {"name": input_path.name, "sha256": sha256(input_blob),
                            "count": len(records)},
        "inspected_input": {"name": inspected_path.name, "sha256": sha256(inspected_blob),
                            "count": len(inspected), "unique_worlds": len(inspected_worlds)},
        "generation_seeds": sorted({r["source_seed"] for r in records}),
        "candidate_records_excluded_from_test": sum(
            r["world_fingerprint"] in inspected_worlds for r in records),
        "checks": {"quota_match": True, "ids_disjoint": True,
                   "worlds_disjoint": True, "all_candidates_accounted_for": True,
                   "test_excludes_inspected_worlds": True},
        "counts": counts,
        "file_sha256": {name: sha256(blob) for name, blob in payloads.items()},
        "script_sha256": sha256(Path(__file__).read_bytes()),
    }
    payloads["split_report.json"] = (
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    return payloads, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--inspected", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--expected-count", type=int, default=600)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在，不允许覆盖：{output_dir}")
    payloads, report = build_outputs(
        args.input.resolve(), args.inspected.resolve(), args.seed, args.expected_count
    )
    # Compute and validate all outputs before creating the fresh destination.
    output_dir.mkdir(parents=True, exist_ok=False)
    for filename, blob in payloads.items():
        with (output_dir / filename).open("xb") as f:
            f.write(blob)
    for name in SPLIT_NAMES:
        print(f"{name}: {report['counts'][name]['total']}道")
    print("PASS: 配额、编号隔离、世界隔离、完整归属及已查看世界排除检查")
    print(f"分组种子：{args.seed}")
    print(f"输出目录：{output_dir}")
    print("test已分配，但尚未冻结；需要后续检查和人工验收")


if __name__ == "__main__":
    main()
