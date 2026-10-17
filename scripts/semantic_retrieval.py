"""Label-balanced cosine selection using the accepted offline demo/dev cache."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(os.environ.get("ICL_REASONING_PROJECT_ROOT",
                          str(Path(__file__).resolve().parents[1]))).resolve()
sys.path.insert(0, str(ROOT / "scripts"))
import build_embeddings as be

VERSION = "semantic_label_balanced_cosine_v1_float64"
EXPECTED_CACHE_HASH = "58d1b554b0341056ef10d386d13276e10b6467e8b57db657d5ec2675d99fc620"
OUTPUT_PATH = "data/retrieval/dev_semantic_minilm_v1"
CACHE_FILES = ("demo_vectors.npy", "dev_vectors.npy", "demo_index.json", "dev_index.json")


def validate_manifest(manifest):
    payload = dict(manifest)
    saved = payload.pop("manifest_hash", None)
    if saved != be.digest_text(be.canonical(payload)) or saved != EXPECTED_CACHE_HASH:
        raise ValueError("向量缓存清单指纹与已验收记录不一致。")
    if (manifest.get("cache_schema_version") != 1
            or manifest.get("status") != "complete"
            or manifest.get("generator_version") != be.VERSION):
        raise ValueError("向量缓存未完成或版本不正确。")
    if set(manifest["output_hashes"]) != set(CACHE_FILES):
        raise ValueError("缓存文件清单不正确。")


def validate_index(index, rows):
    if not isinstance(index, list) or len(index) != len(rows):
        raise ValueError("向量索引数量不正确。")
    for item, row in zip(index, rows):
        if (set(item) != {"id", "text_sha256", "token_count"}
                or item["id"] != row["id"]
                or item["text_sha256"] != be.digest_text(be.render_text(row))
                or type(item["token_count"]) is not int
                or not 1 <= item["token_count"] <= 256):
            raise ValueError("向量行与题目ID/文本指纹不对应。")


def load_inputs(root):
    config, groups, paths, fingerprints = be.prepare(root)
    cache = root / be.CACHE_PATH
    manifest_path = cache / "embedding_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    provenance = manifest["provenance"]
    if (be.canonical(provenance["config"]) != be.canonical(config)
            or provenance["input_hashes"] != fingerprints
            or provenance["row_order"] != "original_jsonl_order"
            or provenance["encode_prompt"] != ""
            or provenance["packages"] != be.PACKAGES):
        raise ValueError("向量缓存与当前配置、原数据、模型或生成源码不对应。")
    vectors = {}
    paths = dict(paths)
    paths["embedding_manifest"] = manifest_path
    paths["retrieval_source"] = Path(__file__).resolve()
    for name in CACHE_FILES:
        path = cache / name
        if be.file_hash(path) != manifest["output_hashes"][name]:
            raise ValueError("向量缓存文件指纹改变：" + name)
        paths["embedding_cache/" + name] = path
    for split in ("demo", "dev"):
        rows = groups[split]
        index = json.loads((cache / f"{split}_index.json").read_text(encoding="utf-8"))
        validate_index(index, rows)
        vectors[split] = np.load(cache / f"{split}_vectors.npy", allow_pickle=False)
        be.validate_vectors(vectors[split], len(rows))
        if manifest["groups"][split] != {
            "count": len(rows), "shape": [len(rows), 384], "dtype": "float32"
        }:
            raise ValueError("缓存形状元数据与实际数组不一致。")
    captured = be.snapshot(paths)
    # A second read ensures the file set did not drift during validation.
    for name, digest in fingerprints.items():
        if captured[name] != digest:
            raise ValueError("缓存验收期间输入发生变化。")
    be.verify_snapshot(paths, captured)
    return config, groups, vectors, paths, captured


def select_for_target(demos, demo_vectors, target, query_vector):
    """Only target ID/world and its provided vector are consulted, never its gold."""
    for key in ("id", "world_fingerprint"):
        if not isinstance(target.get(key), str) or not target[key]:
            raise ValueError("目标题缺少有效ID或世界指纹。")
    be.validate_vectors(demo_vectors, len(demos))
    if not isinstance(query_vector, np.ndarray) or query_vector.shape != (384,):
        raise ValueError("查询向量必须为384维。")
    be.validate_vectors(query_vector.reshape(1, 384), 1)
    for row in demos:
        if (not isinstance(row.get("id"), str) or not row["id"]
                or not isinstance(row.get("world_fingerprint"), str)
                or not row["world_fingerprint"]
                or row.get("label") not in ("True", "False")):
            raise ValueError("示例元数据或标签不合法。")
    for key in ("id", "world_fingerprint"):
        if len({row[key] for row in demos}) != len(demos):
            raise ValueError("示例候选包含重复ID或世界。")

    # Float32 cached embeddings, float64 cosine arithmetic. Normalize explicitly
    # to avoid assuming float32 vectors have mathematically exact unit norms.
    matrix = demo_vectors.astype(np.float64)
    query = query_vector.astype(np.float64)
    dots = np.einsum("ij,j->i", matrix, query, optimize=False)
    lengths = np.sqrt(np.einsum("ij,ij->i", matrix, matrix, optimize=False))
    query_length = np.sqrt(np.einsum("i,i->", query, query, optimize=False))
    scores = dots / (lengths * query_length)
    if not np.isfinite(scores).all() or np.any(np.abs(scores) > 1.000000000001):
        raise ValueError("余弦相似度数值异常。")
    scores = np.clip(scores, -1.0, 1.0)

    selected = []
    for label in ("True", "False"):
        candidates = [
            i for i, row in enumerate(demos)
            if row["label"] == label
            and row["id"] != target["id"]
            and row["world_fingerprint"] != target["world_fingerprint"]
        ]
        if not candidates:
            raise ValueError("隔离后该标签没有可用示例：" + label)
        # Use unrounded scores; exactly equal scores break by ascending ID.
        winner = min(candidates, key=lambda i: (-float(scores[i]), demos[i]["id"]))
        row = demos[winner]
        selected.append({"id": row["id"], "label": label,
                         "world_fingerprint": row["world_fingerprint"],
                         "similarity": float(scores[winner])})
    if selected[0]["id"] == selected[1]["id"] or (
            selected[0]["world_fingerprint"] == selected[1]["world_fingerprint"]):
        raise ValueError("两个示例不独立。")
    return selected


def make_plan(config, groups, vectors):
    config_hash = be.digest_text(be.canonical(config))
    records = []
    # Stable output order while preserving each vector's original row mapping.
    ordered = sorted(enumerate(groups["dev"]), key=lambda pair: pair[1]["id"])
    for index, target in ordered:
        minimal = {key: target[key] for key in ("id", "world_fingerprint")}
        selected = select_for_target(groups["demo"], vectors["demo"], minimal,
                                     vectors["dev"][index])
        record = {
            "selection_schema_version": 1, "stage": "dev", "condition": "C3",
            "algorithm_version": VERSION, "question_id": target["id"],
            "target_world_fingerprint": target["world_fingerprint"],
            "config_hash": config_hash, "embedding_manifest_hash": EXPECTED_CACHE_HASH,
            "demo_ids": [item["id"] for item in selected], "selected": selected,
        }
        record["record_hash"] = be.digest_text(be.canonical(record))
        records.append(record)
    return records


def validate_selection(record):
    payload = dict(record)
    digest = payload.pop("record_hash", None)
    if digest != be.digest_text(be.canonical(payload)):
        raise ValueError("选例记录指纹错误。")
    items = record["selected"]
    if (record["stage"] != "dev" or record["condition"] != "C3"
            or record["algorithm_version"] != VERSION
            or len(items) != 2
            or [item["label"] for item in items] != ["True", "False"]
            or record["demo_ids"] != [item["id"] for item in items]
            or len(set(record["demo_ids"])) != 2
            or len({item["world_fingerprint"] for item in items}) != 2):
        raise ValueError("示例标签平衡、顺序或独立性错误。")
    for item in items:
        if (item["id"] == record["question_id"]
                or item["world_fingerprint"] == record["target_world_fingerprint"]
                or type(item["similarity"]) is not float
                or not np.isfinite(item["similarity"])
                or not -1 <= item["similarity"] <= 1):
            raise ValueError("选例隔离或相似度不合法。")


def save_plan(destination, records, provenance):
    if not records or len({r["question_id"] for r in records}) != len(records):
        raise ValueError("选例计划为空或包含重复目标题。")
    for record in records:
        validate_selection(record)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    path = destination / "selections.jsonl"
    with path.open("x", encoding="utf-8") as file:
        for record in records:
            file.write(be.canonical(record) + "\n")
        file.flush()
        os.fsync(file.fileno())
    reloaded = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if reloaded != records:
        raise ValueError("选例保存后重读不一致。")
    manifest = {
        "selection_manifest_version": 1, "status": "complete", "algorithm_version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(), "stage": "dev",
        "question_count": len(records), "provenance": provenance,
        "selections_sha256": be.file_hash(path),
    }
    manifest["manifest_hash"] = be.digest_text(be.canonical(manifest))
    be.write_json_new(destination / "selection_manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    if Path(sys.prefix) != Path("/opt/icl-reasoning-retrieval"):
        raise ValueError("请使用icl-reasoning-retrieval环境。")
    if {name: version(name) for name in be.PACKAGES} != be.PACKAGES:
        raise ValueError("依赖版本与已验收环境不同。")
    destination = ROOT / OUTPUT_PATH
    if args.write and destination.exists():
        raise FileExistsError("选例目录已存在，禁止覆盖或自动续写：" + str(destination))
    config, groups, vectors, paths, captured = load_inputs(ROOT)
    records = make_plan(config, groups, vectors)
    if len(records) != 20:
        raise ValueError("dev选例记录必须为20条。")
    for record in records:
        validate_selection(record)
    # Reordered processing must not affect selection for a question.
    reverse_groups = dict(groups)
    reverse_vectors = dict(vectors)
    reverse_groups["dev"] = list(reversed(groups["dev"]))
    reverse_vectors["dev"] = vectors["dev"][::-1]
    if records != make_plan(config, reverse_groups, reverse_vectors):
        raise ValueError("目标题处理顺序影响选例。")
    be.verify_snapshot(paths, captured)
    print("向量缓存、数据/源码指纹与题目索引：通过")
    print("算法：", VERSION)
    print("dev题数：20；每题1 True + 1 False；固定True在前")
    print("相同ID/世界排除、重复构建及处理顺序检查：通过")
    for record in records[:3]:
        print("\n目标题：", record["question_id"])
        for item in record["selected"]:
            print(f"  {item['label']} | {item['id']} | cosine={item['similarity']:.8f}")
    counts = Counter(item["id"] for record in records for item in record["selected"])
    print("\n40个示例位置使用的不同demo数：", len(counts))
    print("单个demo最多被选次数：", max(counts.values()))
    if not args.write:
        print("只读预览：未保存选例、未重新编码、未加载模型、未读取test、未调用API。")
        print("拟保存目录：", destination)
        return 0
    manifest = save_plan(destination, records, {
        "config": config, "input_hashes": captured,
        "embedding_manifest_hash": EXPECTED_CACHE_HASH,
        "vector_dtype": "float32", "similarity_arithmetic_dtype": "float64",
        "tie_rule": "exact_unrounded_score_then_question_id_ascending",
        "packages": {name: version(name) for name in be.PACKAGES},
    })
    print("\n选例缓存验收：PASS；已保存20条C3选例记录")
    print("目录：", destination)
    print("清单指纹：", manifest["manifest_hash"])
    print("未修改向量缓存/源数据/Day3结果、未读取test、未调用API；没有模型回答。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("已中断；如有部分选例目录请保留，不要自动重跑或删除。")
        raise SystemExit(130)
    except Exception as exc:
        print("停止：", type(exc).__name__, str(exc))
        print("不要覆盖、删除或自动重写已有缓存；保留报错供核查。")
        raise SystemExit(1)
