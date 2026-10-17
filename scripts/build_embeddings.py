"""Build hash-bound demo/dev embeddings. Default is an offline, read-only preview."""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np

VERSION = "build_embeddings_v1"
ROOT = Path(os.environ.get("ICL_REASONING_PROJECT_ROOT",
                          str(Path(__file__).resolve().parents[1]))).resolve()
CONFIG_PATH = "configs/retrieval_dev_v1.json"
CACHE_PATH = "data/embeddings/minilm_l6_v2_1110a24_retrieval_text_v1"
EXPECTED_CONFIG = {
    "config_schema_version": 1, "config_name": "retrieval_dev_v1", "stage": "dev",
    "model": {
        "repo_id": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "local_path": "external/minilm_l6_v2_1110a24",
        "weight_sha256": "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db",
        "local_files_only": True, "trust_remote_code": False, "backend": "torch",
    },
    "input_text": {
        "version": "retrieval_text_v1", "fields": ["premises", "query"],
        "template": "Premises:\n{premises_joined_by_newline}\n\nQuery:\n{query}",
        "preserve_premise_order": True, "max_seq_length": 256,
        "overlength_policy": "error",
    },
    "encoding": {
        "device": "cpu", "torch_num_threads": 1, "batch_size": 16,
        "precision": "float32", "normalize_embeddings": True,
        "expected_dimension": 384,
    },
    "data": {
        "demo_path": "data/splits/seed20261007/demo.jsonl",
        "eval_path": "data/splits/seed20261007/dev.jsonl",
        "expected_demo_count": 120, "expected_eval_count": 20, "allow_test": False,
    },
    "retrieval": {
        "similarity": "cosine", "shots": 2, "label_order": ["True", "False"],
        "per_label": 1, "exclude_same_id": True, "exclude_same_world": True,
        "tie_break": "question_id_ascending",
    },
}
DATA_HASHES = {
    "data/splits/seed20261007/demo.jsonl": "4c4d7f5984fbe3e433e10581c67c4b7fe2935007e396ee67a60e42bbc4e1a530",
    "data/splits/seed20261007/dev.jsonl": "31323be559518f052c70b67e5c86b3476fd2281197675292d2b8fbc52c40785d",
}
ASSET_SIZES = {
    "README.md": 10502, "config.json": 612,
    "config_sentence_transformers.json": 116, "sentence_bert_config.json": 53,
    "modules.json": 349, "1_Pooling/config.json": 190,
    "tokenizer.json": 466247, "tokenizer_config.json": 350,
    "special_tokens_map.json": 112, "vocab.txt": 231508,
    "model.safetensors": 90868376,
}
ASSET_HASHES = {
    "model.safetensors": EXPECTED_CONFIG["model"]["weight_sha256"],
    "tokenizer.json": "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
    "tokenizer_config.json": "acb92769e8195aabd29b7b2137a9e6d6e25c476a4f15aa4355c233426c61576b",
    "sentence_bert_config.json": "fc1993fde0a95c24ec6c022539d41cf6e2f7c9721e5415d6fb6897472a9cd4b7",
}
PACKAGES = {
    "sentence-transformers": "6.1.0", "transformers": "5.19.0",
    "tokenizers": "0.23.2", "huggingface-hub": "1.33.0", "torch": "2.14.1",
    "numpy": "2.5.3", "scipy": "1.18.1", "scikit-learn": "1.9.1",
}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_config(config):
    # Canonical JSON distinguishes true from 1 and rejects silent new settings.
    if canonical(config) != canonical(EXPECTED_CONFIG):
        raise ValueError("配置与已确认的retrieval_dev_v1不一致；停止，不自动替换设置。")


def render_text(row):
    premises, query = row["premises"], row["query"]
    if (not isinstance(premises, list) or not premises
            or any(not isinstance(p, str) or not p.strip() for p in premises)
            or not isinstance(query, str) or not query.strip()):
        raise ValueError("前提或查询格式不正确。")
    return "Premises:\n" + "\n".join(premises) + "\n\nQuery:\n" + query


def load_rows(path, expected_count):
    text = Path(path).read_text(encoding="utf-8")
    if not text.endswith("\n"):
        raise ValueError("数据尾行不完整。")
    rows = [json.loads(line) for line in text.splitlines()]
    if len(rows) != expected_count:
        raise ValueError("题目数量与配置不一致。")
    for row in rows:
        if not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError("题目ID不合法。")
        if not isinstance(row.get("world_fingerprint"), str):
            raise ValueError("缺少世界指纹。")
        render_text(row)
    for key in ("id", "world_fingerprint"):
        if len({row[key] for row in rows}) != len(rows):
            raise ValueError("数据包含重复ID或世界。")
    # Preserve original JSONL row order; no sorting or shuffling.
    return rows


def build_index(rows, tokenizer, limit):
    index, texts = [], []
    for row in rows:
        text = render_text(row)
        minimal = {key: row[key] for key in ("premises", "query")}
        if text != render_text(minimal):
            raise ValueError("检索文本依赖禁止字段。")
        ids = tokenizer(text, add_special_tokens=True, truncation=False,
                        padding=False)["input_ids"]
        if len(ids) > limit:
            raise ValueError(f"题目超长，禁止截断：{row['id']}，{len(ids)} > {limit}")
        index.append({"id": row["id"], "text_sha256": digest_text(text),
                      "token_count": len(ids)})
        texts.append(text)
    return index, texts


def validate_vectors(vectors, count, dimension=384):
    if (not isinstance(vectors, np.ndarray) or vectors.shape != (count, dimension)
            or vectors.dtype != np.float32):
        raise ValueError("向量形状或float32类型不正确。")
    if not np.isfinite(vectors).all():
        raise ValueError("向量含有NaN或无穷大。")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5, rtol=0):
        raise ValueError("向量未归一化或含有零向量。")


def snapshot(paths):
    return {name: file_hash(path) for name, path in paths.items()}


def verify_snapshot(paths, expected):
    if snapshot(paths) != expected:
        raise ValueError("计算期间输入或源码指纹发生变化；停止保存。")


def prepare(root):
    config_path = root / CONFIG_PATH
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    model_dir = root / config["model"]["local_path"]
    paths = {CONFIG_PATH: config_path}
    paths.update({name: root / name for name in DATA_HASHES})
    paths.update({str(Path(config["model"]["local_path"]) / name): model_dir / name
                  for name in ASSET_SIZES})
    paths["generator_source"] = Path(__file__).resolve()
    fingerprints = snapshot(paths)
    for name, expected in DATA_HASHES.items():
        if fingerprints[name] != expected:
            raise ValueError("Day3固定数据指纹改变：" + name)
    for name, size in ASSET_SIZES.items():
        if (model_dir / name).stat().st_size != size:
            raise ValueError("模型资源大小改变：" + name)
    for name, expected in ASSET_HASHES.items():
        if file_hash(model_dir / name) != expected:
            raise ValueError("已记录模型资源指纹改变：" + name)
    modules = json.loads((model_dir / "modules.json").read_text())
    expected_modules = [
        {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
        {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
        {"idx": 2, "name": "2", "path": "2_Normalize", "type": "sentence_transformers.models.Normalize"},
    ]
    if modules != expected_modules:
        raise ValueError("模型模块清单与已确认结构不同。")
    groups = {
        "demo": load_rows(root / config["data"]["demo_path"], 120),
        "dev": load_rows(root / config["data"]["eval_path"], 20),
    }
    for key in ("id", "world_fingerprint"):
        if ({row[key] for row in groups["demo"]}
                & {row[key] for row in groups["dev"]}):
            raise ValueError("demo/dev存在交叉题目或世界。")
    verify_snapshot(paths, fingerprints)
    return config, groups, paths, fingerprints


def write_json_new(path, value):
    with Path(path).open("x", encoding="utf-8") as file:
        file.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        file.flush()
        os.fsync(file.fileno())


def save_cache(destination, arrays, indexes, provenance):
    if set(arrays) != {"demo", "dev"} or set(indexes) != {"demo", "dev"}:
        raise ValueError("只能保存demo/dev两个分组。")
    for split in ("demo", "dev"):
        ids = [item["id"] for item in indexes[split]]
        if len(set(ids)) != len(ids):
            raise ValueError("缓存索引ID重复。")
        validate_vectors(arrays[split], len(ids))
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    output_hashes = {}
    for split in ("demo", "dev"):
        vector_path = destination / f"{split}_vectors.npy"
        index_path = destination / f"{split}_index.json"
        with vector_path.open("xb") as file:
            np.save(file, arrays[split], allow_pickle=False)
            file.flush()
            os.fsync(file.fileno())
        write_json_new(index_path, indexes[split])
        reloaded = np.load(vector_path, allow_pickle=False)
        if not np.array_equal(reloaded, arrays[split]):
            raise ValueError("保存后向量重读不一致。")
        if json.loads(index_path.read_text()) != indexes[split]:
            raise ValueError("保存后索引重读不一致。")
        output_hashes[vector_path.name] = file_hash(vector_path)
        output_hashes[index_path.name] = file_hash(index_path)
    manifest = {
        "cache_schema_version": 1, "status": "complete", "generator_version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance, "output_hashes": output_hashes,
        "groups": {split: {"count": len(indexes[split]),
                           "shape": list(arrays[split].shape), "dtype": "float32"}
                   for split in ("demo", "dev")},
    }
    manifest["manifest_hash"] = digest_text(canonical(manifest))
    # Written last: missing manifest means incomplete cache, not safe to consume.
    write_json_new(destination / "embedding_manifest.json", manifest)
    return manifest


def check_openmp():
    from threadpoolctl import threadpool_info
    paths = sorted({str(Path(item["filepath"]).resolve())
                    for item in threadpool_info()
                    if item.get("user_api") == "openmp" and item.get("filepath")})
    if len(paths) > 1:
        raise ValueError("检测到多份OpenMP运行库，停止。")
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    if Path(sys.prefix) != Path("/opt/icl-reasoning-retrieval"):
        raise ValueError("必须在icl-reasoning-retrieval环境运行。")
    if os.environ.get("KMP_DUPLICATE_LIB_OK", "").lower() in ("true", "1", "yes"):
        raise ValueError("禁止使用OpenMP重复加载绕过开关。")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    destination = ROOT / CACHE_PATH
    if args.write and destination.exists():
        raise FileExistsError("缓存目录已存在；不覆盖、不自动续写：" + str(destination))
    environment = {name: version(name) for name in PACKAGES}
    if environment != PACKAGES:
        raise ValueError("依赖版本与已验收环境不同。")
    config, groups, paths, fingerprints = prepare(ROOT)
    print("配置、数据隔离与模型资源核查：通过", flush=True)
    from transformers import AutoTokenizer
    model_dir = ROOT / config["model"]["local_path"]
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True,
                                              trust_remote_code=False, token=False)
    indexes, texts = {}, {}
    for split in ("demo", "dev"):
        indexes[split], texts[split] = build_index(groups[split], tokenizer, 256)
        print(f"{split}：{len(texts[split])}题；最大token数："
              f"{max(item['token_count'] for item in indexes[split])}；无截断", flush=True)
    check_openmp()
    verify_snapshot(paths, fingerprints)
    if not args.write:
        print("只读预览通过；未加载权重、未计算或保存题目向量、未读取test、未调用API。")
        print("拟保存目录：", destination)
        return 0
    print("开始离线加载模型并编码140题，CPU单线程，请稍候……", flush=True)
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(1)
    model = SentenceTransformer(str(model_dir), device="cpu", backend="torch",
                                local_files_only=True, trust_remote_code=False, token=False)
    if model.max_seq_length != 256 or model.get_embedding_dimension() != 384:
        raise ValueError("模型输入长度或向量维度不同。")
    arrays = {}
    for split in ("demo", "dev"):
        index_again, _ = build_index(groups[split], model.tokenizer, 256)
        if index_again != indexes[split]:
            raise ValueError("模型内部和预检分词器的计数不同。")
        arrays[split] = model.encode(
            texts[split], batch_size=16, show_progress_bar=True, prompt="",
            precision="float32", convert_to_numpy=True, normalize_embeddings=True,
        )
        validate_vectors(arrays[split], len(groups[split]))
        print(f"{split}编码：通过；shape={arrays[split].shape}", flush=True)
    repeated = model.encode(texts["demo"][:16], batch_size=16, prompt="",
                            show_progress_bar=False, precision="float32",
                            convert_to_numpy=True, normalize_embeddings=True)
    if not np.allclose(repeated, arrays["demo"][:16], atol=1e-6, rtol=1e-5):
        raise ValueError("第一批重复编码在指定容差内不一致。")
    omp = check_openmp()
    verify_snapshot(paths, fingerprints)
    provenance = {
        "config": config, "input_hashes": fingerprints,
        "packages": environment, "python_version": sys.version.split()[0],
        "platform": sys.platform, "openmp_paths": omp,
        "row_order": "original_jsonl_order", "encode_prompt": "",
        "repeat_check": {"demo_first_batch_count": 16, "atol": 1e-6, "rtol": 1e-5},
    }
    manifest = save_cache(destination, arrays, indexes, provenance)
    print("\n向量缓存验收：PASS")
    print("demo：(120, 384)；dev：(20, 384)；float32、有限、单位长度")
    print("第一批重复编码检查：通过；保存后逐值重读：通过")
    print("缓存目录：", destination)
    print("缓存清单指纹：", manifest["manifest_hash"])
    print("未修改源数据、未读取test、未下载新资源、未调用API；尚未进行选例。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("已中断；如已创建缓存目录，请保留它并停止，不要删除或重复运行。")
        raise SystemExit(130)
    except Exception as exc:
        print("停止：", type(exc).__name__, str(exc))
        print("如存在部分缓存，请保留文件；不要自动重跑、覆盖或删除。")
        raise SystemExit(1)
