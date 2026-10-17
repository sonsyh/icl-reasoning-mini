"""Day5 checkpoint 2: offline test embeddings; default read-only preview."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import build_test_plan as plan
be = plan.be
np = be.np
VERSION = 'test_embeddings_v1'
CACHE_PATH = 'data/embeddings/minilm_l6_v2_1110a24_test_retrieval_text_v1'


def validate_index(index, count):
    if len(index) != count:
        raise ValueError('向量索引数量不符。')
    for row in index:
        if (set(row) != {'id', 'text_sha256', 'token_count'}
                or not isinstance(row['id'], str) or not row['id']
                or not isinstance(row['text_sha256'], str) or len(row['text_sha256']) != 64
                or type(row['token_count']) is not int or not 1 <= row['token_count'] <= 256):
            raise ValueError('索引包含非法字段或长度。')
    if len({row['id'] for row in index}) != count:
        raise ValueError('向量索引ID重复。')


def save_cache(destination, vectors, index, provenance):
    be.validate_vectors(vectors, 60)
    validate_index(index, 60)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    vector_path, index_path = destination / 'test_vectors.npy', destination / 'test_index.json'
    with vector_path.open('xb') as file:
        np.save(file, vectors, allow_pickle=False)
        file.flush()
        os.fsync(file.fileno())
    be.write_json_new(index_path, index)
    reread = np.load(vector_path, allow_pickle=False)
    be.validate_vectors(reread, 60)
    if (not np.array_equal(reread, vectors)
            or json.loads(index_path.read_text(encoding='utf-8')) != index):
        raise ValueError('缓存保存后重读不一致；保留现场。')
    manifest = {
        'test_embedding_manifest_version': 1, 'builder_version': VERSION,
        'status': 'complete', 'stage': 'test',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'group': {'count': 60, 'shape': [60, 384], 'dtype': 'float32'},
        'provenance': provenance,
        'output_hashes': {'test_vectors.npy': be.file_hash(vector_path),
                          'test_index.json': be.file_hash(index_path)},
    }
    manifest['manifest_hash'] = be.digest_text(be.canonical(manifest))
    # Write complete marker last. Never resume/repair an existing cache directory.
    manifest_path = destination / 'test_embedding_manifest.json'
    be.write_json_new(manifest_path, manifest)
    if json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
        raise ValueError('缓存清单保存后重读不一致。')
    return manifest


def encode_targets(model, targets, index, texts):
    if model.max_seq_length != 256 or model.get_embedding_dimension() != 384:
        raise ValueError('模型维度或输入限制改变。')
    again, same_texts = be.build_index(targets, model.tokenizer, 256)
    if again != index or same_texts != texts:
        raise ValueError('模型内部与预检分词器不一致。')
    options = {'batch_size': 16, 'prompt': '', 'precision': 'float32',
               'convert_to_numpy': True, 'normalize_embeddings': True}
    vectors = model.encode(texts, show_progress_bar=True, **options)
    be.validate_vectors(vectors, 60)
    repeat = model.encode(texts[:16], show_progress_bar=False, **options)
    be.validate_vectors(repeat, 16)
    if not np.allclose(repeat, vectors[:16], atol=1e-6, rtol=1e-5):
        raise ValueError('首批重复编码在指定容差内不一致。')
    return vectors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--write', action='store_true')
    args = parser.parse_args(argv)
    plan.check_environment()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
    destination = ROOT / CACHE_PATH
    if args.write and (destination.exists() or destination.is_symlink()):
        raise FileExistsError('缓存已存在，不能覆盖或自动续写。')
    config, reference, targets, paths, fingerprints = plan.prepare(ROOT)
    paths = dict(paths)
    paths['test_embedding_source'] = Path(__file__).resolve()
    paths['test_preflight_source'] = ROOT / 'scripts/build_test_plan.py'
    captured = be.snapshot(paths)
    for name, expected in fingerprints.items():
        if captured[name] != expected:
            raise ValueError('预检期间来源改变。')
    print('冻结数据、Day3/Day4文件及demo向量缓存核对：通过', flush=True)
    from transformers import AutoTokenizer
    model_dir = ROOT / reference['model']['local_path']
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True,
                                              trust_remote_code=False, token=False)
    index, texts = be.build_index(targets, tokenizer, 256)
    validate_index(index, 60)
    be.check_openmp()
    be.verify_snapshot(paths, captured)
    print(f'test：60题；最大token数：{max(row["token_count"] for row in index)}；无截断', flush=True)
    if not args.write:
        print('只读预览：PASS；未加载模型权重、未计算或保存向量、未选例、未调用API。')
        print('拟保存目录：', destination)
        return 0
    print('开始离线编码60道test，CPU单线程；不调用DeepSeek……', flush=True)
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(1)
    model = SentenceTransformer(str(model_dir), device='cpu', backend='torch',
                                local_files_only=True, trust_remote_code=False, token=False)
    be.check_openmp()
    vectors = encode_targets(model, targets, index, texts)
    openmp = be.check_openmp()
    be.verify_snapshot(paths, captured)
    provenance = {
        'inference_test_config': config,
        'retrieval_settings': {name: reference[name] for name in ('model', 'input_text', 'encoding', 'retrieval')},
        'input_hashes': captured, 'packages': {name: version(name) for name in be.PACKAGES},
        'python_version': sys.version.split()[0], 'platform': sys.platform,
        'openmp_paths': openmp, 'row_order': 'original_jsonl_order', 'encode_prompt': '',
        'target_fields': list(plan.TARGET_FIELDS),
        'repeat_check': {'test_first_batch_count': 16, 'atol': 1e-6, 'rtol': 1e-5},
        'demo_cache_reused': config['demo_selection']['semantic']['demo_embedding_cache_path'],
    }
    manifest = save_cache(destination, vectors, index, provenance)
    be.verify_snapshot(paths, captured)
    print('test向量缓存验收：PASS；shape=(60, 384)，float32、有限、单位长度')
    print('首批重复编码及保存后逐值重读：通过')
    print('缓存目录：', destination)
    print('清单内部指纹：', manifest['manifest_hash'])
    print('未重建demo向量、未展示test题面/答案、未选例、未保存提示词、未读取密钥、未下载新资源、未调用API。')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('已中断；保留可能存在的缓存，不要删除或重复运行。')
        raise SystemExit(130)
    except Exception as exc:
        print('停止：', type(exc).__name__)
        if isinstance(exc, (ValueError, FileExistsError)):
            print(str(exc))
        print('若已创建部分缓存，请保留现场，不要覆盖、删除或改目录重跑。')
        raise SystemExit(1)
