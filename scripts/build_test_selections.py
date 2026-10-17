"""Fixed test demo selections; cached vectors only, no model or API loading."""
import argparse
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import build_test_plan as plan
import build_test_embeddings as embeddings
from select_random_demos import select_for_condition, ALGORITHM_VERSION
sr, be = plan.sr, plan.be
VERSION = 'test_selections_v1'
OUTPUT_PATH = 'data/retrieval/test_selections_v1'
TEST_EMBEDDING_HASH = 'efbb6ca5c2c5a5a9f060349ab642c8729134194a660fb3b30eababbc779b23f4'
CONDITIONS = ('C0', 'C1', 'C2', 'C3')


def digest(value):
    return be.digest_text(be.canonical(value))


def load_inputs(root):
    root = Path(root)
    config, reference, targets, paths, captured = plan.prepare(root)
    paths = dict(paths)
    paths['test_embedding_source'] = root / 'scripts/build_test_embeddings.py'
    folder = root / embeddings.CACHE_PATH
    manifest_path = folder / 'test_embedding_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    payload = dict(manifest)
    saved = payload.pop('manifest_hash', None)
    if (saved != TEST_EMBEDDING_HASH or digest(payload) != saved
            or manifest['status'] != 'complete' or manifest['stage'] != 'test'
            or manifest['builder_version'] != embeddings.VERSION
            or manifest['group'] != {'count': 60, 'shape': [60, 384], 'dtype': 'float32'}):
        raise ValueError('test向量清单与已验收版本不同。')
    provenance = manifest['provenance']
    if (provenance['input_hashes'] != be.snapshot(paths)
            or provenance['inference_test_config'] != config
            or provenance['retrieval_settings'] != {name: reference[name] for name in ('model', 'input_text', 'encoding', 'retrieval')}
            or provenance['packages'] != be.PACKAGES
            or provenance['row_order'] != 'original_jsonl_order'
            or provenance['encode_prompt'] != '' or provenance['target_fields'] != list(plan.TARGET_FIELDS)):
        raise ValueError('test向量来源、配置或编码规则改变。')
    if set(manifest['output_hashes']) != {'test_vectors.npy', 'test_index.json'}:
        raise ValueError('test向量缓存文件集合错误。')
    for name, expected in manifest['output_hashes'].items():
        path = folder / name
        plan.verify_file(path, expected)
        paths['test_cache/' + name] = path
    index = json.loads((folder / 'test_index.json').read_text(encoding='utf-8'))
    embeddings.validate_index(index, 60)
    for entry, target in zip(index, targets):
        if (entry['id'] != target['id']
                or entry['text_sha256'] != be.digest_text(be.render_text(target))):
            raise ValueError('test向量索引与原始题目文本不对应。')
    query_vectors = be.np.load(folder / 'test_vectors.npy', allow_pickle=False)
    be.validate_vectors(query_vectors, 60)
    _, groups, vectors, cache_paths, cache_hashes = sr.load_inputs(root)
    paths.update(cache_paths)
    paths['test_embedding_manifest'] = manifest_path
    paths['test_selection_source'] = Path(__file__).resolve()
    fingerprints = be.snapshot(paths)
    for name, expected in {**captured, **cache_hashes}.items():
        if fingerprints[name] != expected:
            raise ValueError('选例预检期间来源改变。')
    be.verify_snapshot(paths, fingerprints)
    return config, groups['demo'], targets, vectors['demo'], query_vectors, paths, fingerprints


def make_records(config, demos, targets, demo_vectors, query_vectors):
    if len(targets) != len(query_vectors):
        raise ValueError('目标题和查询向量数量不同。')
    if len({target['id'] for target in targets}) != len(targets):
        raise ValueError('目标题ID重复。')
    records = []
    seed = config['demo_selection']['random']['seed']
    for index, target in sorted(enumerate(targets), key=lambda pair: pair[1]['id']):
        minimal = {key: target[key] for key in ('id', 'world_fingerprint')}
        selected = select_for_condition(demos, minimal, seed, 'C1')
        random_items = [{key: item[key] for key in ('id', 'label', 'world_fingerprint')}
                        for item in selected]
        c2_ids = [item['id'] for item in select_for_condition(demos, minimal, seed, 'C2')]
        if [item['id'] for item in random_items] != c2_ids:
            raise ValueError('C1/C2未共享随机示例。')
        semantic = sr.select_for_target(demos, demo_vectors, minimal, query_vectors[index])
        payload = {'selection_schema_version': 1, 'builder_version': VERSION, 'stage': 'test',
                   'question_id': minimal['id'], 'target_world_fingerprint': minimal['world_fingerprint'],
                   'config_hash': digest(config), 'test_embedding_manifest_hash': TEST_EMBEDDING_HASH,
                   'random_algorithm': ALGORITHM_VERSION, 'random_seed': seed,
                   'semantic_algorithm': sr.VERSION,
                   'conditions': {'C0': [], 'C1': random_items,
                                  'C2': copy.deepcopy(random_items), 'C3': semantic}}
        payload['record_hash'] = digest(payload)
        validate_record(payload)
        records.append(payload)
    return records


def validate_record(record):
    payload = dict(record)
    saved = payload.pop('record_hash', None)
    if saved != digest(payload):
        raise ValueError('选例记录指纹错误。')
    if (record['selection_schema_version'] != 1 or record['builder_version'] != VERSION
            or record['stage'] != 'test' or record['random_algorithm'] != ALGORITHM_VERSION
            or record['semantic_algorithm'] != sr.VERSION or record['random_seed'] != 20261009
            or record['test_embedding_manifest_hash'] != TEST_EMBEDDING_HASH
            or set(record['conditions']) != set(CONDITIONS)
            or record['conditions']['C0'] != []
            or record['conditions']['C1'] != record['conditions']['C2']):
        raise ValueError('选例版本、C0或共享规则错误。')
    for condition in ('C1', 'C2', 'C3'):
        items = record['conditions'][condition]
        if (len(items) != 2 or [item['label'] for item in items] != ['True', 'False']
                or len({item['id'] for item in items}) != 2
                or len({item['world_fingerprint'] for item in items}) != 2):
            raise ValueError('示例标签平衡、顺序或独立性错误。')
        for item in items:
            fields = {'id', 'label', 'world_fingerprint'} | ({'similarity'} if condition == 'C3' else set())
            if (set(item) != fields or item['id'] == record['question_id']
                    or item['world_fingerprint'] == record['target_world_fingerprint']):
                raise ValueError('示例字段或目标隔离错误。')
            if condition == 'C3' and (type(item['similarity']) is not float
                    or not math.isfinite(item['similarity']) or not -1 <= item['similarity'] <= 1):
                raise ValueError('余弦相似度不合法。')


def save_records(destination, records, provenance):
    if len(records) != 60 or len({r['question_id'] for r in records}) != 60:
        raise ValueError('要求60条唯一test选例记录。')
    for record in records:
        validate_record(record)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    path = destination / 'selections.jsonl'
    with path.open('x', encoding='utf-8') as file:
        for record in records:
            file.write(be.canonical(record) + '\n')
        file.flush()
        os.fsync(file.fileno())
    if [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] != records:
        raise ValueError('选例保存后重读不一致。')
    manifest = {'test_selection_manifest_version': 1, 'builder_version': VERSION,
                'stage': 'test', 'status': 'complete', 'question_count': 60,
                'planned_responses': 240, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'provenance': provenance, 'selections_sha256': be.file_hash(path)}
    manifest['manifest_hash'] = digest(manifest)
    be.write_json_new(destination / 'selection_manifest.json', manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--write', action='store_true')
    args = parser.parse_args(argv)
    plan.check_environment()
    destination = ROOT / OUTPUT_PATH
    if args.write and (destination.exists() or destination.is_symlink()):
        raise FileExistsError('选例目录已存在，不覆盖或自动续写。')
    config, demos, targets, dv, qv, paths, captured = load_inputs(ROOT)
    records = make_records(config, demos, targets, dv, qv)
    if records != make_records(config, demos, list(reversed(targets)), dv, qv[::-1]):
        raise ValueError('改变目标题处理顺序影响选择。')
    if records != make_records(config, list(reversed(demos)), targets, dv[::-1], qv):
        raise ValueError('改变demo处理顺序影响选择。')
    be.verify_snapshot(paths, captured)
    print('配置、冻结数据、demo/test向量缓存及来源指纹：通过')
    print('选例计划：60条；C0无示例；C1/C2共用随机示例；C3使用语义示例。')
    print('标签平衡、True在前、目标隔离及处理顺序不变性：通过')
    for row in records[:3]:
        print('\n目标题ID：', row['question_id'])
        print('C1/C2随机示例：', [item['id'] for item in row['conditions']['C1']])
        print('C3语义示例：', [(item['id'], round(item['similarity'], 8)) for item in row['conditions']['C3']])
    if not args.write:
        print('\n只读选例预览：PASS；未保存文件、未重新编码、未加载模型权重、未读取密钥、未调用API。')
        print('拟保存目录：', destination)
        return 0
    manifest = save_records(destination, records, {'config': config, 'input_hashes': captured,
                                                  'test_embedding_manifest_hash': TEST_EMBEDDING_HASH})
    be.verify_snapshot(paths, captured)
    print('\n选例缓存验收：PASS；已保存60条四条件选例计划。')
    print('目录：', destination)
    print('清单内部指纹：', manifest['manifest_hash'])
    print('未修改旧缓存或源数据、未重新编码、未保存提示词、未调用API；正式回答仍为0/240。')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('已中断；保留可能存在的部分选例目录，不要删除或重跑。')
        raise SystemExit(130)
    except Exception as exc:
        print('停止：', type(exc).__name__)
        if isinstance(exc, (ValueError, FileExistsError)):
            print(str(exc))
        print('保留现场，不要覆盖、删除或换目录重跑。')
        raise SystemExit(1)
