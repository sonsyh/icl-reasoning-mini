"""Day5 plan preparation, checkpoint 1: offline read-only preflight only.

No weights, vectors, selections, prompts or API requests are generated here.
The later write stage must be reviewed separately before it is implemented.
"""
import argparse
import copy
from importlib.metadata import version
import json
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import semantic_retrieval as sr

be = sr.be
VERSION = 'test_plan_preflight_v1'
CONFIG_PATH = 'configs/inference_test_v1.json'
CONFIG_SHA256 = '288914d5a5c2712973187f769773d44c6a50d7bbca635cf482aa45f453ad9dd0'
ACCEPTANCE_SHA256 = 'd24c0f3f2abdbfefb86cc7795484e233197f497921fad406b36c98780df0b270'
FREEZE_SHA256 = '443ec302948a1c7b049206d841f76a97ea3ccc1ca156f9265c2d9c354d85cfc3'
TEST_SHA256 = 'ee97abe51219de5cae098789c78d3daeddcdea7c23a9dc7ce685d84ea9b20ad7'
TARGET_FIELDS = ('id', 'world_fingerprint', 'premises', 'query')


def verify_file(path, expected):
    if be.file_hash(path) != expected:
        raise ValueError('文件指纹改变：' + str(path))


def validate_settings(config, baseline):
    if (config['stage'] != 'test' or config['conditions'] != ['C0', 'C1', 'C2', 'C3']
            or config['data']['allow_test'] is not True
            or type(config['data']['expected_eval_count']) is not int
            or config['data']['expected_eval_count'] != 60
            or config['data']['expected_demo_count'] != 120
            or config['data']['eval_path'] != 'data/splits/seed20261007/test.jsonl'
            or config['evaluation']['planned_responses'] != 240):
        raise ValueError('不是已固定的60题、四条件test设置。')
    for name in ('api', 'runtime', 'credentials'):
        if be.canonical(config[name]) != be.canonical(baseline[name]):
            raise ValueError('与Day3固定参数不同：' + name)
    if (config['demo_selection']['random']['seed'] != 20261009
            or config['demo_selection']['semantic']['method'] != sr.VERSION
            or config['evaluation']['target_input_fields'] != list(TARGET_FIELDS)):
        raise ValueError('选例规则或允许的目标题字段改变。')


def minimal_targets(rows):
    # Discard test gold labels, depths and proofs before the planning pipeline.
    targets = []
    for row in rows:
        target = {key: copy.deepcopy(row[key]) for key in TARGET_FIELDS}
        if not all(isinstance(target[key], str) and target[key]
                   for key in ('id', 'world_fingerprint')):
            raise ValueError('test题目ID或世界指纹不合法。')
        be.render_text(target)
        targets.append(target)
    for key in ('id', 'world_fingerprint'):
        if len({row[key] for row in targets}) != len(targets):
            raise ValueError('test存在重复ID或世界。')
    return targets


def check_isolation(targets, demo, dev):
    for key in ('id', 'world_fingerprint'):
        values = {row[key] for row in targets}
        if values & {row[key] for row in demo} or values & {row[key] for row in dev}:
            raise ValueError('test与demo/dev存在交叉：' + key)


def prepare(root):
    root = Path(root)
    config_path = root / CONFIG_PATH
    verify_file(config_path, CONFIG_SHA256)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    acceptance_path = root.parent / 'markdown/Day4_Acceptance_2026-10-13.json'
    verify_file(acceptance_path, ACCEPTANCE_SHA256)
    acceptance = json.loads(acceptance_path.read_text(encoding='utf-8'))
    paths = {name: root / name for name in acceptance['file_sha256']}
    for name, expected in acceptance['file_sha256'].items():
        verify_file(paths[name], expected)
    reference, groups, vectors, cache_paths, captured = sr.load_inputs(root)
    semantic = config['demo_selection']['semantic']
    if (semantic['reference_config_path'] != be.CONFIG_PATH
            or semantic['reference_config_sha256'] != be.file_hash(root / be.CONFIG_PATH)
            or semantic['inherit_sections'] != ['model', 'input_text', 'encoding', 'retrieval']
            or semantic['demo_embedding_cache_path'] != be.CACHE_PATH
            or semantic['demo_embedding_manifest_hash'] != sr.EXPECTED_CACHE_HASH):
        raise ValueError('Day4检索资源引用不一致。')
    baseline = json.loads((root / 'configs/inference_dev_v1.json').read_text(encoding='utf-8'))
    validate_settings(config, baseline)
    freeze_path = root / 'data/splits/seed20261007/freeze_record.json'
    test_path = root / 'data/splits/seed20261007/test.jsonl'
    verify_file(freeze_path, FREEZE_SHA256)
    freeze = json.loads(freeze_path.read_text(encoding='utf-8'))
    if (freeze['test_frozen'] is not True
            or freeze['dataset_files_sha256']['test.jsonl'] != TEST_SHA256
            or config['data']['expected_test_sha256'] != TEST_SHA256):
        raise ValueError('test冻结信息不一致。')
    # Verify the frozen bytes before parsing; never print test content/gold.
    verify_file(test_path, TEST_SHA256)
    text = test_path.read_text(encoding='utf-8')
    if not text.endswith('\n'):
        raise ValueError('test文件尾行不完整。')
    targets = minimal_targets([json.loads(line) for line in text.splitlines()])
    if len(targets) != 60:
        raise ValueError('test不是60题。')
    check_isolation(targets, groups['demo'], groups['dev'])
    paths.update(cache_paths)
    paths.update({'test_config': config_path, 'test_data': test_path,
                  'freeze_record': freeze_path, 'day4_acceptance': acceptance_path,
                  'test_preflight_source': Path(__file__).resolve()})
    fingerprints = be.snapshot(paths)
    for name, expected in captured.items():
        if fingerprints[name] != expected:
            raise ValueError('预检期间来源文件改变。')
    return config, reference, targets, paths, fingerprints


def inspect_lengths(targets, tokenizer, limit=256):
    original = copy.deepcopy(targets)
    index, texts = be.build_index(targets, tokenizer, limit)
    if targets != original:
        raise ValueError('长度检查修改了目标题。')
    if not index or any(type(row['token_count']) is not int or row['token_count'] <= 0 for row in index):
        raise ValueError('分词计数为空或不合法。')
    lengths = [row['token_count'] for row in index]
    return {'count': len(index), 'min': min(lengths), 'median': statistics.median(lengths),
            'max': max(lengths), 'limit': limit, 'overlength_count': 0}


def check_environment():
    if Path(sys.prefix).name != 'icl-reasoning-retrieval':
        raise ValueError('请激活icl-reasoning-retrieval环境。')
    if os.environ.get('KMP_DUPLICATE_LIB_OK'):
        raise ValueError('禁止OpenMP重复加载绕过。')
    if {name: version(name) for name in be.PACKAGES} != be.PACKAGES:
        raise ValueError('计算环境版本改变。')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    check_environment()
    config, reference, targets, paths, fingerprints = prepare(ROOT)
    print('冻结数据、Day3/Day4来源与demo向量缓存指纹：通过', flush=True)
    print('test与demo/dev的ID及世界隔离：通过', flush=True)
    print('正在离线加载分词器检查长度；不加载模型权重……', flush=True)
    # Explicitly local, no network fallback, no executable model repository code.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / reference['model']['local_path']),
        local_files_only=True, trust_remote_code=False, token=False)
    report = inspect_lengths(targets, tokenizer, reference['input_text']['max_seq_length'])
    be.verify_snapshot(paths, fingerprints)
    print('Day5只读预检：PASS')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print('规划范围：60道test × C0/C1/C2/C3 = 240组（尚未构建）。')
    print('目标规划字段仅id/world_fingerprint/premises/query；答案、深度、参考链未参与。')
    print('未展示test题面/答案、未计算或保存向量、未选例、未保存提示词、未读取密钥、未调用API。')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        # Avoid printing arbitrary payloads containing test contents or secrets.
        print('停止：预检未通过；异常类型：', type(exc).__name__)
        if isinstance(exc, ValueError):
            print(str(exc))
        raise SystemExit(1)
