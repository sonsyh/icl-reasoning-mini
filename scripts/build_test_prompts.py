"""Build frozen C0-C3 test messages; default read-only, never sends API requests."""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import build_test_selections as selections
import build_c3_prompts as cp
from build_prompts import build_messages, hash_messages, PROMPT_VERSION
from deepseek_client import build_request
be = selections.be
VERSION = 'test_prompt_adapter_v1'
OUTPUT_PATH = 'data/prompts/test_four_conditions_v1'
SELECTION_HASH = 'edaea0740a7df6f71b8439a4255dfcd653b09c4c70ebfe1504bbfef953841b68'
CONDITIONS = ('C0', 'C1', 'C2', 'C3')


def digest(value):
    return selections.digest(value)


def read_jsonl(path):
    text = Path(path).read_text(encoding='utf-8')
    if not text.endswith('\n'):
        raise ValueError('缓存尾行不完整。')
    return [json.loads(line) for line in text.splitlines()]


def load_bundle(root):
    root = Path(root)
    config, demos, targets, dv, qv, paths, captured = selections.load_inputs(root)
    folder = root / selections.OUTPUT_PATH
    manifest_path, records_path = folder / 'selection_manifest.json', folder / 'selections.jsonl'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    payload = dict(manifest)
    saved = payload.pop('manifest_hash', None)
    if (saved != SELECTION_HASH or digest(payload) != saved
            or manifest['status'] != 'complete' or manifest['stage'] != 'test'
            or manifest['question_count'] != 60 or manifest['planned_responses'] != 240
            or manifest['builder_version'] != selections.VERSION
            or manifest['provenance'] != {'config': config, 'input_hashes': captured,
                'test_embedding_manifest_hash': selections.TEST_EMBEDDING_HASH}):
        raise ValueError('选例清单或来源与已验收版本不同。')
    selections.plan.verify_file(records_path, manifest['selections_sha256'])
    records = read_jsonl(records_path)
    if records != selections.make_records(config, demos, targets, dv, qv):
        raise ValueError('保存选例与固定算法重建结果不同。')
    baseline_path = root / config['provenance']['baseline_inference_config_path']
    baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
    for key in ('api', 'runtime', 'credentials'):
        if be.canonical(config[key]) != be.canonical(baseline[key]):
            raise ValueError('test推理参数与dev基线不同。')
    paths = dict(paths)
    paths.update({'test_selection_manifest': manifest_path, 'test_selections': records_path,
                  'test_prompt_source': Path(__file__).resolve()})
    return config, baseline, demos, targets, records, paths, be.snapshot(paths)


def render(target, condition, demos):
    if condition not in CONDITIONS:
        raise ValueError('不支持的条件。')
    minimal = {key: copy.deepcopy(target[key]) for key in selections.plan.TARGET_FIELDS}
    return build_messages(minimal, 'C2' if condition == 'C3' else condition, demos)


def request_parameters(config):
    # Pure request assembly using the unchanged validated dev baseline.
    # The caller checks test API/runtime equality; no client is instantiated.
    request = build_request(config, [{'role': 'user', 'content': 'parameter fixture'}])
    return {key: value for key, value in request.items() if key != 'messages'}


def make_records(config, baseline, demos, targets, chosen, count):
    demo_map = {demo['id']: demo for demo in demos}
    target_map = {target['id']: target for target in targets}
    if (len(target_map) != len(targets) or len(chosen) != len(targets)
            or {row['question_id'] for row in chosen} != set(target_map)):
        raise ValueError('选例与目标题集合不一致。')
    parameters = request_parameters(baseline)
    records = []
    original = copy.deepcopy((demos, targets, chosen))
    for selection in sorted(chosen, key=lambda row: row['question_id']):
        selections.validate_record(selection)
        target = target_map[selection['question_id']]
        if selection['target_world_fingerprint'] != target['world_fingerprint']:
            raise ValueError('目标题世界与选例记录不同。')
        messages_by_condition = {}
        for condition in CONDITIONS:
            metadata = selection['conditions'][condition]
            picked = [demo_map[item['id']] for item in metadata]
            for item, demo in zip(metadata, picked):
                if any(item[key] != demo[key] for key in ('id', 'label', 'world_fingerprint')):
                    raise ValueError('选例元数据与demo不一致。')
            messages = render(target, condition, picked)
            expected_roles = ['system', 'user'] if condition == 'C0' else ['system', 'user', 'assistant', 'user', 'assistant', 'user']
            if [msg['role'] for msg in messages] != expected_roles:
                raise ValueError('消息结构错误。')
            poisoned = dict(target, label='DO_NOT_USE', depth=-1, gold_cot=['DO_NOT_USE'])
            if messages != render(poisoned, condition, picked) or messages != render(target, condition, picked):
                raise ValueError('目标题禁止字段影响消息或构建不稳定。')
            tokens = count(messages)
            if type(tokens) is not int or tokens <= 0 or tokens + config['api']['max_tokens'] > 20000:
                raise ValueError('消息计数不合法或超过保守预检阈值；停止，不自动截断。')
            # 20000 is a local conservative guard, not a claimed provider context limit.
            row = {'test_prompt_schema_version': 1, 'builder_version': VERSION, 'stage': 'test',
                   'condition': condition, 'question_id': target['id'],
                   'demo_ids': [demo['id'] for demo in picked],
                   'selection_record_hash': selection['record_hash'],
                   'config_hash': digest(config), 'base_prompt_version': PROMPT_VERSION,
                   'prompt': messages, 'prompt_hash': hash_messages(messages),
                   'input_tokens_offline': tokens, 'request_parameters': parameters}
            row['record_hash'] = digest(row)
            validate_record(row)
            records.append(row)
            messages_by_condition[condition] = messages
        p = messages_by_condition
        if (p['C0'][0] != p['C1'][0] or p['C2'][0] != p['C3'][0]
                or any(p[c][-1] != p['C0'][-1] for c in CONDITIONS)
                or p['C1'][1] != p['C2'][1] or p['C1'][3] != p['C2'][3]):
            raise ValueError('条件间任务说明、目标或共享示例题面不同。')
    if (demos, targets, chosen) != original:
        raise ValueError('构建修改了原始数据。')
    return records


def validate_record(record):
    payload = dict(record)
    saved = payload.pop('record_hash', None)
    if (saved != digest(payload) or record['prompt_hash'] != hash_messages(record['prompt'])
            or record['stage'] != 'test' or record['condition'] not in CONDITIONS
            or record['builder_version'] != VERSION or record['base_prompt_version'] != PROMPT_VERSION
            or type(record['input_tokens_offline']) is not int or record['input_tokens_offline'] <= 0
            or len(record['demo_ids']) != (0 if record['condition'] == 'C0' else 2)):
        raise ValueError('提示词记录、消息指纹或元数据错误。')
    forbidden = {'label', 'gold_label', 'depth', 'gold_cot'}
    if forbidden & set(record):
        raise ValueError('提示词缓存不应保存目标答案、深度或参考链。')


def token_report(records):
    result = {}
    for condition in CONDITIONS:
        values = [r['input_tokens_offline'] for r in records if r['condition'] == condition]
        if not values:
            raise ValueError('缺少条件。')
        result[condition] = {'n': len(values), 'min': min(values), 'median': statistics.median(values),
                             'mean': statistics.mean(values), 'max': max(values), 'total': sum(values)}
    change = (result['C3']['mean'] / result['C2']['mean'] - 1) * 100
    result['C3_vs_C2_mean_change_percent'] = change
    result['length_difference_over_20_percent'] = abs(change) > 20
    return result


def save_records(destination, records, provenance):
    if (len(records) != 240 or len({(r['question_id'], r['condition']) for r in records}) != 240
            or any(sum(r['condition'] == c for r in records) != 60 for c in CONDITIONS)):
        raise ValueError('要求完整的60题×四条件240组消息。')
    for row in records:
        validate_record(row)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    path = destination / 'prompts.jsonl'
    with path.open('x', encoding='utf-8') as file:
        for row in records:
            file.write(be.canonical(row) + '\n')
        file.flush()
        os.fsync(file.fileno())
    if read_jsonl(path) != records:
        raise ValueError('提示词保存后重读不一致。')
    manifest = {'test_prompt_manifest_version': 1, 'builder_version': VERSION,
                'stage': 'test', 'status': 'complete', 'question_count': 60, 'prompt_count': 240,
                'created_at_utc': datetime.now(timezone.utc).isoformat(), 'provenance': provenance,
                'token_report': token_report(records), 'prompts_sha256': be.file_hash(path)}
    manifest['manifest_hash'] = digest(manifest)
    be.write_json_new(destination / 'prompt_manifest.json', manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--write', action='store_true')
    args = parser.parse_args(argv)
    selections.plan.check_environment()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
    destination = ROOT / OUTPUT_PATH
    if args.write and (destination.exists() or destination.is_symlink()):
        raise FileExistsError('提示词缓存已存在，不覆盖或自动续写。')
    config, baseline, demos, targets, chosen, paths, captured = load_bundle(ROOT)
    count = cp.load_counter(ROOT)
    records = make_records(config, baseline, demos, targets, chosen, count)
    if records != make_records(config, baseline, demos[::-1], targets[::-1], chosen[::-1], count):
        raise ValueError('改变处理顺序影响提示词。')
    be.verify_snapshot(paths, captured)
    print('test提示词离线验收：PASS；60题×四条件=240组消息。')
    print('C1/C2示例ID及题面一致；C2/C3系统指令一致；四组目标一致：通过')
    print('目标禁止字段隔离、构建可重复、源记录未修改：通过')
    print('完整消息token统计（DeepSeek-V4 chat，含消息标记，未重复添加特殊token）：')
    report = token_report(records)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report['length_difference_over_20_percent']:
        print('长度差异超过20%：需记录混杂，不自动更改已锁定的方法或提示词。')
    if not args.write:
        print('只读预览：未保存提示词、未展示test题面/答案、未重新编码、未加载模型权重、未读取密钥、未调用API。')
        print('拟保存目录：', destination)
        return 0
    manifest = save_records(destination, records, {'config': config, 'input_hashes': captured,
                                                  'selection_manifest_hash': SELECTION_HASH,
                                                  'counting_mode': 'DeepSeek-V4 chat; drop_thinking=True; BOS=True; tokenizer special_tokens=False'})
    be.verify_snapshot(paths, captured)
    print('提示词缓存保存：PASS；240组消息及长度报告已保存。')
    print('目录：', destination)
    print('清单内部指纹：', manifest['manifest_hash'])
    print('未修改旧文件、未读取密钥、未调用API；正式回答仍为0/240。')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('已中断；保留可能存在的缓存，不要删除或重跑。')
        raise SystemExit(130)
    except Exception as exc:
        print('停止：', type(exc).__name__)
        if isinstance(exc, (ValueError, FileExistsError)):
            print(str(exc))
        print('保留现场，不要覆盖、删除或换目录重跑。')
        raise SystemExit(1)
