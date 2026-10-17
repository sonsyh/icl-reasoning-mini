"""C3 prompt adapter: reuse pinned C2 rendering, no API calls. Default preview."""
import argparse
import copy
from datetime import datetime, timezone
import importlib.util
from importlib.metadata import version
import json
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(os.environ.get("ICL_REASONING_PROJECT_ROOT",
                          str(Path(__file__).resolve().parents[1]))).resolve()
sys.path.insert(0, str(ROOT / "scripts"))
import semantic_retrieval as sr
from build_prompts import build_messages, hash_messages, PROMPT_VERSION
from deepseek_client import build_request, validate_config
from experiment_log import read_records

be = sr.be
VERSION = "c3_prompt_adapter_v1"
SELECTION_HASH = "332569ad86fc4ee0d496078aa5b4265c10434bd2545ca4fd9a497a73affe6a51"
DAY3_HASH = "99fdc1f78f7b769b1b82b743df799ced61e075109f810209b3336fde44fd9ba5"
INFERENCE_PATH = "configs/inference_dev_v1.json"
DAY3_PATH = "runs/day3_dev_live_v1_20261008/results.jsonl"
OUTPUT_PATH = "data/prompts/dev_c3_minilm_v1"
TOKENIZER_PATH = "external/deepseek_v4_tokenizer_b5968e9"
TOKENIZER_HASHES = {
    "tokenizer.json": "8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf",
    "tokenizer_config.json": "6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547",
    "encoding/encoding_dsv4.py": "bdbd57c132a1b3725042323d02b98b9d1df28e5f388f134399555d041f5055e0",
}


def build_c3_messages(target, demos):
    # The metadata condition is C3. C2 here selects only the existing CoT format;
    # build_messages does NOT select demos and never calls a model.
    minimal = {key: target[key] for key in ("id", "world_fingerprint", "premises", "query")}
    return build_messages(minimal, "C2", demos)


def validate_selection_manifest(manifest):
    payload = dict(manifest)
    digest = payload.pop("manifest_hash", None)
    if (digest != be.digest_text(be.canonical(payload)) or digest != SELECTION_HASH
            or manifest.get("status") != "complete" or manifest.get("stage") != "dev"
            or manifest.get("question_count") != 20
            or manifest.get("algorithm_version") != sr.VERSION):
        raise ValueError("选例缓存清单与已验收记录不一致。")


def load_bundle(root):
    config, groups, vectors, paths, fingerprints = sr.load_inputs(root)
    folder = root / sr.OUTPUT_PATH
    manifest_path = folder / "selection_manifest.json"
    selection_path = folder / "selections.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_selection_manifest(manifest)
    if (manifest["provenance"]["input_hashes"] != fingerprints
            or be.file_hash(selection_path) != manifest["selections_sha256"]):
        raise ValueError("选例文件或其来源指纹改变。")
    text = selection_path.read_text(encoding="utf-8")
    if not text.endswith("\n"):
        raise ValueError("选例尾行不完整。")
    selections = [json.loads(line) for line in text.splitlines()]
    if selections != sr.make_plan(config, groups, vectors):
        raise ValueError("保存选例与当前固定算法重建结果不同。")

    inference_path, day3_path = root / INFERENCE_PATH, root / DAY3_PATH
    if be.file_hash(day3_path) != DAY3_HASH:
        raise ValueError("Day3结果文件已改变。")
    inference = json.loads(inference_path.read_text(encoding="utf-8"))
    validate_config(inference)
    baseline = read_records(day3_path)
    if len(baseline) != 60 or any(r["is_simulated"] is not False for r in baseline):
        raise ValueError("Day3不是已验收的60条真实记录。")
    c2_rows = [row for row in baseline if row["condition"] == "C2"]
    if len(c2_rows) != 20 or len({row["question_id"] for row in c2_rows}) != 20:
        raise ValueError("C2记录不完整或重复。")
    baseline_map = {row["question_id"]: row for row in c2_rows}
    if set(baseline_map) != {row["id"] for row in groups["dev"]}:
        raise ValueError("C2题目集合与dev不同。")
    demos = {row["id"]: row for row in groups["demo"]}
    for target in groups["dev"]:
        row = baseline_map[target["id"]]
        if row["config_hash"] != be.digest_text(be.canonical(inference)):
            raise ValueError("当前推理配置与Day3不同。")
        for name, expected in row["artifact_hashes"].items():
            if be.file_hash(root / name) != expected:
                raise ValueError("Day3固定文件指纹改变：" + name)
            paths["day3_artifact/" + name] = root / name
        reconstructed = build_messages(target, "C2", [demos[qid] for qid in row["demo_ids"]])
        if reconstructed != row["prompt"]:
            raise ValueError("重新构建的C2提示词与历史记录不同。")
    paths = dict(paths)
    paths["selection_manifest"] = manifest_path
    paths["selections"] = selection_path
    paths["inference_config"] = inference_path
    paths["day3_results"] = day3_path
    paths["c3_adapter_source"] = Path(__file__).resolve()
    tokenizer_dir = root / TOKENIZER_PATH
    for name, digest in TOKENIZER_HASHES.items():
        path = tokenizer_dir / name
        if be.file_hash(path) != digest:
            raise ValueError("DeepSeek本地分词资源指纹改变：" + name)
        paths["deepseek_tokenizer/" + name] = path
    # config is also used by AutoTokenizer; pin its current bytes in provenance.
    paths["deepseek_tokenizer/config.json"] = tokenizer_dir / "config.json"
    captured = be.snapshot(paths)
    for name, digest in fingerprints.items():
        if captured[name] != digest:
            raise ValueError("验收期间缓存来源改变。")
    be.verify_snapshot(paths, captured)
    return inference, groups, selections, baseline_map, paths, captured


def count_messages(messages, tokenizer, encode_messages):
    before = copy.deepcopy(messages)
    encoded = encode_messages(messages, thinking_mode="chat", drop_thinking=True,
                              add_default_bos_token=True)
    if messages != before or not isinstance(encoded, str) or not encoded.endswith("</think>"):
        raise ValueError("消息编码被修改或非思考模式结束标记异常。")
    ids = tokenizer.encode(encoded, add_special_tokens=False, truncation=False)
    if not ids:
        raise ValueError("分词结果为空。")
    return len(ids)


def load_counter(root):
    # Execute only the previously audited, hash-pinned official encoding module.
    directory = root / TOKENIZER_PATH
    for name, digest in TOKENIZER_HASHES.items():
        if be.file_hash(directory / name) != digest:
            raise ValueError("分词资源指纹改变：" + name)
    spec = importlib.util.spec_from_file_location(
        "_pinned_dsv4_c3_counting", directory / "encoding/encoding_dsv4.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True,
                                              trust_remote_code=False, token=False)
    return lambda messages: count_messages(messages, tokenizer, module.encode_messages)


def audit_case(target, demos, selection, baseline, inference, count):
    if selection["question_id"] != target["id"]:
        raise ValueError("选例记录与目标题不对应。")
    if selection["demo_ids"] != [demo["id"] for demo in demos]:
        raise ValueError("提示词示例与语义选例记录不一致。")
    original_target, original_demos = copy.deepcopy(target), copy.deepcopy(demos)
    messages = build_c3_messages(target, demos)
    if [m["role"] for m in messages] != ["system", "user", "assistant", "user", "assistant", "user"]:
        raise ValueError("C3消息结构错误。")
    if messages[0] != baseline["prompt"][0] or messages[-1] != baseline["prompt"][-1]:
        raise ValueError("C3与C2的任务指令或目标题不同。")
    poisoned = dict(target, label="DO_NOT_USE", depth=-1, gold_cot=["DO_NOT_USE"])
    if messages != build_c3_messages(poisoned, demos):
        raise ValueError("目标题的答案/深度/参考链影响提示词。")
    if messages != build_messages(target, "C2", demos):
        raise ValueError("C3不复用原C2模板。")
    request = build_request(inference, messages)
    parameters = {key: value for key, value in request.items() if key != "messages"}
    if parameters != baseline["inference_parameters"]:
        raise ValueError("C3与C2推理参数不同。")
    c2_count, c3_count = count(baseline["prompt"]), count(messages)
    if type(c2_count) is not int or type(c3_count) is not int or min(c2_count, c3_count) <= 0:
        raise ValueError("token计数必须为正整数。")
    if messages != build_c3_messages(target, demos) or target != original_target or demos != original_demos:
        raise ValueError("构建不稳定或修改了原始记录。")
    result = {
        "prompt_cache_schema_version": 1, "stage": "dev", "condition": "C3",
        "adapter_version": VERSION, "base_prompt_version": PROMPT_VERSION,
        "question_id": target["id"], "demo_ids": selection["demo_ids"],
        "selection_record_hash": selection["record_hash"],
        "prompt": messages, "prompt_hash": hash_messages(messages),
        "input_tokens_offline": c3_count, "request_parameters": parameters,
        "inference_config_hash": be.digest_text(be.canonical(inference)),
        "baseline_c2": {"prompt_hash": baseline["prompt_hash"],
                        "input_tokens_offline": c2_count,
                        "input_tokens_reported": baseline["token_usage"]["prompt_tokens"]},
    }
    result["record_hash"] = be.digest_text(be.canonical(result))
    return result


def validate_prompt_record(record):
    payload = dict(record)
    digest = payload.pop("record_hash", None)
    if digest != be.digest_text(be.canonical(payload)) or record["prompt_hash"] != hash_messages(record["prompt"]):
        raise ValueError("提示词记录或消息指纹错误。")
    if (record["stage"] != "dev" or record["condition"] != "C3"
            or record["adapter_version"] != VERSION
            or len(record["demo_ids"]) != 2 or len(set(record["demo_ids"])) != 2
            or type(record["input_tokens_offline"]) is not int
            or record["input_tokens_offline"] <= 0):
        raise ValueError("提示词缓存元数据错误。")


def token_report(records):
    c2 = [row["baseline_c2"]["input_tokens_offline"] for row in records]
    c3 = [row["input_tokens_offline"] for row in records]
    if not c2 or len(c2) != len(c3) or min(c2 + c3) <= 0:
        raise ValueError("token统计输入不合法。")
    def stats(values):
        return {"n": len(values), "min": min(values), "median": statistics.median(values),
                "mean": statistics.mean(values), "max": max(values), "total": sum(values)}
    change = (statistics.mean(c3) / statistics.mean(c2) - 1) * 100
    return {"C2": stats(c2), "C3": stats(c3), "mean_change_percent": change,
            "length_difference_over_20_percent": abs(change) > 20,
            "C2_offline_matches_reported": sum(
                row["baseline_c2"]["input_tokens_offline"] == row["baseline_c2"]["input_tokens_reported"]
                for row in records)}


def save_prompts(destination, records, provenance):
    if not records or len({r["question_id"] for r in records}) != len(records):
        raise ValueError("提示词记录为空或重复。")
    for record in records:
        validate_prompt_record(record)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    path = destination / "prompts.jsonl"
    with path.open("x", encoding="utf-8") as file:
        for record in records:
            file.write(be.canonical(record) + "\n")
        file.flush()
        os.fsync(file.fileno())
    if [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] != records:
        raise ValueError("提示词保存后重读不一致。")
    manifest = {
        "prompt_manifest_version": 1, "status": "complete", "stage": "dev",
        "condition": "C3", "adapter_version": VERSION, "question_count": len(records),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance, "token_report": token_report(records),
        "prompts_sha256": be.file_hash(path),
    }
    manifest["manifest_hash"] = be.digest_text(be.canonical(manifest))
    be.write_json_new(destination / "prompt_manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--write", action="store_true")
    parser.add_argument("--show-first", action="store_true")
    args = parser.parse_args(argv)
    if Path(sys.prefix) != Path("/opt/icl-reasoning-retrieval"):
        raise ValueError("请使用icl-reasoning-retrieval环境。")
    if {name: version(name) for name in be.PACKAGES} != be.PACKAGES:
        raise ValueError("依赖版本与已验收环境不同。")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    destination = ROOT / OUTPUT_PATH
    if args.write and destination.exists():
        raise FileExistsError("提示词目录已存在，禁止覆盖：" + str(destination))
    inference, groups, selections, baseline, paths, captured = load_bundle(ROOT)
    count = load_counter(ROOT)
    demos = {row["id"]: row for row in groups["demo"]}
    targets = {row["id"]: row for row in groups["dev"]}
    records = [audit_case(targets[row["question_id"]], [demos[qid] for qid in row["demo_ids"]],
                          row, baseline[row["question_id"]], inference, count)
               for row in selections]
    if len(records) != 20:
        raise ValueError("C3提示词必须为20组。")
    for record in records:
        validate_prompt_record(record)
    be.verify_snapshot(paths, captured)
    report = token_report(records)
    print("离线提示词验收：PASS；C3提示词20组")
    print("C2/C3任务说明、目标题与请求参数一致：通过")
    print("语义示例ID/顺序、参考链展示、目标题答案不影响提示词：通过")
    print("构建可重复、原始记录未修改：通过")
    print("完整消息token统计（DeepSeek-V4 chat，本地计数，包含消息标记）：")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    first = records[0]
    print("预览目标题：", first["question_id"])
    print("C3示例ID：", first["demo_ids"])
    print("消息SHA-256：", first["prompt_hash"])
    if args.show_first:
        for index, message in enumerate(first["prompt"], 1):
            print(f"\n[{index}: {message['role']}]\n{message['content']}")
    if report["length_difference_over_20_percent"]:
        print("注意：平均长度差超过20%提醒阈值，需讨论并记录；不会自动修改旧提示词。")
    if not args.write:
        print("只读预览：未保存提示词、未读取test、未加载模型权重、未调用API。")
        print("拟保存目录：", destination)
        return 0
    manifest = save_prompts(destination, records, {
        "input_hashes": captured, "base_inference_config": inference,
        "selection_manifest_hash": SELECTION_HASH,
        "counting_mode": "DeepSeek-V4 chat, add_special_tokens=False, no truncation",
        "packages": {name: version(name) for name in be.PACKAGES},
    })
    print("\n提示词缓存保存：PASS；20组C3消息与长度报告已保存")
    print("目录：", destination)
    print("清单指纹：", manifest["manifest_hash"])
    print("未修改Day3配置/源码/结果或检索缓存、未读取test、未调用API；没有C3回答。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("已中断；如有部分提示词目录请保留，不要自动重跑或删除。")
        raise SystemExit(130)
    except Exception as exc:
        print("停止：", type(exc).__name__, str(exc))
        print("不要覆盖、删除或自动重写已有实验文件。")
        raise SystemExit(1)
