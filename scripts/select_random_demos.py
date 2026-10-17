"""固定种子、标签平衡的随机2-shot选择器；不调用API。"""

import hashlib
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/inference_dev_v1.json"
ALGORITHM_VERSION = "random_label_balanced_v1_true_first"


def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        records = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path.name}包含重复ID。")

    return records


def load_inputs():
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)

    if config["stage"] != "dev":
        raise ValueError("当前选择器只支持dev阶段。")

    data = config["data"]
    expected_paths = {
        "demo_path": "data/splits/seed20261007/demo.jsonl",
        "eval_path": "data/splits/seed20261007/dev.jsonl",
    }

    if data["allow_test"] is not False:
        raise ValueError("必须禁止使用test。")

    for key, expected in expected_paths.items():
        if data[key] != expected:
            raise ValueError(f"{key}不是本版允许的数据路径。")

    selection = config["demo_selection"]
    expected_selection = {
        "method": "random_label_balanced",
        "shots": 2,
        "true_count": 1,
        "false_count": 1,
        "applies_to_conditions": ["C1", "C2"],
        "share_selection_between": ["C1", "C2"],
    }

    for key, expected in expected_selection.items():
        if selection[key] != expected:
            raise ValueError(f"选例配置{key}不符合本版规则。")

    if type(selection["seed"]) is not int:
        raise ValueError("选例种子必须是整数。")

    demos = read_jsonl(ROOT / data["demo_path"])
    targets = read_jsonl(ROOT / data["eval_path"])

    return config, demos, targets


def select_random_demos(demos, target, seed):
    """仅使用目标题ID和世界指纹，不使用目标答案或参考链。"""
    target_id = target["id"]
    target_world = target["world_fingerprint"]

    seed_text = json.dumps(
        [ALGORITHM_VERSION, seed, target_id],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    seed_digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(seed_digest, "big"))

    selected = []

    for label in ("True", "False"):
        pool = sorted(
            (
                demo
                for demo in demos
                if demo["label"] == label
                and demo["id"] != target_id
                and demo["world_fingerprint"] != target_world
            ),
            key=lambda demo: demo["id"],
        )

        if not pool:
            raise ValueError(f"目标题{target_id}没有可用的{label}示例。")

        selected.append(rng.choice(pool))

    if len({demo["world_fingerprint"] for demo in selected}) != 2:
        raise ValueError("选中的两条示例来自同一个世界。")

    return selected


def select_for_condition(demos, target, seed, condition):
    if condition == "C0":
        return []

    if condition not in ("C1", "C2"):
        raise ValueError("当前只支持C0、C1、C2。")

    # condition不参与随机种子，因此C1/C2选择完全一致。
    return select_random_demos(demos, target, seed)


if __name__ == "__main__":
    config, demos, targets = load_inputs()
    seed = config["demo_selection"]["seed"]

    print("选择算法：", ALGORITHM_VERSION)
    print("随机种子：", seed)
    print("demo数量：", len(demos))
    print("dev数量：", len(targets))

    for target in sorted(targets, key=lambda item: item["id"])[:3]:
        selected = select_for_condition(demos, target, seed, "C1")
        print("\n目标题：", target["id"])
        for demo in selected:
            print("示例：", demo["label"], demo["id"])

    print("\n只读预览：未保存文件、未读取test、未调用API。")
