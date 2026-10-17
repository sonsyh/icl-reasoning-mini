"""Day6 descriptive plots from frozen Day5 records. Default is read-only.

No API clients, model imports, secret reads, or changes to experiment artifacts.
Matplotlib is imported only when rendering; tests render into temporary folders.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

from answer_parser import score_answer

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "day5_test_live_v1_20261013"
RUN_PATH = Path("runs") / RUN_ID
TEST_PATH = Path("data/splits/seed20261007/test.jsonl")
ACCEPTANCE_PATH = ROOT.parent / "markdown/Day5_Acceptance_2026-10-13.json"
ACCEPTANCE_SHA256 = "572d45f9028a012598e2cdf0dcd02d0b1e7e4af13d5ad7b64c793a6844cfd45f"
OUTPUT_PATH = ROOT.parent / "figures/day6_test_v1"
CONDITIONS = ("C0", "C1", "C2", "C3")
DEPTHS = (2, 3, 4)
VERSION = "day6_formal_test_plots_v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_jsonl(path):
    text = Path(path).read_text(encoding="utf-8")
    require(bool(text) and text.endswith("\n"), "JSONL为空或尾行不完整，不自动修复。")
    rows = [json.loads(line) for line in text.splitlines()]
    require(all(isinstance(row, dict) for row in rows), "JSONL记录必须为对象。")
    return rows


def verify_sources(root, acceptance):
    hashes = acceptance["file_sha256"]
    require(len(hashes) == acceptance["unique_file_count"], "验收文件数量不一致。")
    required = {str(TEST_PATH), str(RUN_PATH / "results.jsonl"),
                "scripts/answer_parser.py", str(RUN_PATH / "run.lock")}
    require(required <= set(hashes), "验收清单缺少必需来源。")
    for relative, expected in hashes.items():
        path = Path(root) / relative
        require(path.is_file() and not path.is_symlink(), f"来源缺失或为符号链接：{relative}")
        require(file_hash(path) == expected, f"来源指纹改变：{relative}")
    return len(hashes)


@contextmanager
def shared_run_lock(root):
    path = Path(root) / RUN_PATH / "run.lock"
    require(path.is_file() and not path.is_symlink(), "运行锁不存在或为符号链接。")
    with path.open("rb") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("正式运行正在使用日志，停止绘图核查。") from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def summarize(records, targets):
    """Re-score records; reject missing/duplicate pairs and invalid metadata."""
    require(len(targets) == 60, "正式test必须为60道题。")
    target_by_id = {}
    for target in targets:
        qid = target["id"]
        require(qid not in target_by_id, "test题目ID重复。")
        require(type(target["depth"]) is int and target["depth"] in DEPTHS, "深度必须为2/3/4整数。")
        require(target["label"] in ("True", "False"), "标准标签非法。")
        target_by_id[qid] = target
    require(Counter(t["depth"] for t in targets) == {2: 20, 3: 20, 4: 20}, "各深度必须为20题。")
    require(Counter(t["label"] for t in targets) == {"True": 30, "False": 30}, "test标签分布改变。")
    require(len(records) == 240, "正式回答必须为240条。")
    scores, response_ids = {}, set()
    for row in records:
        payload = dict(row)
        saved_hash = payload.pop("record_hash", None)
        require(saved_hash == digest(payload), "回答记录内部指纹改变。")
        qid, condition = row["question_id"], row["condition"]
        require(qid in target_by_id and condition in CONDITIONS, "未知题目或条件。")
        key = (qid, condition)
        require(key not in scores, "同题同条件回答重复。")
        require(row["stage"] == "test" and row["run_id"] == RUN_ID, "混入其他运行记录。")
        require(row["is_simulated"] is False and row["status"] == "response", "只接受正式真实回答。")
        require(type(row["attempt_no"]) is int and row["attempt_no"] == 1, "本次正式记录必须为首次尝试。")
        require(row["gold_label"] == target_by_id[qid]["label"], "保存标准答案不一致。")
        response = row["response"]
        rid = response["response_id"]
        require(isinstance(rid, str) and bool(rid) and rid not in response_ids, "响应ID缺失或重复。")
        response_ids.add(rid)
        require(response["model"] == "deepseek-v4-pro" and response["finish_reason"] == "stop",
                "模型别名改变或响应未正常结束。")
        actual = score_answer(response["raw_output"], target_by_id[qid]["label"],
                              finish_reason=response["finish_reason"])
        require(actual == row["score"], "重新评分与保存评分不一致。")
        scores[key] = actual
    require(set(scores) == {(qid, c) for qid in target_by_id for c in CONDITIONS}, "四条件覆盖不完整。")

    def count(qids, c):
        n = len(qids)
        fields = ("strict_correct", "lenient_correct", "strict_format_ok")
        result = {"n": n, **{f: sum(scores[(q, c)][f] for q in qids) for f in fields}}
        result["parse_ok"] = sum(scores[(q, c)]["parse_status"] == "ok" for q in qids)
        result["strict_accuracy_percent"] = 100 * result["strict_correct"] / n
        result["lenient_accuracy_percent"] = 100 * result["lenient_correct"] / n
        return result

    conditions = {c: count(list(target_by_id), c) for c in CONDITIONS}
    depth = {str(d): {c: count([q for q, t in target_by_id.items() if t["depth"] == d], c)
                     for c in CONDITIONS} for d in DEPTHS}
    return {"plot_version": VERSION, "run_id": RUN_ID, "stage": "test",
            "question_count": 60, "response_count": 240, "conditions": conditions, "depth": depth,
            "all_strict_lenient_equal": all(scores[k]["strict_correct"] == scores[k]["lenient_correct"]
                                             for k in scores),
            "description": "Single frozen test run; descriptive accuracy only; no significance or internal-mechanism claim."}


def compare_acceptance(report, acceptance):
    require(acceptance["status"] == "PASS" and acceptance["run_id"] == RUN_ID,
            "只接受已通过的Day5正式验收。")
    for c in CONDITIONS:
        for key in ("n", "strict_correct", "lenient_correct", "strict_format_ok", "parse_ok"):
            require(report["conditions"][c][key] == acceptance["conditions"][c][key],
                    f"总体统计与验收不一致：{c}/{key}")
    for d in DEPTHS:
        for c in CONDITIONS:
            require(report["depth"][str(d)][c]["n"] == acceptance["depth"][str(d)]["n"],
                    "按深度题数与验收不一致。")
            require(report["depth"][str(d)][c]["strict_correct"] ==
                    acceptance["depth"][str(d)]["strict_correct"][c], "按深度答对数与验收不一致。")


def load_statistics(root=ROOT, acceptance_path=ACCEPTANCE_PATH,
                    expected_acceptance_hash=ACCEPTANCE_SHA256):
    acceptance_path = Path(acceptance_path)
    require(file_hash(acceptance_path) == expected_acceptance_hash, "Day5验收文件指纹改变。")
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    with shared_run_lock(root):
        protected_count = verify_sources(root, acceptance)
        report = summarize(read_jsonl(Path(root) / RUN_PATH / "results.jsonl"),
                           read_jsonl(Path(root) / TEST_PATH))
        compare_acceptance(report, acceptance)
        verify_sources(root, acceptance)
        require(file_hash(acceptance_path) == expected_acceptance_hash, "读取期间验收文件改变。")
    report["protected_file_count"] = protected_count
    # Preserve the hashes verified under the shared lock, not a later unlocked read.
    report["source_sha256"] = {"acceptance": expected_acceptance_hash,
        "results": acceptance["file_sha256"][str(RUN_PATH / "results.jsonl")],
        "test": acceptance["file_sha256"][str(TEST_PATH)]}
    return report


def make_figures(report):
    # No GUI, remote resources, Chinese-font dependency, or model imports.
    import matplotlib
    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    style = {"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
             "axes.spines.right": False, "svg.fonttype": "none"}
    with plt.rc_context(style):
        first, ax = plt.subplots(figsize=(9, 5.7))
        xs = list(range(4))
        strict = [report["conditions"][c]["strict_accuracy_percent"] for c in CONDITIONS]
        lenient = [report["conditions"][c]["lenient_accuracy_percent"] for c in CONDITIONS]
        for delta, values, color, label in ((-0.18, strict, "#27649C", "Strict"),
                                           (0.18, lenient, "#C87B20", "Lenient")):
            bars = ax.bar([x + delta for x in xs], values, width=0.34, color=color, label=label)
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, value - 5, f"{value:.2f}%",
                        ha="center", va="top", color="white", weight="bold", fontsize=10)
        ax.set_xticks(xs, ["C0\nZero-shot", "C1\nRandom answers", "C2\nRandom CoT", "C3\nSemantic CoT"])
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 20))
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Formal test accuracy by condition", loc="left", pad=16, weight="bold")
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", bbox_to_anchor=(0, 1.015), frameon=False, ncols=2)
        first.text(0.10, 0.06, "Frozen test: n=60 per condition | Strict and lenient agree on all 240 responses\n"
                   "deepseek-v4-pro | thinking disabled | temperature=0 | two-shot for C1-C3",
                   fontsize=9, color="#444444")
        first.subplots_adjust(left=0.10, right=0.97, bottom=0.22, top=0.87)

        second, ax = plt.subplots(figsize=(9, 5.7))
        for c, color, marker, line, width in (("C2", "#27649C", "o", "-", 3.0),
                                             ("C3", "#C87B20", "D", "--", 1.8)):
            values = [report["depth"][str(d)][c]["strict_accuracy_percent"] for d in DEPTHS]
            ax.plot(DEPTHS, values, label=c, color=color, marker=marker, linestyle=line,
                    linewidth=width, markersize=9 if c == "C2" else 5, markerfacecolor="white"
                    if c == "C2" else color, clip_on=False)
        for d in DEPTHS:
            count2 = report["depth"][str(d)]["C2"]["strict_correct"]
            count3 = report["depth"][str(d)]["C3"]["strict_correct"]
            ax.text(d, 94, f"C2: {count2}/20 | C3: {count3}/20", ha="center", fontsize=10)
        ax.set_xlim(1.75, 4.25)
        ax.set_xticks(DEPTHS, ["2 hops", "3 hops", "4 hops"])
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 20))
        ax.set_ylabel("Strict accuracy (%)")
        ax.set_title("Random CoT vs semantic CoT by reasoning depth", loc="left", pad=16, weight="bold")
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
        ax.legend(loc="lower left", frameon=False, ncols=2)
        second.text(0.10, 0.06, "Frozen test: n=20 per depth, per condition | Both traces overlap at 100%\n"
                    "No observed retrieval accuracy gain over random CoT in this single run; ceiling applies.",
                    fontsize=9, color="#444444")
        second.subplots_adjust(left=0.10, right=0.97, bottom=0.18, top=0.87)
    return {"four_condition_accuracy": first, "c2_c3_accuracy_by_depth": second}


def save_figures(report, output=OUTPUT_PATH, *, before_finalize=None):
    output = Path(output).absolute()
    # Lexists behavior also rejects a dangling symlink; do not repair old output.
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"输出目录已存在，禁止覆盖：{output}")
    require(not any(p.is_symlink() for p in output.parents), "输出路径包含符号链接。")
    figures = make_figures(report)
    try:
        output.mkdir(parents=True, exist_ok=False)
        for name, figure in figures.items():
            for extension in ("png", "svg"):
                # Exclusive file handles prevent accidental overwrites.
                with (output / f"{name}.{extension}").open("xb") as stream:
                    figure.savefig(stream, format=extension, dpi=180, facecolor="white")
        with (output / "plot_statistics.json").open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        if before_finalize is not None:
            before_finalize()
        files = {p.name: file_hash(p) for p in sorted(output.iterdir())}
        manifest = {"plot_version": VERSION, "status": "complete", "run_id": RUN_ID,
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "source_sha256": report["source_sha256"], "file_sha256": files,
                    "script_sha256": file_hash(__file__), "matplotlib_version": version("matplotlib"),
                    "python_version": sys.version.split()[0], "axis_limits_percent": [0, 100],
                    "no_api_call": True, "no_experiment_changed": True,
                    "notes": "English labels avoid machine-dependent CJK fonts; no CI, jitter or causal claims."}
        with (output / "plot_manifest.json").open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        return manifest
    finally:
        from matplotlib import pyplot as plt
        for figure in figures.values():
            plt.close(figure)
    # A rendering failure leaves a new incomplete directory, never deletes user files.


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true", help="只读核查统计，不导入Matplotlib。")
    modes.add_argument("--write", action="store_true", help="核查后新建图片目录；禁止覆盖。")
    args = parser.parse_args(argv)
    if args.write and (OUTPUT_PATH.exists() or OUTPUT_PATH.is_symlink()):
        raise FileExistsError(f"输出目录已存在，禁止覆盖：{OUTPUT_PATH}")
    report = load_statistics()
    print(f"Day5来源指纹、240条回答重评分及统计核对：PASS（{report['protected_file_count']}个受保护文件）")
    print(json.dumps({"conditions": report["conditions"], "depth": report["depth"]},
                     ensure_ascii=False, indent=2))
    if not args.write:
        print("只读预览：未导入绘图库、未保存图片、未修改实验文件、未读取密钥、未调用API。")
        print("拟保存目录：", OUTPUT_PATH)
        return 0
    def check_before_finalize():
        require(load_statistics() == report, "保存前后来源或统计改变；停止，不修复。")
    saved = save_figures(report, before_finalize=check_before_finalize)
    print("图片保存：PASS；两张图各PNG/SVG，另附统计JSON及指纹清单。")
    print("保存目录：", OUTPUT_PATH)
    print("输出清单指纹：", digest(saved))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"停止：{error}", file=sys.stderr)
        raise SystemExit(1)
