"""Read-only acceptance audit for the current PrOntoQA ModusPonens dataset.

No model API calls. Unsupported sentences fail explicitly. This is a unary
forward-chaining checker, not a general English or first-order logic solver.
Only aggregate counts and failure IDs/codes are emitted, never test prose.
A report is written only when --report is supplied, exclusively to a new file.
"""

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import platform
import random
import re
import sys
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NAMES = ("demo", "dev", "test", "unused")
LABELS = ("True", "False")
EXPECTED_COUNTS = {"demo": 120, "dev": 20, "test": 60, "unused": 400}
PER_LABEL = {"demo": {2: 20, 3: 20, 4: 20},
             "dev": {2: 4, 3: 3, 4: 3}, "test": {2: 10, 3: 10, 4: 10}}
FIELDS = {"id", "source_file", "source_example_id", "source_seed", "depth",
          "premises", "query", "label", "gold_cot", "world_fingerprint",
          "prompt_fingerprint"}


def sha256(blob):
    return hashlib.sha256(blob).hexdigest()


def normalize(text):
    return " ".join(text.split())


def digest(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":")).encode("utf-8"))


def schema_errors(record):
    if not isinstance(record, dict) or set(record) != FIELDS:
        return ["schema_fields"]
    errors = []
    for key in ("id", "source_file", "source_example_id", "query"):
        if not isinstance(record[key], str) or not record[key].strip():
            errors.append(f"schema_{key}")
    if type(record["source_seed"]) is not int:
        errors.append("schema_source_seed")
    if type(record["depth"]) is not int or record["depth"] not in (2, 3, 4):
        errors.append("schema_depth")
    if record["label"] not in LABELS:
        errors.append("schema_label")
    for key in ("premises", "gold_cot"):
        values = record[key]
        if not (isinstance(values, list) and values
                and all(isinstance(s, str) and s.strip() for s in values)):
            errors.append(f"schema_{key}")
    if errors:
        return errors
    if Path(record["source_file"]).name != record["source_file"]:
        errors.append("unsafe_source_filename")
    premises = [normalize(s) for s in record["premises"]]
    query = normalize(record["query"])
    if premises != record["premises"] or query != record["query"]:
        errors.append("whitespace_not_normalized")
    if record["world_fingerprint"] != digest(sorted(set(premises))):
        errors.append("world_fingerprint")
    if record["prompt_fingerprint"] != digest({"premises": premises, "query": query}):
        errors.append("prompt_fingerprint")
    expected_id = (f"prontoqa_v1_seed{record['source_seed']}_"
                   f"h{record['depth']}_{record['source_example_id']}")
    if record["id"] != expected_id:
        errors.append("provenance_id")
    if record["source_file"] != (
        f"{record['depth']}hop_random_seed{record['source_seed']}.json"
    ):
        errors.append("provenance_filename")
    return errors


def atom(text):
    text = text.strip().lower()
    negative = text.startswith("not ")
    if negative:
        text = text[4:]
    text = re.sub(r"^(?:a|an) ", "", text)
    if not re.fullmatch(r"[a-z]+", text):
        raise ValueError("unsupported_atom")
    if text.endswith("uses"):
        text = text[:-2]
    return text, negative


def parse_sentence(sentence):
    sentence = normalize(sentence)
    match = re.fullmatch(r"(?:Each|Every) ([a-z]+) is (.+)\.", sentence)
    if match:
        return "rule", (atom(match[1]), atom(match[2]))
    match = re.fullmatch(r"([A-Z][a-z]+) are (.+)\.", sentence)
    if match:
        return "rule", (atom(match[1]), atom(match[2]))
    match = re.fullmatch(r"([A-Z][a-z]+) is (.+)\.", sentence)
    if match:
        return "fact", (match[1], atom(match[2]))
    raise ValueError("unsupported_sentence")


def chain_errors(record, person, goal):
    chain = record["gold_cot"]
    premises = {normalize(s) for s in record["premises"]}
    if len(chain) != 2 * record["depth"] + 1:
        return ["gold_chain_shape"]
    # Parse every sentence even if another chain check would fail early.
    parsed = [parse_sentence(s) for s in chain]
    kind, current = parsed[0]
    if kind != "fact" or normalize(chain[0]) not in premises:
        return ["gold_initial_fact"]
    for index in range(1, len(chain), 2):
        kind, rule = parsed[index]
        if kind != "rule" or normalize(chain[index]) not in premises:
            return ["gold_rule_not_in_premises"]
        antecedent, consequent = rule
        if current[1] != antecedent:
            return ["gold_rule_condition"]
        next_kind, next_fact = parsed[index + 1]
        if next_kind != "fact" or next_fact != (current[0], consequent):
            return ["gold_invalid_conclusion"]
        current = next_fact
    if current != (person, goal):
        return ["gold_final_goal"]
    return []


def logic_result(record):
    try:
        parsed = [parse_sentence(s) for s in record["premises"]]
        kind, query_info = parse_sentence(record["query"])
        if kind != "fact":
            raise ValueError("query_not_fact")
        person, target = query_info
        opposite = (target[0], not target[1])
        graph = defaultdict(list)
        distance = {}
        for kind, content in parsed:
            if kind == "rule":
                antecedent, consequent = content
                graph[antecedent].append(consequent)
            else:
                name, fact = content
                if name == person:
                    distance[fact] = 0
        queue = deque(distance)
        while queue:
            current = queue.popleft()
            for result in graph[current]:
                if result not in distance:
                    distance[result] = distance[current] + 1
                    queue.append(result)
        positive, negative = target in distance, opposite in distance
        actual = ("Both" if positive and negative else "True" if positive
                  else "False" if negative else "Unknown")
        errors = []
        if actual == "Unknown":
            errors.append("unknown_target")
        if actual == "Both":
            errors.append("both_target_and_negation")
        if actual != record["label"]:
            errors.append("label_mismatch")
        if any((name, not neg) in distance for name, neg in distance):
            errors.append("reachable_contradiction")
        goal = target if record["label"] == "True" else opposite
        shortest = distance.get(goal)
        if shortest != record["depth"]:
            errors.append("shortest_depth_mismatch")
        gold_errors = chain_errors(record, person, goal)
        errors.extend(gold_errors)
        return {"actual_label": actual, "shortest_depth": shortest,
                "gold_chain_passed": not gold_errors, "errors": errors}
    except ValueError as error:
        return {"actual_label": "Unparsed", "shortest_depth": None,
                "gold_chain_passed": False, "errors": [str(error)]}


class Audit:
    def __init__(self):
        self.failures = []
        self.hashes = {}

    def check(self, condition, code, dataset="", record_id=None):
        if not condition:
            failure = {"check": code, "dataset": dataset}
            if record_id is not None:
                failure["id"] = record_id
            self.failures.append(failure)
        return bool(condition)

    def read_bytes(self, path, dataset):
        try:
            blob = path.read_bytes()
        except OSError:
            self.check(False, "file_unreadable", dataset)
            return None
        self.hashes[dataset] = sha256(blob)
        return blob

    def read_jsonl(self, path, dataset):
        blob = self.read_bytes(path, dataset)
        if blob is None:
            return []
        try:
            lines = blob.decode("utf-8").splitlines()
        except UnicodeDecodeError:
            self.check(False, "invalid_utf8", dataset)
            return []
        records = []
        for number, line in enumerate(lines, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                self.check(False, "invalid_jsonl", dataset, f"line:{number}")
                continue
            errors = schema_errors(record)
            record_id = record.get("id") if isinstance(record, dict) else None
            for code in errors:
                self.check(False, code, dataset, record_id or f"line:{number}")
            if not errors:
                records.append(record)
        return records


def review_errors(review, dev, dev_hash):
    """Check the saved assisted-review trace; never relabel it as human-only."""
    errors = []
    try:
        rng = random.Random(20261008)
        selected = []
        for depth in (2, 3, 4):
            for label in LABELS:
                pool = sorted((r for r in dev if r["depth"] == depth
                               and r["label"] == label), key=lambda r: r["id"])
                rng.shuffle(pool)
                selected.extend(pool[:2])
        rng.shuffle(selected)
        rows = review["reviews"]
        expected = {"status": "assistant_assisted_review_complete",
                    "review_seed": 20261008, "source_split": "dev",
                    "dev_sha256": dev_hash, "checked": 12, "passed": 12,
                    "user_blind_answers": 1, "assistant_nonblind_checks": 11,
                    "all_twelve_human_blind_review_completed": False,
                    "test_content_inspected_by_review": False}
        if any(review.get(k) != v for k, v in expected.items()):
            errors.append("review_metadata")
        if len(rows) != 12 or len(selected) != 12:
            return errors + ["review_count"]
        if [r["id"] for r in rows] != [r["id"] for r in selected]:
            return errors + ["review_selection"]
        for number, (entry, source) in enumerate(zip(rows, selected), 1):
            valid = entry["review_id"] == f"Q{number:02d}"
            for field in ("source_file", "source_example_id", "depth", "label"):
                valid &= entry[field] == source[field]
            valid &= entry["predicted_label"] == source["label"]
            valid &= entry["result"] == "PASS" and not logic_result(source)["errors"]
            valid &= (entry["reference_chain_matches"] is True
                      and entry["shortest_depth_matches"] is True
                      and entry["shortcut_found"] is False
                      and entry["reachable_contradiction_found"] is False)
            blind = number == 1
            valid &= entry["label_hidden_before_answer"] is blind
            valid &= entry["gold_hidden_before_answer"] is blind
            valid &= entry["reviewer"] == (
                "user_blind_answer_plus_assistant_check" if blind else "assistant_nonblind")
            valid &= entry["conclusion_relation"] == (
                "matches_query" if source["label"] == "True" else "negates_query")
            path = [atom(s) for s in entry["path"]]
            indices = entry["applied_premise_indices"]
            valid &= len(path) == source["depth"] + 1
            valid &= len(indices) == source["depth"]
            valid &= all(type(i) is int and 1 <= i <= len(source["premises"])
                         for i in indices)
            valid &= parse_sentence(source["gold_cot"][0]) == (
                "fact", (entry["person"], path[0]))
            valid &= [parse_sentence(source["premises"][i - 1]) for i in indices] == [
                ("rule", (path[i], path[i + 1])) for i in range(len(path) - 1)]
            valid &= parse_sentence(source["gold_cot"][-1]) == (
                "fact", (entry["person"], path[-1]))
            if not valid:
                errors.append(f"review_trace_Q{number:02d}")
    except (KeyError, TypeError, ValueError, IndexError, AttributeError):
        errors.append("invalid_review_record")
    return errors


def audit_freeze(audit, args, dev):
    """A new hash-pinned record supersedes, but never rewrites, creation metadata."""
    freeze_path = args.split_dir / "freeze_record.json"
    if not freeze_path.exists():
        return {"test_frozen": False, "freeze_integrity_verified": False,
                "manual_acceptance": "not_frozen"}
    failure_start = len(audit.failures)
    try:
        blob = audit.read_bytes(freeze_path, "freeze_record")
        record = json.loads(blob)
        audit.check(record["freeze_schema_version"] == 1
                    and record["test_frozen"] is True
                    and record["freeze_method"] == "hash_pinned_snapshot"
                    and record["split_seed"] == args.split_seed
                    and record["generation_seed"] == 20261006
                    and record["review_seed"] == 20261008
                    and record["source_commit"] == "066f73fc70bfaa92ed2476fd3c41b4951ad6e63a",
                    "freeze_metadata")
        for name in NAMES:
            audit.check(record["dataset_files_sha256"][f"{name}.jsonl"]
                        == audit.hashes.get(name), "frozen_dataset_hash", name)
        for field, dataset in (("split_report_sha256", "split_report"),
                               ("candidate_input_sha256", "candidates"),
                               ("inspected_input_sha256", "inspected")):
            audit.check(record[field] == audit.hashes.get(dataset), "frozen_input_hash", dataset)
        artifacts = {}
        expected_paths = {
            "review_record": "data/validation/dev_review_seed20261008.json",
            "manual_notes": "data/manual_validation.md",
            "pre_freeze_validation": "data/validation/seed20261007_pre_freeze_validation.json",
        }
        for field, relative in expected_paths.items():
            audit.check(record[field]["path_relative"] == relative, "frozen_artifact_path", field)
            data = audit.read_bytes(ROOT / relative, field)
            audit.check(record[field]["sha256"] == audit.hashes.get(field),
                        "frozen_artifact_hash", field)
            if field != "manual_notes":
                artifacts[field] = json.loads(data)
        for error in review_errors(artifacts["review_record"], dev, audit.hashes.get("dev")):
            audit.check(False, error, "review_record")
        audit.check(record["review_policy"] == "user_authorized_assistant_assisted_review"
                    and record["user_blind_answers"] == 1
                    and record["assistant_nonblind_checks"] == 11
                    and record["all_twelve_human_blind_review_completed"] is False,
                    "freeze_review_policy")
        before = artifacts["pre_freeze_validation"]
        audit.check(before["status"] == "PASS" and before["failure_count"] == 0
                    and before["failures"] == [] and before["test_frozen"] is False
                    and before["split_seed"] == args.split_seed
                    and before["byte_identical_split_rebuild"] is True
                    and before["validator_sha256"] == sha256(Path(__file__).read_bytes()),
                    "pre_freeze_acceptance")
        audit.check(before["counts"] == {"candidates": 600, "inspected": 24,
                    **EXPECTED_COUNTS}, "pre_freeze_counts")
        for name in ("demo", "dev", "test"):
            audit.check(before["logic"][name]["checked"] == EXPECTED_COUNTS[name]
                        and before["logic"][name]["passed"] == EXPECTED_COUNTS[name]
                        and before["logic"][name]["gold_chains_passed"] == EXPECTED_COUNTS[name],
                        "pre_freeze_logic", name)
        for name in (*NAMES, "candidates", "inspected", "split_report"):
            audit.check(before["file_sha256"][name] == audit.hashes.get(name),
                        "pre_freeze_basis_hash", name)
    except (KeyError, TypeError, ValueError, OSError, UnicodeDecodeError, AttributeError):
        audit.check(False, "invalid_freeze_record")
    verified = len(audit.failures) == failure_start
    return {"test_frozen": verified, "freeze_integrity_verified": verified,
            "manual_acceptance": "assistant_assisted_review_complete" if verified
            else "invalid_freeze_record"}


def run_audit(args):
    audit = Audit()
    candidates = audit.read_jsonl(args.candidates, "candidates")
    inspected = audit.read_jsonl(args.inspected, "inspected")
    groups = {name: audit.read_jsonl(args.split_dir / f"{name}.jsonl", name)
              for name in NAMES}
    audit.check(len(candidates) == 600, "candidate_count", "candidates")
    audit.check(len(inspected) == 24, "inspected_count", "inspected")
    for field in ("id", "world_fingerprint", "prompt_fingerprint"):
        audit.check(len({r[field] for r in candidates}) == len(candidates),
                    f"candidate_unique_{field}", "candidates")
    audit.check(all(r["source_seed"] == 20261006 for r in candidates),
                "candidate_generation_seed", "candidates")

    # Check conversion against raw data without using the converter function.
    raw_cache = {}
    source_pairs = set()
    for filename in sorted({r["source_file"] for r in candidates}):
        blob = audit.read_bytes(args.raw_dir / filename, f"raw/{filename}")
        if blob is not None:
            try:
                value = json.loads(blob)
                if not isinstance(value, dict):
                    raise ValueError
                raw_cache[filename] = value
            except (ValueError, UnicodeDecodeError):
                audit.check(False, "invalid_raw_json", f"raw/{filename}")
    for record in candidates:
        key = (record["source_file"], record["source_example_id"])
        source_pairs.add(key)
        try:
            original = raw_cache[key[0]][key[1]]["test_example"]
            expected_premises = [normalize(s) + "."
                                 for s in original["question"].split(".") if s.strip()]
            prefix = "True or false: "
            valid = (original["query"].startswith(prefix)
                     and record["premises"] == expected_premises
                     and record["query"] == normalize(original["query"][len(prefix):])
                     and record["label"] == original["answer"]
                     and record["gold_cot"] == original["chain_of_thought"])
        except (KeyError, TypeError, AttributeError):
            valid = False
        audit.check(valid, "raw_content_preservation", "candidates", record["id"])
    expected_pairs = {(filename, key) for filename, value in raw_cache.items()
                      for key in value}
    audit.check(source_pairs == expected_pairs and len(source_pairs) == 600,
                "raw_records_accounted_for", "candidates")

    by_id = {r["id"]: r for r in candidates}
    ids, worlds, prompts = {}, {}, {}
    for name, records in groups.items():
        audit.check(len(records) == EXPECTED_COUNTS[name], "split_count", name)
        ids[name] = {r["id"] for r in records}
        worlds[name] = {r["world_fingerprint"] for r in records}
        prompts[name] = {r["prompt_fingerprint"] for r in records}
        audit.check(len(ids[name]) == len(records), "split_unique_ids", name)
        audit.check(len(worlds[name]) == len(records), "split_unique_worlds", name)
        for record in records:
            audit.check(by_id.get(record["id"]) == record,
                        "split_content_preservation", name, record["id"])
        if name in PER_LABEL:
            actual = Counter((r["depth"], r["label"]) for r in records)
            expected = Counter({(d, label): n for d, n in PER_LABEL[name].items()
                                for label in LABELS})
            audit.check(actual == expected, "split_quota", name)
    for first, second in itertools.combinations(NAMES, 2):
        for field, values in (("ids", ids), ("worlds", worlds), ("prompts", prompts)):
            audit.check(not values[first] & values[second],
                        f"cross_split_{field}", f"{first}/{second}")
    audit.check(set().union(*ids.values()) == set(by_id)
                and sum(len(r) for r in groups.values()) == len(candidates),
                "all_candidates_accounted_for")
    inspected_worlds = {r["world_fingerprint"] for r in inspected}
    audit.check(not worlds["test"] & inspected_worlds, "test_excludes_inspected_worlds", "test")

    report_blob = audit.read_bytes(args.split_dir / "split_report.json", "split_report")
    split_report = {}
    try:
        if report_blob is not None:
            split_report = json.loads(report_blob)
        if not isinstance(split_report, dict):
            raise ValueError
    except (ValueError, UnicodeDecodeError):
        audit.check(False, "invalid_split_report")
        split_report = {}
    try:
        audit.check(split_report["split_seed"] == args.split_seed, "split_seed")
        expected_quotas = {
            str(depth): {name: PER_LABEL[name][depth] for name in PER_LABEL}
            for depth in (2, 3, 4)
        }
        audit.check(split_report["quotas_per_label"] == expected_quotas,
                    "recorded_quotas")
        audit.check(split_report["stratum_order"] == [
            [depth, label] for depth in (2, 3, 4) for label in LABELS
        ], "recorded_stratum_order")
        audit.check(split_report["selection_order"] == ["test", "dev", "demo", "unused"],
                    "recorded_selection_order")
        audit.check(split_report["final_file_shuffle_order"] == list(NAMES),
                    "recorded_file_shuffle_order")
        audit.check(type(split_report["test_frozen"]) is bool,
                    "recorded_freeze_flag_type")
        audit.check(split_report["checks"] == {
            "quota_match": True, "ids_disjoint": True, "worlds_disjoint": True,
            "all_candidates_accounted_for": True, "test_excludes_inspected_worlds": True,
        }, "recorded_check_flags")
        for key, dataset in (("candidate_input", "candidates"), ("inspected_input", "inspected")):
            audit.check(split_report[key]["sha256"] == audit.hashes.get(dataset),
                        "recorded_input_hash", dataset)
            expected_count = len(candidates) if dataset == "candidates" else len(inspected)
            audit.check(split_report[key]["count"] == expected_count,
                        "recorded_input_count", dataset)
        audit.check(split_report["inspected_input"]["unique_worlds"] == len(inspected_worlds),
                    "recorded_inspected_worlds")
        audit.check(split_report["candidate_records_excluded_from_test"] == sum(
            r["world_fingerprint"] in inspected_worlds for r in candidates),
            "recorded_test_exclusions")
        audit.check(split_report["generation_seeds"] == [20261006], "recorded_generation_seeds")
        for name in NAMES:
            audit.check(split_report["file_sha256"][f"{name}.jsonl"] == audit.hashes.get(name),
                        "recorded_split_hash", name)
            actual_counts = {str(d): {label: sum(r["depth"] == d and r["label"] == label
                                               for r in groups[name]) for label in LABELS}
                             for d in (2, 3, 4)}
            audit.check(split_report["counts"][name] == {
                "total": len(groups[name]), "by_depth_and_label": actual_counts},
                "recorded_split_counts", name)
        splitter_path = ROOT / "scripts/split_candidates.py"
        audit.check(split_report["script_sha256"] == sha256(splitter_path.read_bytes()),
                    "recorded_splitter_hash")
    except (KeyError, TypeError, OSError):
        audit.check(False, "split_report_missing_or_invalid_metadata")

    # Only the reproduction step reuses the splitter; all checks above are independent.
    reproducible = False
    try:
        import split_candidates
        rebuilt, _ = split_candidates.build_outputs(
            args.candidates, args.inspected, args.split_seed, 600)
        reproducible = all(sha256(rebuilt[f"{name}.jsonl"]) == audit.hashes.get(name)
                           for name in NAMES)
        audit.check(reproducible, "byte_identical_split_rebuild")
    except (ValueError, KeyError, TypeError, OSError, UnicodeDecodeError, ImportError):
        audit.check(False, "split_rebuild_failed")

    logic = {}
    for name in ("demo", "dev", "test"):
        actual_labels = Counter()
        passed = gold_passed = 0
        for record in groups[name]:
            result = logic_result(record)
            actual_labels[result["actual_label"]] += 1
            passed += not result["errors"]
            gold_passed += result["gold_chain_passed"]
            for code in result["errors"]:
                audit.check(False, code, name, record["id"])
        logic[name] = {"checked": len(groups[name]), "passed": passed,
                       "gold_chains_passed": gold_passed,
                       "actual_labels": dict(actual_labels)}
    audit.check(sum(v["checked"] for v in logic.values()) == 200, "formal_logic_count")
    freeze = audit_freeze(audit, args, groups["dev"])
    packages = {}
    for name in ("numpy", "scipy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "validation_schema_version": 1,
        "status": "PASS" if not audit.failures else "FAIL",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "600 candidate structure/conversion; four splits; 200 formal logic/proofs",
        "logic_system": "unary explicit-positive/negative ModusPonens forward chaining",
        "split_seed": args.split_seed,
        **freeze,
        "split_report_test_frozen_at_creation": split_report.get("test_frozen"),
        "split_report_manual_acceptance_at_creation": split_report.get("manual_acceptance"),
        "environment": {"python_version": platform.python_version(),
                        "python_executable": sys.executable, "platform": platform.platform(),
                        "packages": packages},
        "counts": {"candidates": len(candidates), "inspected": len(inspected),
                   **{name: len(group) for name, group in groups.items()}},
        "logic": logic,
        "byte_identical_split_rebuild": reproducible,
        "file_sha256": audit.hashes,
        "validator_sha256": sha256(Path(__file__).read_bytes()),
        "failure_count": len(audit.failures),
        "failures": audit.failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path,
                        default=ROOT / "data/processed/candidates_seed20261006.jsonl")
    parser.add_argument("--inspected", type=Path,
                        default=ROOT / "data/processed/step2_candidates.jsonl")
    parser.add_argument("--raw-dir", type=Path,
                        default=ROOT / "data/raw/step4_candidates_seed20261006")
    parser.add_argument("--split-dir", type=Path,
                        default=ROOT / "data/splits/seed20261007")
    parser.add_argument("--split-seed", type=int, default=20261007)
    parser.add_argument("--report", type=Path,
                        help="Optional new report file; existing files are never overwritten")
    args = parser.parse_args()
    if args.report:
        if args.report.exists():
            parser.error("报告路径已存在，不允许覆盖")
        # Prevent accidental creation of a report in an input or source directory.
        safe_root = (ROOT / "data/validation").resolve()
        if not args.report.resolve().is_relative_to(safe_root):
            parser.error("报告只能写入项目的data/validation目录")
    report = run_audit(args)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    print(f"验收结果：{report['status']}")
    print(f"候选题：{report['counts']['candidates']}；分组记录："
          f"{sum(report['counts'][name] for name in NAMES)}")
    for name, result in report["logic"].items():
        print(f"{name}：逻辑通过{result['passed']}/{result['checked']}，"
              f"参考链通过{result['gold_chains_passed']}/{result['checked']}")
    print(f"逐字节重建一致：{report['byte_identical_split_rebuild']}")
    print(f"失败项：{report['failure_count']}")
    print(f"test_frozen：{report['test_frozen']}（本脚本不会冻结测试集）")
    if args.report:
        print(f"报告：{args.report.resolve()}")
    else:
        print("只读运行，未保存验收报告")
    for failure in report["failures"][:10]:
        print(json.dumps(failure, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
