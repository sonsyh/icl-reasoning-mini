"""Verify the reduced public showcase; synthetic tests only, no network/API."""
import hashlib
import json
from pathlib import Path
import socket
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = (
    "test_answer_parser.py", "test_deepseek_client.py", "test_experiment_log.py",
    "test_retry_resume.py", "test_budget_guard.py", "test_dev_runner.py",
    "test_dev_runner_budget.py",
)


def verify_manifest():
    manifest = json.loads((ROOT / "PUBLIC_MANIFEST.json").read_text())
    mapping = manifest["files_sha256"]
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*")
              if p.is_file() and not any(x in (".git", "__pycache__") for x in p.relative_to(ROOT).parts)
              and p.name != "PUBLIC_MANIFEST.json"}
    if actual != set(mapping):
        raise ValueError("Public file set changed")
    for name, digest in mapping.items():
        path = ROOT / name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Public file fingerprint changed: " + name)


def verify_summary():
    report = json.loads((ROOT / "results/summary.json").read_text())
    if report["question_count"] != 60 or report["response_count"] != 240:
        raise ValueError("Unexpected aggregate scope")
    for condition, expected in zip(("C0", "C1", "C2", "C3"), (48, 53, 60, 60)):
        row = report["conditions"][condition]
        if row["n"] != 60 or row["strict_correct"] != expected or row["lenient_correct"] != expected:
            raise ValueError("Unexpected aggregate count")
        grouped = sum(report["depth"][str(d)][condition]["strict_correct"] for d in (2, 3, 4))
        if grouped != expected:
            raise ValueError("Depth aggregates disagree")


def main():
    sys.dont_write_bytecode = True
    verify_manifest()
    verify_summary()
    def deny(*args, **kwargs):
        raise RuntimeError("Network is disabled in public checks")
    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = deny
    sys.path.insert(0, str(ROOT / "scripts"))
    suite = unittest.TestSuite()
    for pattern in PATTERNS:
        suite.addTests(unittest.defaultTestLoader.discover(str(ROOT / "scripts"), pattern=pattern))
    # Review/freeze integration classes require excluded private artifacts.
    suite.addTests(unittest.defaultTestLoader.loadTestsFromNames((
        "test_validate_dataset.LogicTests", "test_validate_dataset.SchemaTests")))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    print(json.dumps({"tests_run": result.testsRun, "successful": result.wasSuccessful(),
                      "failures": len(result.failures), "errors": len(result.errors),
                      "skipped": len(result.skipped),
                      "manifest_and_aggregate_checks": "PASS", "real_api_requests": 0,
                      "scope": "selected synthetic public tests, not full experiment reproduction"}))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
