"""模拟记录＋自动清理的临时日志；不读取真实数据或调用API。"""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from build_prompts import build_messages
from experiment_log import (
    ARTIFACT_PATHS,
    append_record,
    build_record,
    create_log,
    read_records,
    validate_record,
)


CONFIG = {
    "stage": "dev",
    "conditions": ["C0", "C1", "C2"],
    "data": {"allow_test": False},
    "api": {
        "base_url": "https://api.deepseek.com",
        "endpoint": "/chat/completions",
        "model": "deepseek-v4-pro",
        "thinking": {"type": "disabled"},
        "temperature": 0,
        "max_tokens": 1024,
        "stream": False,
        "response_format": {"type": "text"},
    },
    "runtime": {
        "timeout_seconds": 120,
        "concurrency": 1,
        "transient_error_retries": 2,
        "sdk_max_retries": 0,
    },
    "credentials": {"api_key_env": "DEEPSEEK_API_KEY"},
    "demo_selection": {"seed": 20261009},
    "provenance": {"provider_documented_version": "offline-fixture"},
}

TARGET = {
    "id": "offline_target",
    "world_fingerprint": "offline_world",
    "depth": 2,
    "premises": [
        "Wugs are vugs.",
        "Vugs are blue.",
        "Alex is a wug.",
    ],
    "query": "Alex is not blue.",
    "label": "False",
}

RESPONSE = {
    "raw_output": "Answer: False",
    "finish_reason": "stop",
    "model": "offline-model",
    "response_id": "offline-response",
    "latency_seconds": 0.01,
    "token_usage": {
        "prompt_tokens": 25,
        "completion_tokens": 3,
        "total_tokens": 28,
    },
}


def make_record(*, response=None, error_kind=None, attempt_no=1):
    if response is None and error_kind is None:
        response = copy.deepcopy(RESPONSE)

    return build_record(
        run_id="offline_test_run",
        config=copy.deepcopy(CONFIG),
        target=copy.deepcopy(TARGET),
        condition="C0",
        demos=[],
        messages=build_messages(TARGET, "C0"),
        # 明确是模拟指纹，不代表真实文件。
        artifact_hashes={name: "0" * 64 for name in ARTIFACT_PATHS},
        response=response,
        error_kind=error_kind,
        attempt_no=attempt_no,
        is_simulated=True,
    )


class ExperimentLogTests(unittest.TestCase):

    def test_standard_record(self):
        record = make_record()
        validate_record(record)
        self.assertTrue(record["is_simulated"])
        self.assertEqual(record["status"], "completed")
        self.assertTrue(record["correct"])
        self.assertEqual(record["token_usage"]["total_tokens"], 28)
        self.assertIsNone(record["cost_estimate_cny"])

    def test_wrong_answer_record(self):
        response = copy.deepcopy(RESPONSE)
        response["raw_output"] = "Answer: True"
        record = make_record(response=response)
        self.assertEqual(record["status"], "completed")
        self.assertFalse(record["correct"])

    def test_incomplete_record(self):
        response = copy.deepcopy(RESPONSE)
        response["finish_reason"] = "length"
        response["token_usage"] = None
        record = make_record(response=response)
        self.assertEqual(record["status"], "incomplete")
        self.assertFalse(record["correct"])
        self.assertIsNone(record["token_usage"])

    def test_error_record(self):
        record = make_record(error_kind="timeout")
        self.assertEqual(record["status"], "api_error")
        self.assertIsNone(record["raw_output"])
        self.assertIsNone(record["score"])
        self.assertIsNone(record["correct"])
        self.assertIsNone(record["token_usage"])

    def test_no_secret_extra_fields(self):
        response = copy.deepcopy(RESPONSE)
        response["api_key"] = "offline-secret-placeholder"
        response["headers"] = {
            "Authorization": "offline-secret-placeholder"
        }
        record = make_record(response=response)
        serialized = json.dumps(record)
        self.assertNotIn("offline-secret-placeholder", serialized)
        self.assertNotIn("Authorization", serialized)

    def test_hash_and_tampering(self):
        record = make_record()
        record["raw_output"] = "changed"
        with self.assertRaises(ValueError):
            validate_record(record)

    def test_roundtrip_append(self):
        with tempfile.TemporaryDirectory() as directory:
            path = create_log(Path(directory) / "offline.jsonl")
            first = make_record(error_kind="connection")
            second = make_record(attempt_no=2)

            append_record(path, first)
            append_record(path, second)

            self.assertEqual(read_records(path), [first, second])

            # 已收到回答后，不允许再请求一次来换答案。
            with self.assertRaises(ValueError):
                append_record(path, make_record(attempt_no=3))

    def test_existing_log_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = create_log(Path(directory) / "offline.jsonl")
            record = make_record()
            append_record(path, record)
            original = path.read_bytes()

            with self.assertRaises(FileExistsError):
                create_log(path)

            self.assertEqual(path.read_bytes(), original)

    def test_duplicate_attempt_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = create_log(Path(directory) / "offline.jsonl")
            record = make_record(error_kind="timeout")
            append_record(path, record)
            original = path.read_bytes()

            with self.assertRaises(ValueError):
                append_record(path, record)

            self.assertEqual(path.read_bytes(), original)

    def test_partial_tail_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "offline.jsonl"
            path.write_text('{"unfinished":', encoding="utf-8")
            original = path.read_bytes()

            with self.assertRaises(ValueError):
                append_record(path, make_record())

            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
