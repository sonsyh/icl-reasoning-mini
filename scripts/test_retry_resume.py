"""只使用已有模拟日志夹具；不读取数据集、不发送请求。"""

import copy
import unittest

from build_prompts import hash_messages
from experiment_log import hash_json
from retry_resume import plan_next_attempt
from test_experiment_log import RESPONSE, make_record


def reseal(record):
    """仅用于构造内部指纹一致的模拟测试记录。"""
    record["prompt_hash"] = hash_messages(record["prompt"])
    record["logical_request_id"] = hash_json([
        record["run_id"],
        record["question_id"],
        record["condition"],
        record["config_hash"],
        record["prompt_hash"],
        hash_json(record["artifact_hashes"]),
    ])
    record.pop("record_hash", None)
    record["record_hash"] = hash_json(record)
    return record


class RetryResumeTests(unittest.TestCase):

    def setUp(self):
        self.expected = make_record()

    def plan(self, records, pending=False):
        return plan_next_attempt(
            records,
            self.expected,
            pending_attempt=pending,
        )

    def test_new_request(self):
        result = self.plan([])
        self.assertEqual(result["action"], "start")
        self.assertEqual(result["next_attempt_no"], 1)

    def test_transient_errors(self):
        for kind in ("timeout", "connection", "rate_limit", "server_error"):
            with self.subTest(kind=kind):
                result = self.plan([make_record(error_kind=kind)])
                self.assertEqual(result["action"], "retry")
                self.assertEqual(result["next_attempt_no"], 2)
                self.assertEqual(result["wait_seconds"], 2)

    def test_third_attempt(self):
        records = [
            make_record(error_kind="timeout"),
            make_record(error_kind="connection", attempt_no=2),
        ]
        result = self.plan(records)
        self.assertEqual(result["action"], "retry")
        self.assertEqual(result["next_attempt_no"], 3)
        self.assertEqual(result["wait_seconds"], 4)

    def test_retry_limit(self):
        records = [
            make_record(error_kind="timeout", attempt_no=number)
            for number in (1, 2, 3)
        ]
        result = self.plan(records)
        self.assertEqual(result["action"], "stop")
        self.assertIsNone(result["next_attempt_no"])

    def test_non_retryable_errors(self):
        for kind in (
            "authentication", "permission", "invalid_request", "unknown_error",
        ):
            with self.subTest(kind=kind):
                result = self.plan([make_record(error_kind=kind)])
                self.assertEqual(result["action"], "block")

    def test_wrong_answer_skipped(self):
        response = copy.deepcopy(RESPONSE)
        response["raw_output"] = "Answer: True"
        record = make_record(response=response)
        self.assertFalse(record["correct"])
        self.assertEqual(self.plan([record])["action"], "skip")

    def test_format_failure_skipped(self):
        response = copy.deepcopy(RESPONSE)
        response["raw_output"] = "I cannot give a clear answer."
        record = make_record(response=response)
        self.assertFalse(record["correct"])
        self.assertEqual(self.plan([record])["action"], "skip")

    def test_truncated_response_skipped(self):
        response = copy.deepcopy(RESPONSE)
        response["finish_reason"] = "length"
        record = make_record(response=response)
        self.assertEqual(self.plan([record])["action"], "skip")

    def test_pending_outcome_blocked(self):
        result = self.plan([], pending=True)
        self.assertEqual(result["action"], "block")
        self.assertEqual(result["reason"], "unresolved_attempt")

    def test_config_mismatch(self):
        self.expected["config_hash"] = "1" * 64
        reseal(self.expected)
        with self.assertRaises(ValueError):
            self.plan([make_record()])

    def test_prompt_mismatch(self):
        self.expected["prompt"][-1]["content"] += "\nChanged input."
        reseal(self.expected)
        with self.assertRaises(ValueError):
            self.plan([make_record()])

    def test_missing_attempt_rejected(self):
        record = make_record(error_kind="timeout", attempt_no=2)
        with self.assertRaises(ValueError):
            self.plan([record])

    def test_extra_attempt_after_response_rejected(self):
        records = [
            make_record(),
            make_record(error_kind="timeout", attempt_no=2),
        ]
        with self.assertRaises(ValueError):
            self.plan(records)

    def test_tampered_record_rejected(self):
        record = make_record()
        record["raw_output"] = "changed without updating hash"
        with self.assertRaises(ValueError):
            self.plan([record])

    def test_other_question_does_not_cause_skip(self):
        record = make_record()
        record["question_id"] = "offline_other_question"
        reseal(record)
        self.assertEqual(self.plan([record])["action"], "start")

    def test_inputs_not_modified(self):
        records = [make_record(error_kind="timeout")]
        original_records = copy.deepcopy(records)
        original_expected = copy.deepcopy(self.expected)

        self.plan(records)

        self.assertEqual(records, original_records)
        self.assertEqual(self.expected, original_expected)

    def test_pending_flag_must_be_boolean(self):
        for value in (None, 0, "false"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.plan([], pending=value)

    def test_simulated_and_real_records_cannot_mix(self):
        record = make_record()
        record["is_simulated"] = False
        reseal(record)
        with self.assertRaises(ValueError):
            self.plan([record])

    def test_dataset_snapshot_mismatch(self):
        record = make_record()
        record["artifact_hashes"]["data/splits/seed20261007/dev.jsonl"] = "1" * 64
        reseal(record)
        with self.assertRaises(ValueError):
            self.plan([record])

    def test_sdk_retry_must_be_disabled(self):
        self.expected["runtime"]["sdk_max_retries"] = 1
        reseal(self.expected)
        with self.assertRaises(ValueError):
            self.plan([])

    def test_score_summary_mismatch_rejected(self):
        record = make_record()
        record["correct"] = False
        reseal(record)
        with self.assertRaises(ValueError):
            self.plan([record])

    def test_retry_after_non_transient_error_rejected(self):
        records = [
            make_record(error_kind="authentication"),
            make_record(error_kind="timeout", attempt_no=2),
        ]
        with self.assertRaises(ValueError):
            self.plan(records)


if __name__ == "__main__":
    unittest.main()
