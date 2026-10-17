"""假设价格与自动清理的临时账本；不读取密钥、数据集或调用API。"""

import json
import tempfile
import unittest
from pathlib import Path

from budget_guard import BudgetError, BudgetGuard


class BudgetGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "mock_budget.jsonl"

    def tearDown(self):
        self.temp.cleanup()

    def guard(self, *, create=True, **changes):
        settings = {
            "project_id": "offline_project",
            "cap_cny": "50",
            # 纯测试假设价格，不是DeepSeek官方价格。
            "input_rate_per_million": "10",
            "output_rate_per_million": "20",
            "is_simulated": True,
        }
        settings.update(changes)
        return BudgetGuard(self.path, create=create, **settings)

    def reserve(self, guard, attempt_id="runA:q1:C0:a1"):
        return guard.reserve(
            attempt_id, max_input_tokens=10000, max_output_tokens=10000,
        )

    def test_reserve_and_settle(self):
        guard = self.guard()
        view = self.reserve(guard)
        self.assertEqual(view["reserved_cny"], "0.300000")
        self.assertEqual(view["available_cny"], "49.700000")
        self.assertEqual(view["accounted_cost_cny"], "0.000000")
        view = guard.settle(
            "runA:q1:C0:a1", input_tokens=1000, output_tokens=1000,
        )
        self.assertEqual(view["accounted_cost_cny"], "0.030000")
        self.assertEqual(view["reserved_cny"], "0.000000")
        self.assertEqual(view["available_cny"], "49.970000")
        self.assertEqual(view["pending_attempt_ids"], [])

    def test_insufficient_budget_does_not_write(self):
        guard = self.guard(cap_cny="0.2")
        before = self.path.read_bytes()
        with self.assertRaises(BudgetError):
            self.reserve(guard)
        self.assertEqual(self.path.read_bytes(), before)

    def test_exact_cap_is_allowed_but_next_request_is_blocked(self):
        guard = self.guard(cap_cny="0.3")
        view = self.reserve(guard)
        self.assertEqual(view["available_cny"], "0.000000")
        guard.settle(
            "runA:q1:C0:a1", input_tokens=10000, output_tokens=10000,
        )
        with self.assertRaises(BudgetError):
            guard.reserve(
                "runA:q2:C0:a1", max_input_tokens=1, max_output_tokens=0,
            )

    def test_small_positive_cost_rounds_up(self):
        guard = self.guard(
            input_rate_per_million="0.000001",
            output_rate_per_million="0.000001",
        )
        view = guard.reserve(
            "tiny", max_input_tokens=1, max_output_tokens=0,
        )
        self.assertEqual(view["reserved_cny"], "0.000001")

    def test_pending_request_blocks_new_reservation(self):
        guard = self.guard()
        self.reserve(guard)
        before = self.path.read_bytes()
        with self.assertRaises(BudgetError):
            self.reserve(guard, "runA:q2:C0:a1")
        self.assertEqual(self.path.read_bytes(), before)

    def test_unknown_preserves_reservation_after_restart(self):
        guard = self.guard()
        self.reserve(guard)
        guard.mark_unknown("runA:q1:C0:a1")
        resumed = self.guard(create=False)
        view = resumed.summary()
        self.assertEqual(view["reserved_cny"], "0.300000")
        self.assertEqual(view["pending_attempt_ids"], ["runA:q1:C0:a1"])
        with self.assertRaises(BudgetError):
            self.reserve(resumed, "runB:q2:C0:a1")

    def test_restart_and_new_run_do_not_reset_spend(self):
        guard = self.guard()
        self.reserve(guard)
        guard.settle(
            "runA:q1:C0:a1", input_tokens=1000, output_tokens=1000,
        )
        resumed = self.guard(create=False)
        self.reserve(resumed, "runB:q1:C0:a1")
        view = resumed.settle(
            "runB:q1:C0:a1", input_tokens=1000, output_tokens=1000,
        )
        self.assertEqual(view["accounted_cost_cny"], "0.060000")

    def test_retry_uses_a_separate_reservation(self):
        guard = self.guard()
        self.reserve(guard)
        guard.settle(
            "runA:q1:C0:a1", input_tokens=1000, output_tokens=1000,
        )
        view = self.reserve(guard, "runA:q1:C0:a2")
        self.assertEqual(view["accounted_cost_cny"], "0.030000")
        self.assertEqual(view["reserved_cny"], "0.300000")
        self.assertEqual(view["available_cny"], "49.670000")

    def test_attempt_cannot_be_reused_or_settled_twice(self):
        guard = self.guard()
        self.reserve(guard)
        guard.settle(
            "runA:q1:C0:a1", input_tokens=1000, output_tokens=1000,
        )
        with self.assertRaises(BudgetError):
            self.reserve(guard)
        with self.assertRaises(BudgetError):
            guard.settle(
                "runA:q1:C0:a1", input_tokens=1000, output_tokens=1000,
            )

    def test_exceeded_token_bound_is_recorded_and_halts(self):
        guard = self.guard()
        self.reserve(guard)
        with self.assertRaises(BudgetError):
            guard.settle(
                "runA:q1:C0:a1", input_tokens=10001, output_tokens=1000,
            )
        view = self.guard(create=False).summary()
        self.assertTrue(view["halted"])
        self.assertEqual(view["accounted_cost_cny"], "0.120010")
        with self.assertRaises(BudgetError):
            self.reserve(guard, "runA:q2:C0:a1")

    def test_missing_usage_does_not_release_reservation(self):
        guard = self.guard()
        self.reserve(guard)
        before = self.path.read_bytes()
        with self.assertRaises(BudgetError):
            guard.settle(
                "runA:q1:C0:a1", input_tokens=None, output_tokens=100,
            )
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(guard.summary()["reserved_cny"], "0.300000")

    def test_invalid_money_is_rejected(self):
        for value in (50.0, True, "-1", "NaN", "Infinity", "0.0000001"):
            with self.subTest(value=value):
                with self.assertRaises(BudgetError):
                    self.guard(cap_cny=value)
        for value in ("0", "-1", 10.0):
            with self.subTest(rate=value):
                with self.assertRaises(BudgetError):
                    self.guard(input_rate_per_million=value)
        self.assertFalse(self.path.exists())

    def test_invalid_token_bounds_are_rejected(self):
        guard = self.guard()
        before = self.path.read_bytes()
        for value in (-1, True, 1.5, None):
            with self.subTest(value=value):
                with self.assertRaises(BudgetError):
                    guard.reserve(
                        "bad", max_input_tokens=value, max_output_tokens=100,
                    )
        self.assertEqual(self.path.read_bytes(), before)

    def test_context_changes_are_blocked(self):
        self.guard()
        for changes in (
            {"cap_cny": "60"},
            {"input_rate_per_million": "11"},
            {"project_id": "another_project"},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(BudgetError):
                    self.guard(create=False, **changes)

    def test_partial_tail_is_not_repaired(self):
        guard = self.guard()
        with self.path.open("a", encoding="utf-8") as file:
            file.write('{"unfinished":')
        before = self.path.read_bytes()
        with self.assertRaises(BudgetError):
            guard.summary()
        self.assertEqual(self.path.read_bytes(), before)

    def test_tampered_header_is_rejected(self):
        guard = self.guard()
        entry = json.loads(self.path.read_text())
        entry["data"]["cap_cny"] = "500.000000"
        # 故意修改临时测试账本，但不更新指纹。
        with self.path.open("w", encoding="utf-8") as file:
            file.write(json.dumps(entry) + "\n")
        before = self.path.read_bytes()
        with self.assertRaises(BudgetError):
            guard.summary()
        self.assertEqual(self.path.read_bytes(), before)

    def test_existing_ledger_is_not_overwritten(self):
        self.guard()
        before = self.path.read_bytes()
        with self.assertRaises(FileExistsError):
            self.guard()
        self.assertEqual(self.path.read_bytes(), before)

    def test_real_mode_is_blocked_before_creation(self):
        with self.assertRaises(BudgetError):
            self.guard(is_simulated=False)
        self.assertFalse(self.path.exists())

    def test_evidence_and_binding_survive_restart(self):
        guard = self.guard()
        binding = guard.inspect()["binding"]
        self.reserve(guard)
        guard.settle(
            "runA:q1:C0:a1", input_tokens=1000, output_tokens=1000,
            evidence_hash="a" * 64,
        )
        inspection = self.guard(create=False).inspect()
        self.assertEqual(inspection["binding"], binding)
        item = inspection["attempts"]["runA:q1:C0:a1"]
        self.assertEqual(item["evidence_hash"], "a" * 64)
        self.assertEqual((item["input_tokens"], item["output_tokens"]), (1000, 1000))
        item["status"] = "unknown"
        self.assertEqual(guard.inspect()["attempts"]["runA:q1:C0:a1"]["status"], "settled")

    def test_invalid_evidence_keeps_reservation_unchanged(self):
        guard = self.guard()
        self.reserve(guard)
        before = self.path.read_bytes()
        with self.assertRaises(BudgetError):
            guard.settle(
                "runA:q1:C0:a1", input_tokens=1000, output_tokens=1000,
                evidence_hash="not-a-hash",
            )
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(guard.summary()["reserved_cny"], "0.300000")


if __name__ == "__main__":
    unittest.main()
