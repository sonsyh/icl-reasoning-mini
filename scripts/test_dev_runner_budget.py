"""预算与运行器集成验收：仅模拟用量、价格与临时文件。"""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fcntl

from budget_guard import BudgetError, BudgetGuard
from dev_runner import DevRunner, SimulatedAPIError, SimulatedResponder
from experiment_log import ARTIFACT_PATHS, read_records
from test_dev_runner import InjectedCrash
from test_experiment_log import CONFIG, RESPONSE, TARGET


class BudgetedRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / "runA"
        self.ledger_path = self.root / "project_budget.jsonl"
        self.config = copy.deepcopy(CONFIG)
        self.config["budget"] = {"currency": "CNY", "project_total_cap": "50"}
        self.artifacts = {name: "0" * 64 for name in ARTIFACT_PATHS}

    def tearDown(self):
        self.temp.cleanup()

    def guard(self, *, cap="50", path=None, create=True):
        return BudgetGuard(
            self.ledger_path if path is None else path,
            project_id="offline_shared_project",
            cap_cny=cap,
            # 这两个单价只是假设价格，不是官方计费参数。
            input_rate_per_million="10",
            output_rate_per_million="20",
            create=create,
        )

    def runner(self, guard, *, create=True, path=None, run_id="runA", bound=20000):
        return DevRunner(
            self.path if path is None else path,
            run_id=run_id,
            config=self.config,
            artifact_hashes=self.artifacts,
            budget_guard=guard,
            max_input_tokens=bound,
            create=create,
        )

    def run_target(self, runner, responder, *, checkpoint=None):
        return runner.run_one(
            copy.deepcopy(TARGET), "C0", [],
            responder=responder, checkpoint=checkpoint,
        )

    @staticmethod
    def crash_at(name):
        def checkpoint(stage):
            if stage == name:
                raise InjectedCrash(stage)
        return checkpoint

    def test_budget_guard_is_mandatory(self):
        with self.assertRaises(ValueError):
            DevRunner(
                self.path, run_id="runA", config=self.config,
                artifact_hashes=self.artifacts, create=True,
            )
        self.assertFalse(self.path.exists())

    def test_reservation_exists_before_simulated_call(self):
        guard = self.guard()
        runner = self.runner(guard)
        responder = SimulatedResponder([RESPONSE])
        checked = []

        def checkpoint(stage):
            if stage == "after_start":
                view = guard.summary()
                self.assertEqual(responder.calls, 0)
                self.assertEqual(view["reserved_cny"], "0.220480")
                self.assertEqual(len(view["pending_attempt_ids"]), 1)
                checked.append(True)

        result = self.run_target(runner, responder, checkpoint=checkpoint)
        self.assertEqual(checked, [True])
        self.assertEqual(result["budget"]["accounted_cost_cny"], "0.000310")
        self.assertEqual(result["budget"]["reserved_cny"], "0.000000")
        record = read_records(runner.result_path)[0]
        item = next(iter(guard.inspect()["attempts"].values()))
        self.assertEqual(item["evidence_hash"], record["record_hash"])

    def test_insufficient_budget_prevents_call_and_start_marker(self):
        self.config["budget"]["project_total_cap"] = "0.2"
        guard = self.guard(cap="0.2")
        runner = self.runner(guard)
        before = self.ledger_path.read_bytes()
        responder = SimulatedResponder([RESPONSE])
        result = self.run_target(runner, responder)
        self.assertEqual(result["reason"], "budget_insufficient")
        self.assertEqual(result["new_attempts"], 0)
        self.assertEqual(responder.calls, 0)
        self.assertEqual(runner.journal_path.read_text(), "")
        self.assertEqual(read_records(runner.result_path), [])
        self.assertEqual(self.ledger_path.read_bytes(), before)

    def test_different_runs_share_one_project_budget(self):
        self.config["budget"]["project_total_cap"] = "0.2206"
        guard = self.guard(cap="0.2206")
        first = SimulatedResponder([RESPONSE])
        self.run_target(self.runner(guard), first)
        self.assertEqual(first.calls, 1)

        resumed_guard = self.guard(cap="0.2206", create=False)
        second = SimulatedResponder([RESPONSE])
        second_runner = self.runner(
            resumed_guard, path=self.root / "runB", run_id="runB",
        )
        result = self.run_target(second_runner, second)
        self.assertEqual(second.calls, 0)
        self.assertEqual(result["reason"], "budget_insufficient")
        self.assertEqual(result["budget"]["accounted_cost_cny"], "0.000310")

    def test_known_error_usage_is_charged_before_retry(self):
        guard = self.guard()
        runner = self.runner(guard)
        usage = {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105}
        responder = SimulatedResponder([
            SimulatedAPIError("connection", token_usage=usage), RESPONSE,
        ])
        result = self.run_target(runner, responder)
        self.assertEqual(responder.calls, 2)
        self.assertEqual(result["planned_wait_seconds"], [2])
        self.assertEqual(result["budget"]["accounted_cost_cny"], "0.001410")
        records = read_records(runner.result_path)
        self.assertEqual(records[0]["token_usage"], usage)
        items = guard.inspect()["attempts"]
        self.assertEqual(len(items), 2)
        self.assertTrue(all(item["status"] == "settled" for item in items.values()))

    def test_retry_cannot_bypass_budget_limit(self):
        self.config["budget"]["project_total_cap"] = "0.221"
        guard = self.guard(cap="0.221")
        runner = self.runner(guard)
        responder = SimulatedResponder([
            SimulatedAPIError("server_error", token_usage={
                "prompt_tokens": 10000, "completion_tokens": 0, "total_tokens": 10000,
            }),
            RESPONSE,
        ])
        result = self.run_target(runner, responder)
        self.assertEqual(responder.calls, 1)
        self.assertEqual(result["reason"], "budget_insufficient")
        self.assertEqual(result["budget"]["accounted_cost_cny"], "0.100000")
        self.assertEqual(len(read_records(runner.result_path)), 1)

    def test_transient_errors_without_usage_are_not_assumed_free(self):
        for kind in ("connection", "timeout", "rate_limit", "server_error"):
            with self.subTest(kind=kind):
                guard = self.guard(path=self.root / f"budget_{kind}.jsonl")
                runner = self.runner(guard, path=self.root / kind, run_id=kind)
                responder = SimulatedResponder([SimulatedAPIError(kind), RESPONSE])
                result = self.run_target(runner, responder)
                self.assertEqual(responder.calls, 1)
                self.assertEqual(result["reason"], "budget_usage_unknown")
                self.assertEqual(result["budget"]["reserved_cny"], "0.220480")
                retry = SimulatedResponder([RESPONSE])
                self.run_target(runner, retry)
                self.assertEqual(retry.calls, 0)

    def test_missing_or_inconsistent_usage_blocks_without_recall(self):
        variants = (
            None, {}, {"prompt_tokens": 25},
            {"prompt_tokens": 25, "completion_tokens": 3, "total_tokens": 99},
        )
        for index, usage in enumerate(variants):
            with self.subTest(usage=usage):
                guard = self.guard(path=self.root / f"budget_{index}.jsonl")
                runner = self.runner(guard, path=self.root / f"usage_{index}")
                response = copy.deepcopy(RESPONSE)
                response["token_usage"] = usage
                result = self.run_target(runner, SimulatedResponder([response]))
                self.assertEqual(result["reason"], "budget_usage_unknown")
                self.assertEqual(len(read_records(runner.result_path)), 1)
                second = SimulatedResponder([RESPONSE])
                self.run_target(runner, second)
                self.assertEqual(second.calls, 0)
                self.assertEqual(guard.summary()["reserved_cny"], "0.220480")

    def test_logged_result_recovers_fee_once_without_recall(self):
        guard = self.guard()
        runner = self.runner(guard)
        responder = SimulatedResponder([RESPONSE])
        with self.assertRaises(InjectedCrash):
            self.run_target(runner, responder, checkpoint=self.crash_at("after_log"))
        self.assertEqual(guard.summary()["accounted_cost_cny"], "0.000000")
        before_result = runner.result_path.read_bytes()
        resumed = self.runner(self.guard(create=False), create=False)
        second = SimulatedResponder([])
        self.run_target(resumed, second)
        before_ledger = self.ledger_path.read_bytes()
        self.run_target(resumed, second)
        self.assertEqual(second.calls, 0)
        self.assertEqual(guard.summary()["accounted_cost_cny"], "0.000310")
        self.assertEqual(resumed.result_path.read_bytes(), before_result)
        self.assertEqual(self.ledger_path.read_bytes(), before_ledger)

    def test_crash_after_fee_settlement_does_not_charge_twice(self):
        guard = self.guard()
        runner = self.runner(guard)
        with self.assertRaises(InjectedCrash):
            self.run_target(
                runner, SimulatedResponder([RESPONSE]),
                checkpoint=self.crash_at("after_budget_settle"),
            )
        before = self.ledger_path.read_bytes()
        second = SimulatedResponder([])
        self.run_target(self.runner(self.guard(create=False), create=False), second)
        self.assertEqual(second.calls, 0)
        self.assertEqual(self.ledger_path.read_bytes(), before)
        self.assertEqual(guard.summary()["accounted_cost_cny"], "0.000310")

    def test_crash_between_reservation_and_start_keeps_budget_blocked(self):
        guard = self.guard()
        runner = self.runner(guard)
        first = SimulatedResponder([RESPONSE])
        with self.assertRaises(InjectedCrash):
            self.run_target(runner, first, checkpoint=self.crash_at("after_budget_reserve"))
        self.assertEqual(first.calls, 0)
        self.assertEqual(runner.journal_path.read_text(), "")
        second = SimulatedResponder([RESPONSE])
        result = self.run_target(self.runner(self.guard(create=False), create=False), second)
        self.assertEqual(second.calls, 0)
        self.assertEqual(result["reason"], "budget_unconfirmed")
        self.assertEqual(guard.summary()["reserved_cny"], "0.220480")

    def test_crash_without_logged_result_preserves_reservation(self):
        for stage in ("after_start", "after_response"):
            with self.subTest(stage=stage):
                guard = self.guard(path=self.root / f"{stage}.jsonl")
                runner = self.runner(guard, path=self.root / stage)
                first = SimulatedResponder([RESPONSE])
                with self.assertRaises(InjectedCrash):
                    self.run_target(runner, first, checkpoint=self.crash_at(stage))
                second = SimulatedResponder([RESPONSE])
                result = self.run_target(runner, second)
                self.assertEqual(second.calls, 0)
                self.assertEqual(result["reason"], "unresolved_attempt")
                self.assertEqual(guard.summary()["reserved_cny"], "0.220480")

    def test_wrong_or_truncated_answer_is_charged_but_not_retried(self):
        for index, (text, reason) in enumerate((
            ("Answer: True", "stop"),
            ("answer: false", "stop"),
            ("Answer: False", "length"),
        )):
            with self.subTest(text=text, reason=reason):
                guard = self.guard(path=self.root / f"content_{index}.jsonl")
                runner = self.runner(guard, path=self.root / f"content_{index}")
                response = copy.deepcopy(RESPONSE)
                response.update(raw_output=text, finish_reason=reason)
                first = SimulatedResponder([response, RESPONSE])
                result = self.run_target(runner, first)
                self.assertEqual(result["action"], "skip")
                self.assertEqual(first.calls, 1)
                self.assertEqual(guard.summary()["accounted_cost_cny"], "0.000310")

    def test_corrupt_budget_stops_before_call_without_repair(self):
        guard = self.guard()
        runner = self.runner(guard)
        with self.ledger_path.open("a", encoding="utf-8") as file:
            file.write('{"unfinished":')
        before = self.ledger_path.read_bytes()
        responder = SimulatedResponder([RESPONSE])
        with self.assertRaises(BudgetError):
            self.run_target(runner, responder)
        self.assertEqual(responder.calls, 0)
        self.assertEqual(self.ledger_path.read_bytes(), before)
        self.assertEqual(runner.journal_path.read_text(), "")

    def test_resume_cannot_switch_to_fresh_budget_or_bounds(self):
        guard = self.guard()
        self.runner(guard)
        replacement = self.guard(path=self.root / "other_budget.jsonl")
        with self.assertRaises(ValueError):
            self.runner(replacement, create=False)
        with self.assertRaises(ValueError):
            self.runner(guard, create=False, bound=30000)

    def test_cap_or_currency_mismatch_is_rejected_before_directory_creation(self):
        guard = self.guard()
        for changes in (
            {"currency": "CNY", "project_total_cap": 60},
            {"currency": "USD", "project_total_cap": 50},
        ):
            with self.subTest(changes=changes):
                self.config["budget"] = changes
                with self.assertRaises(ValueError):
                    self.runner(guard)
                self.assertFalse(self.path.exists())

    def test_output_bound_violation_is_recorded_and_halts_resume(self):
        guard = self.guard()
        runner = self.runner(guard)
        response = copy.deepcopy(RESPONSE)
        response["token_usage"] = {
            "prompt_tokens": 25, "completion_tokens": 1025, "total_tokens": 1050,
        }
        responder = SimulatedResponder([response])
        with self.assertRaises(BudgetError):
            self.run_target(runner, responder)
        self.assertEqual(responder.calls, 1)
        self.assertTrue(guard.summary()["halted"])
        self.assertEqual(guard.summary()["accounted_cost_cny"], "0.020750")
        second = SimulatedResponder([RESPONSE])
        result = self.run_target(self.runner(self.guard(create=False), create=False), second)
        self.assertEqual(second.calls, 0)
        self.assertEqual(result["reason"], "budget_halted")

    def test_settlement_evidence_must_match_logged_result(self):
        guard = self.guard()
        runner = self.runner(guard)
        with self.assertRaises(InjectedCrash):
            self.run_target(
                runner, SimulatedResponder([RESPONSE]),
                checkpoint=self.crash_at("after_log"),
            )
        attempt_id = next(iter(guard.inspect()["attempts"]))
        guard.settle(attempt_id, input_tokens=25, output_tokens=3, evidence_hash="1" * 64)
        before_budget = self.ledger_path.read_bytes()
        before_journal = runner.journal_path.read_bytes()
        responder = SimulatedResponder([])
        with self.assertRaises(ValueError):
            self.run_target(runner, responder)
        self.assertEqual(responder.calls, 0)
        self.assertEqual(self.ledger_path.read_bytes(), before_budget)
        self.assertEqual(runner.journal_path.read_bytes(), before_journal)

    def test_unexpected_simulator_error_preserves_unknown_reservation(self):
        guard = self.guard()
        runner = self.runner(guard)
        with self.assertRaises(RuntimeError):
            self.run_target(runner, SimulatedResponder([]))
        item = next(iter(guard.inspect()["attempts"].values()))
        self.assertEqual(item["status"], "unknown")
        self.assertEqual(guard.summary()["reserved_cny"], "0.220480")
        second = SimulatedResponder([RESPONSE])
        result = self.run_target(runner, second)
        self.assertEqual(second.calls, 0)
        self.assertEqual(result["reason"], "unresolved_attempt")

    def test_busy_project_ledger_prevents_call(self):
        guard = self.guard()
        runner = self.runner(guard)
        responder = SimulatedResponder([RESPONSE])
        before = self.ledger_path.read_bytes()
        with self.ledger_path.open("r+") as other_owner:
            fcntl.flock(other_owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                with self.assertRaises(BudgetError):
                    self.run_target(runner, responder)
            finally:
                fcntl.flock(other_owner, fcntl.LOCK_UN)
        self.assertEqual(responder.calls, 0)
        self.assertEqual(self.ledger_path.read_bytes(), before)
        self.assertEqual(runner.journal_path.read_text(), "")

    def test_failed_reservation_prevents_call(self):
        guard = self.guard()
        runner = self.runner(guard)
        responder = SimulatedResponder([RESPONSE])
        with patch.object(guard, "reserve", side_effect=BudgetError("模拟预留失败")):
            result = self.run_target(runner, responder)
        self.assertEqual(result["reason"], "budget_reservation_blocked")
        self.assertEqual(responder.calls, 0)
        self.assertEqual(runner.journal_path.read_text(), "")

    def test_reservation_written_before_exception_remains_blocking(self):
        guard = self.guard()
        runner = self.runner(guard)
        real_reserve = guard.reserve

        def interrupted_reserve(*args, **kwargs):
            real_reserve(*args, **kwargs)
            raise BudgetError("模拟预留已落盘但返回前中断")

        responder = SimulatedResponder([RESPONSE])
        with patch.object(guard, "reserve", side_effect=interrupted_reserve):
            result = self.run_target(runner, responder)
        self.assertEqual(result["reason"], "budget_reservation_blocked")
        self.assertEqual(result["budget"]["reserved_cny"], "0.220480")
        self.assertEqual(responder.calls, 0)
        self.assertEqual(runner.journal_path.read_text(), "")
        self.assertEqual(self.run_target(runner, responder)["reason"], "budget_unconfirmed")
        self.assertEqual(responder.calls, 0)

    def test_unknown_fee_in_another_run_blocks_shared_project(self):
        guard = self.guard()
        first = SimulatedResponder([SimulatedAPIError("timeout")])
        self.run_target(self.runner(guard), first)
        second_runner = self.runner(guard, path=self.root / "runB", run_id="runB")
        second = SimulatedResponder([RESPONSE])
        result = self.run_target(second_runner, second)
        self.assertEqual(second.calls, 0)
        self.assertEqual(result["reason"], "budget_unconfirmed")
        self.assertEqual(result["budget"]["reserved_cny"], "0.220480")


if __name__ == "__main__":
    unittest.main()
