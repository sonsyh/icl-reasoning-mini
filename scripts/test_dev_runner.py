"""全部使用模拟输入和临时目录，不读取真实数据集或调用API。"""

import copy
import tempfile
import unittest
from pathlib import Path

from budget_guard import BudgetGuard
from dev_runner import (
    DevRunner,
    SimulatedAPIError,
    SimulatedResponder,
)
from experiment_log import ARTIFACT_PATHS, read_records
from test_experiment_log import CONFIG, RESPONSE, TARGET


ZERO_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


class InjectedCrash(RuntimeError):
    pass


class DevRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "offline_run"
        self.artifacts = {
            name: "0" * 64 for name in ARTIFACT_PATHS
        }

    def tearDown(self):
        self.temp.cleanup()

    def runner(self, *, create=True, config=None, **kwargs):
        config = copy.deepcopy(CONFIG if config is None else config)
        config.setdefault("budget", {"currency": "CNY", "project_total_cap": 50})
        budget_path = self.path.with_suffix(".budget.jsonl")
        guard = BudgetGuard(
            budget_path,
            project_id="offline_test_project",
            cap_cny=config["budget"]["project_total_cap"],
            input_rate_per_million="10",
            output_rate_per_million="20",
            create=not budget_path.exists(),
        )
        return DevRunner(
            self.path,
            run_id="offline_runner_test",
            config=config,
            artifact_hashes=self.artifacts,
            budget_guard=guard,
            max_input_tokens=20000,
            create=create,
            **kwargs,
        )

    def run_target(self, runner, responder, **kwargs):
        return runner.run_one(
            copy.deepcopy(TARGET),
            "C0",
            [],
            responder=responder,
            **kwargs,
        )

    def crash_at(self, name):
        def checkpoint(stage):
            if stage == name:
                raise InjectedCrash(stage)
        return checkpoint

    def test_success_and_resume(self):
        first = SimulatedResponder([RESPONSE])
        result = self.run_target(self.runner(), first)
        self.assertEqual(first.calls, 1)
        self.assertEqual(result["action"], "skip")
        self.assertEqual(result["new_attempts"], 1)

        second = SimulatedResponder([])
        result = self.run_target(
            self.runner(create=False), second
        )
        self.assertEqual(second.calls, 0)
        self.assertEqual(result["new_attempts"], 0)
        self.assertEqual(len(read_records(
            self.path / "results.jsonl"
        )), 1)

    def test_transient_retry(self):
        responder = SimulatedResponder([
            SimulatedAPIError("connection", token_usage=ZERO_USAGE),
            RESPONSE,
        ])
        result = self.run_target(self.runner(), responder)
        records = read_records(self.path / "results.jsonl")

        self.assertEqual(responder.calls, 2)
        self.assertEqual(result["planned_wait_seconds"], [2])
        self.assertEqual(
            [r["attempt_no"] for r in records], [1, 2]
        )
        self.assertEqual(records[0]["error_kind"], "connection")
        self.assertTrue(records[1]["correct"])
        self.assertTrue(all(r["is_simulated"] for r in records))

    def test_retry_limit(self):
        responder = SimulatedResponder([
            # 已确认零用量仅为模拟证据，不能从timeout错误名称推断。
            SimulatedAPIError("timeout", token_usage=ZERO_USAGE)
            for _ in range(3)
        ])
        result = self.run_target(self.runner(), responder)

        self.assertEqual(responder.calls, 3)
        self.assertEqual(result["action"], "stop")
        self.assertEqual(result["planned_wait_seconds"], [2, 4])

        second = SimulatedResponder([])
        result = self.run_target(
            self.runner(create=False), second
        )
        self.assertEqual(second.calls, 0)
        self.assertEqual(result["action"], "stop")

    def test_non_retryable_error(self):
        responder = SimulatedResponder([
            SimulatedAPIError("authentication"),
            RESPONSE,
        ])
        result = self.run_target(self.runner(), responder)
        self.assertEqual(responder.calls, 1)
        self.assertEqual(result["action"], "block")
        self.assertEqual(result["reason"], "non_retryable_error")

    def test_content_not_retried(self):
        variants = (
            ("Answer: True", "stop"),
            ("answer: false", "stop"),
            ("Answer: False", "length"),
            ("无法回答", "stop"),
        )
        for index, (text, reason) in enumerate(variants):
            with self.subTest(text=text, reason=reason):
                self.path = Path(self.temp.name) / f"variant_{index}"
                response = copy.deepcopy(RESPONSE)
                response["raw_output"] = text
                response["finish_reason"] = reason

                responder = SimulatedResponder([response, RESPONSE])
                result = self.run_target(self.runner(), responder)
                self.assertEqual(responder.calls, 1)
                self.assertEqual(result["action"], "skip")

                second = SimulatedResponder([])
                self.run_target(
                    self.runner(create=False), second
                )
                self.assertEqual(second.calls, 0)

    def test_crash_before_result_blocks_resume(self):
        stages = ("after_start", "after_response")
        for index, stage in enumerate(stages):
            with self.subTest(stage=stage):
                self.path = Path(self.temp.name) / f"crash_{index}"
                responder = SimulatedResponder([RESPONSE])

                with self.assertRaises(InjectedCrash):
                    self.run_target(
                        self.runner(),
                        responder,
                        checkpoint=self.crash_at(stage),
                    )

                self.assertEqual(
                    responder.calls,
                    0 if stage == "after_start" else 1,
                )
                second = SimulatedResponder([RESPONSE])
                result = self.run_target(
                    self.runner(create=False), second
                )
                self.assertEqual(second.calls, 0)
                self.assertEqual(result["action"], "block")
                self.assertEqual(
                    result["reason"], "unresolved_attempt"
                )

    def test_logged_result_is_reconciled_without_recall(self):
        responder = SimulatedResponder([RESPONSE])
        with self.assertRaises(InjectedCrash):
            self.run_target(
                self.runner(),
                responder,
                checkpoint=self.crash_at("after_log"),
            )

        self.assertEqual(responder.calls, 1)
        before = (self.path / "results.jsonl").read_bytes()

        second = SimulatedResponder([])
        runner = self.runner(create=False)
        result = self.run_target(runner, second)

        self.assertEqual(second.calls, 0)
        self.assertEqual(result["action"], "skip")
        self.assertEqual(
            (self.path / "results.jsonl").read_bytes(), before
        )
        self.assertEqual(
            [e["state"] for e in runner.read_events()],
            ["started", "recorded"],
        )

    def test_context_drift_is_blocked(self):
        self.runner()
        changed = copy.deepcopy(CONFIG)
        changed["api"]["temperature"] = 1
        with self.assertRaises(ValueError):
            self.runner(create=False, config=changed)

        self.artifacts[ARTIFACT_PATHS[0]] = "1" * 64
        with self.assertRaises(ValueError):
            self.runner(create=False)

    def test_offline_gate(self):
        with self.assertRaises(ValueError):
            self.runner(is_simulated=False)
        self.assertFalse(self.path.exists())

        runner = self.runner()
        with self.assertRaises(TypeError):
            self.run_target(runner, object())
        self.assertEqual(
            (self.path / "attempts.jsonl").read_text(), ""
        )

    def test_broken_journal_is_not_repaired(self):
        runner = self.runner()
        path = self.path / "attempts.jsonl"

        # 故意在临时目录制造不完整尾行。
        with path.open("a", encoding="utf-8") as file:
            file.write('{"state":')

        before = path.read_bytes()
        responder = SimulatedResponder([RESPONSE])
        with self.assertRaises(ValueError):
            self.run_target(runner, responder)

        self.assertEqual(responder.calls, 0)
        self.assertEqual(path.read_bytes(), before)

    def test_partial_result_is_not_repaired(self):
        runner = self.runner()
        path = self.path / "results.jsonl"
        with path.open("a", encoding="utf-8") as file:
            file.write('{"unfinished":')

        before = path.read_bytes()
        responder = SimulatedResponder([RESPONSE])
        with self.assertRaises(ValueError):
            self.run_target(runner, responder)

        self.assertEqual(responder.calls, 0)
        self.assertEqual(path.read_bytes(), before)

    def test_existing_directory_is_not_overwritten(self):
        self.runner()
        path = self.path / "run_manifest.json"
        before = path.read_bytes()

        with self.assertRaises(FileExistsError):
            self.runner()

        self.assertEqual(path.read_bytes(), before)

    def test_three_conditions_share_demo_selection(self):
        runner = self.runner()
        demos = []

        for label in ("True", "False"):
            item = copy.deepcopy(TARGET)
            item["id"] = "demo_" + label
            item["world_fingerprint"] = "demo_world_" + label
            item["label"] = label
            item["gold_cot"] = [
                "Alex is a wug.",
                "Wugs are vugs.",
                "Alex is a vug.",
                "Vugs are blue.",
                "Alex is blue.",
            ]
            item["query"] = (
                "Alex is blue."
                if label == "True" else "Alex is not blue."
            )
            demos.append(item)

        for condition in ("C0", "C1", "C2"):
            responder = SimulatedResponder([RESPONSE])
            runner.run_one(
                TARGET,
                condition,
                demos,
                responder=responder,
            )
            self.assertEqual(responder.calls, 1)

            # 模拟接口只收到请求消息与参数，没有评分用标签字段。
            self.assertNotIn("gold_label", responder.inputs[0])
            self.assertNotIn("depth", responder.inputs[0])

        records = read_records(self.path / "results.jsonl")
        self.assertEqual(len(records), 3)
        self.assertEqual(records[0]["demo_ids"], [])
        self.assertEqual(
            records[1]["demo_ids"], records[2]["demo_ids"]
        )
        self.assertEqual(len(records[1]["demo_ids"]), 2)


if __name__ == "__main__":
    unittest.main()
