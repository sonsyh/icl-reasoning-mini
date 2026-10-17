"""Synthetic acceptance-checker tests; no real test-set text is used."""

import copy
import json
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import validate_dataset as validator


def sample():
    premises = ["Ava is a vumpus.", "Vumpuses are yumpuses.",
                "Yumpuses are not feisty."]
    query = "Ava is not feisty."
    return {
        "id": "prontoqa_v1_seed20261006_h2_example1",
        "source_file": "2hop_random_seed20261006.json",
        "source_example_id": "example1", "source_seed": 20261006, "depth": 2,
        "premises": premises, "query": query, "label": "True",
        "gold_cot": [premises[0], premises[1], "Ava is a yumpus.",
                     premises[2], query],
        "world_fingerprint": validator.digest(sorted(set(premises))),
        "prompt_fingerprint": validator.digest({"premises": premises, "query": query}),
    }


class LogicTests(unittest.TestCase):
    def test_valid_true(self):
        result = validator.logic_result(sample())
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["shortest_depth"], 2)
        self.assertTrue(result["gold_chain_passed"])

    def test_valid_false(self):
        record = sample()
        record["query"] = "Ava is feisty."
        record["label"] = "False"
        self.assertEqual(validator.logic_result(record)["errors"], [])

    def test_false_negative_query(self):
        record = sample()
        record["premises"] = [s.replace("not feisty", "blue") for s in record["premises"]]
        record["gold_cot"] = [s.replace("not feisty", "blue") for s in record["gold_cot"]]
        record["query"] = "Ava is not blue."
        record["label"] = "False"
        self.assertEqual(validator.logic_result(record)["errors"], [])

    def test_wrong_label(self):
        record = sample()
        record["label"] = "False"
        self.assertIn("label_mismatch", validator.logic_result(record)["errors"])

    def test_unknown_is_not_false(self):
        record = sample()
        record["premises"][-1] = "Yumpuses are kind."
        record["label"] = "False"
        result = validator.logic_result(record)
        self.assertEqual(result["actual_label"], "Unknown")
        self.assertIn("unknown_target", result["errors"])
        self.assertIn("label_mismatch", result["errors"])

    def test_both_sides_proven(self):
        record = sample()
        record["premises"].append("Yumpuses are feisty.")
        result = validator.logic_result(record)
        self.assertEqual(result["actual_label"], "Both")
        self.assertIn("reachable_contradiction", result["errors"])

    def test_other_attribute_contradiction(self):
        record = sample()
        record["premises"].extend(["Vumpuses are red.", "Vumpuses are not red."])
        result = validator.logic_result(record)
        self.assertEqual(result["actual_label"], "True")
        self.assertIn("reachable_contradiction", result["errors"])

    def test_shortcut_is_detected(self):
        record = sample()
        record["premises"].append("Vumpuses are not feisty.")
        result = validator.logic_result(record)
        self.assertEqual(result["shortest_depth"], 1)
        self.assertIn("shortest_depth_mismatch", result["errors"])

    def test_wrong_depth(self):
        record = sample()
        record["depth"] = 3
        self.assertIn("shortest_depth_mismatch", validator.logic_result(record)["errors"])

    def test_missing_gold_rule(self):
        record = sample()
        record["gold_cot"][1] = "Each vumpus is a yumpus."
        self.assertIn("gold_rule_not_in_premises", validator.logic_result(record)["errors"])

    def test_missing_initial_fact(self):
        record = sample()
        record["gold_cot"][0] = "Ava is a numpus."
        self.assertIn("gold_initial_fact", validator.logic_result(record)["errors"])

    def test_wrong_intermediate_person(self):
        record = sample()
        record["gold_cot"][2] = "Bea is a yumpus."
        self.assertIn("gold_invalid_conclusion", validator.logic_result(record)["errors"])

    def test_invalid_rule_condition(self):
        record = sample()
        rule = "Yumpuses are vumpuses."
        record["premises"].append(rule)
        record["gold_cot"][1] = rule
        self.assertIn("gold_rule_condition", validator.logic_result(record)["errors"])

    def test_wrong_final_conclusion(self):
        record = sample()
        record["gold_cot"][-1] = "Ava is feisty."
        self.assertIn("gold_invalid_conclusion", validator.logic_result(record)["errors"])

    def test_bad_chain_length(self):
        record = sample()
        record["gold_cot"].pop()
        self.assertIn("gold_chain_shape", validator.logic_result(record)["errors"])

    def test_unsupported_premise_fails(self):
        record = sample()
        record["premises"].append("Some vumpuses are kind.")
        result = validator.logic_result(record)
        self.assertEqual(result["actual_label"], "Unparsed")
        self.assertIn("unsupported_sentence", result["errors"])

    def test_unsupported_gold_sentence_fails(self):
        record = sample()
        record["gold_cot"][2] = "Therefore Ava is a yumpus."
        self.assertIn("unsupported_sentence", validator.logic_result(record)["errors"])

    def test_article_and_plural_normalization(self):
        self.assertEqual(validator.parse_sentence("Every numpus is an impus."),
                         ("rule", (("numpus", False), ("impus", False))))
        self.assertEqual(validator.parse_sentence("Ava is not a vumpus."),
                         ("fact", ("Ava", ("vumpus", True))))


class SchemaTests(unittest.TestCase):
    def test_valid_schema(self):
        self.assertEqual(validator.schema_errors(sample()), [])

    def test_missing_field(self):
        record = sample()
        del record["depth"]
        self.assertEqual(validator.schema_errors(record), ["schema_fields"])

    def test_boolean_label_is_rejected(self):
        record = sample()
        record["label"] = True
        self.assertIn("schema_label", validator.schema_errors(record))

    def test_boolean_depth_is_rejected(self):
        record = sample()
        record["depth"] = True
        self.assertIn("schema_depth", validator.schema_errors(record))

    def test_wrong_hash_is_rejected(self):
        record = sample()
        record["world_fingerprint"] = "0" * 64
        self.assertIn("world_fingerprint", validator.schema_errors(record))

    def test_world_invariant_prompt_order_sensitive(self):
        record = sample()
        original = copy.deepcopy(record)
        record["premises"] = list(reversed(record["premises"]))
        self.assertEqual(record["world_fingerprint"],
                         validator.digest(sorted(set(record["premises"]))))
        self.assertNotEqual(record["prompt_fingerprint"], validator.digest({
            "premises": record["premises"], "query": record["query"]}))
        self.assertEqual(original["gold_cot"], record["gold_cot"])






if __name__ == "__main__":
    unittest.main()
