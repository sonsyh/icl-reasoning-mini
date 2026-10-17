"""只使用模拟字符串的离线测试。"""

import unittest

from answer_parser import parse_answer, score_answer


class AnswerParserTests(unittest.TestCase):

    def test_standard_labels(self):
        for label in ("True", "False"):
            with self.subTest(label=label):
                result = score_answer(
                    f"Answer: {label}",
                    label,
                    finish_reason="stop",
                )
                self.assertTrue(result["strict_format_ok"])
                self.assertTrue(result["strict_correct"])
                self.assertTrue(result["lenient_correct"])

    def test_cot_and_blank_lines(self):
        text = (
            "The reasoning discusses True and False.\n"
            "Max is fruity, contradicting the query.\n"
            "\nAnswer: False\n\n  \n"
        )
        result = score_answer(text, "False", finish_reason="stop")
        self.assertEqual(result["last_nonempty_line"], "Answer: False")
        self.assertTrue(result["strict_correct"])

    def test_lenient_variants(self):
        variants = (
            "answer: false",
            "Answer:False",
            " Answer: False ",
            "False",
            "FALSE.",
            "Final answer: False",
            "The answer is false.",
            "Answer: False!",
            "Answer: False.",
        )
        for text in variants:
            with self.subTest(text=text):
                result = score_answer(
                    text, "False", finish_reason="stop"
                )
                self.assertFalse(result["strict_format_ok"])
                self.assertFalse(result["strict_correct"])
                self.assertTrue(result["lenient_correct"])

    def test_reject_non_answers(self):
        examples = (
            "True or False",
            "Answer: True or False",
            "Not True",
            "Answer: not True",
            "False?",
            "The query contains the word False.",
            "False because Max is fruity.",
            "Answer: False\nThank you.",
            "The answer might be False.",
        )
        for text in examples:
            with self.subTest(text=text):
                result = score_answer(
                    text, "False", finish_reason="stop"
                )
                self.assertIsNone(result["lenient_label"])
                self.assertFalse(result["strict_correct"])
                self.assertFalse(result["lenient_correct"])

    def test_conflicting_answers(self):
        for text in (
            "Answer: True\nAnswer: False",
            "The answer is true.\nAnswer: False",
        ):
            with self.subTest(text=text):
                result = score_answer(
                    text, "False", finish_reason="stop"
                )
                self.assertTrue(result["strict_format_ok"])
                self.assertEqual(result["parse_status"], "ambiguous")
                self.assertIsNone(result["lenient_label"])
                self.assertFalse(result["strict_correct"])
                self.assertFalse(result["lenient_correct"])

    def test_repeated_same_answer(self):
        result = score_answer(
            "Answer: False\nAnswer: False",
            "False",
            finish_reason="stop",
        )
        self.assertEqual(result["parse_status"], "ok")
        self.assertTrue(result["strict_correct"])

    def test_non_text_and_empty(self):
        for value in (None, {"answer": "False"}):
            with self.subTest(value=value):
                result = parse_answer(value)
                self.assertEqual(result["parse_status"], "non_text")
                self.assertIsNone(result["lenient_label"])

        for text in ("", " \n\t\n"):
            with self.subTest(text=text):
                result = parse_answer(text)
                self.assertEqual(result["parse_status"], "empty")
                self.assertFalse(result["strict_format_ok"])

    def test_completion_status(self):
        for reason in ("length", "content_filter", None, "unexpected"):
            with self.subTest(reason=reason):
                result = score_answer(
                    "Answer: False",
                    "False",
                    finish_reason=reason,
                )
                self.assertTrue(result["strict_format_ok"])
                self.assertFalse(result["completion_ok"])
                self.assertEqual(result["score_status"], "incomplete")
                self.assertFalse(result["strict_correct"])
                self.assertFalse(result["lenient_correct"])

    def test_invalid_gold_labels(self):
        for label in ("Unknown", "Both", "false", False, None):
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    score_answer(
                        "Answer: False",
                        label,
                        finish_reason="stop",
                    )

    def test_scoring_and_gold_independence(self):
        right = score_answer(
            "Answer: False", "False", finish_reason="stop"
        )
        wrong = score_answer(
            "Answer: False", "True", finish_reason="stop"
        )
        self.assertEqual(right["lenient_label"], wrong["lenient_label"])
        self.assertTrue(right["strict_correct"])
        self.assertTrue(wrong["strict_format_ok"])
        self.assertFalse(wrong["strict_correct"])
        self.assertFalse(wrong["lenient_correct"])


if __name__ == "__main__":
    unittest.main()
