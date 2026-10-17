"""仅使用模拟客户端和假密钥，不发起网络请求。"""

import copy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from deepseek_client import load_config, build_request, request_once


ROOT = Path(__file__).resolve().parents[1]


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "configs/inference_dev_v1.json")
        self.messages = [{"role": "user", "content": "Offline test fixture."}]
        self.factory = MagicMock()
        self.client = self.factory.return_value.__enter__.return_value
        self.client.chat.completions.create.return_value = SimpleNamespace(
            id="offline-response",
            model="offline-model",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Answer: True"),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                model_dump=lambda: {
                    "prompt_tokens": 10,
                    "completion_tokens": 3,
                    "total_tokens": 13,
                }
            ),
        )

    def test_request_parameters(self):
        request = build_request(self.config, self.messages)
        self.assertEqual(request["model"], "deepseek-v4-pro")
        self.assertEqual(request["temperature"], 0)
        self.assertEqual(request["max_tokens"], 1024)
        self.assertIs(request["stream"], False)
        self.assertEqual(
            request["extra_body"], {"thinking": {"type": "disabled"}}
        )
        self.assertNotIn("api_key", request)
        self.assertIsNot(request["messages"], self.messages)

    def test_simulated_response(self):
        with patch.dict(
            "os.environ", {"DEEPSEEK_API_KEY": "offline-test-key"}, clear=True
        ):
            result = request_once(
                self.config, self.messages, client_factory=self.factory
            )

        self.factory.assert_called_once_with(
            api_key="offline-test-key",
            base_url="https://api.deepseek.com",
            timeout=120,
            max_retries=0,
        )
        self.client.chat.completions.create.assert_called_once_with(
            **build_request(self.config, self.messages)
        )
        self.assertEqual(result["raw_output"], "Answer: True")
        self.assertEqual(result["token_usage"]["total_tokens"], 13)
        self.assertEqual(result["finish_reason"], "stop")
        self.factory.return_value.__exit__.assert_called_once()

    def test_default_call_is_blocked(self):
        with self.assertRaises(RuntimeError):
            request_once(self.config, self.messages)

    def test_missing_key_is_blocked(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError):
                request_once(
                    self.config, self.messages, client_factory=self.factory
                )
        self.factory.assert_not_called()

    def test_wrong_address_is_blocked(self):
        config = copy.deepcopy(self.config)
        config["api"]["base_url"] = "https://example.invalid"
        with self.assertRaises(ValueError):
            build_request(config, self.messages)

    def test_thinking_mode_is_blocked(self):
        config = copy.deepcopy(self.config)
        config["api"]["thinking"]["type"] = "enabled"
        with self.assertRaises(ValueError):
            build_request(config, self.messages)


if __name__ == "__main__":
    unittest.main()
