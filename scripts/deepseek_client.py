"""单次调用模块；没有真实客户端默认值，不会自动发起请求。"""

import copy
import json
import os
import time
from pathlib import Path


def load_config(path):
    with Path(path).open(encoding="utf-8") as file:
        config = json.load(file)
    validate_config(config)
    return config


def validate_config(config):
    api = config["api"]
    runtime = config["runtime"]

    if config["stage"] != "dev" or config["data"]["allow_test"] is not False:
        raise ValueError("当前模块只接受禁止test的dev配置。")
    if api["base_url"] != "https://api.deepseek.com":
        raise ValueError("API地址不是已确认的DeepSeek地址。")
    if api["endpoint"] != "/chat/completions":
        raise ValueError("当前只支持Chat Completions接口。")
    if api["model"] != "deepseek-v4-pro":
        raise ValueError("模型与当前研究配置不一致。")
    if api["thinking"] != {"type": "disabled"}:
        raise ValueError("必须显式关闭API思考模式。")
    if api["stream"] is not False:
        raise ValueError("当前只支持非流式响应。")
    if runtime["sdk_max_retries"] != 0:
        raise ValueError("必须关闭SDK自动重试。")
    if config["credentials"]["api_key_env"] != "DEEPSEEK_API_KEY":
        raise ValueError("密钥环境变量名称不符合当前配置。")


def build_request(config, messages):
    validate_config(config)

    if not isinstance(messages, list) or not messages:
        raise ValueError("messages必须是非空列表。")
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("每条消息必须是字典。")
        if message.get("role") not in ("system", "user", "assistant"):
            raise ValueError("当前不支持该消息角色。")
        if not isinstance(message.get("content"), str):
            raise ValueError("消息内容必须是字符串。")

    api = config["api"]
    return {
        "model": api["model"],
        "messages": copy.deepcopy(messages),
        "temperature": api["temperature"],
        "max_tokens": api["max_tokens"],
        "stream": api["stream"],
        "response_format": copy.deepcopy(api["response_format"]),
        "extra_body": {"thinking": copy.deepcopy(api["thinking"])},
    }


def request_once(config, messages, *, client_factory=None):
    if client_factory is None:
        raise RuntimeError("未启用真实调用：必须显式传入客户端工厂。")

    request = build_request(config, messages)
    key = os.environ.get(config["credentials"]["api_key_env"], "")
    if not key or any(character.isspace() for character in key):
        raise ValueError("密钥缺失或包含空白字符。")

    start = time.perf_counter()
    with client_factory(
        api_key=key,
        base_url=config["api"]["base_url"],
        timeout=config["runtime"]["timeout_seconds"],
        max_retries=config["runtime"]["sdk_max_retries"],
    ) as client:
        response = client.chat.completions.create(**request)

    choice = response.choices[0]
    return {
        "raw_output": choice.message.content,
        "finish_reason": choice.finish_reason,
        "model": response.model,
        "response_id": response.id,
        "token_usage": (
            response.usage.model_dump() if response.usage is not None else None
        ),
        "latency_seconds": time.perf_counter() - start,
    }
