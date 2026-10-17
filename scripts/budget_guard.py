"""模拟预算账本：精确金额、请求前预留、持久化与串行锁。

本版禁止真实计费，只使用调用者明确提供的模拟价格。它不发送请求，
不自动把超时或接口错误当作零费用，也不自动撤销未确认的预留。
"""

import copy
import hashlib
import json
import os
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from pathlib import Path

import fcntl


VERSION = "offline_budget_guard_v2_evidence"
UNIT = Decimal("0.000001")
MILLION = Decimal("1000000")


class BudgetError(ValueError):
    pass


def canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def amount(value, *, positive=False):
    """拒绝浮点数，金额最多六位小数。"""
    if type(value) not in (str, int, Decimal):
        raise BudgetError("金额必须使用字符串、整数或Decimal，不能用float。")

    try:
        result = Decimal(value)
    except InvalidOperation:
        raise BudgetError("金额不是有效数字。") from None

    if (
        not result.is_finite()
        or result < 0
        or result > Decimal("1000000000")
    ):
        raise BudgetError("金额必须是合理范围内的非负有限数。")
    if positive and result == 0:
        raise BudgetError("预算上限和模拟单价必须大于零。")
    if result != result.quantize(UNIT):
        raise BudgetError("金额最多允许六位小数。")
    return result


def money_text(value):
    return format(value.quantize(UNIT), ".6f")


def tokens(value):
    if type(value) is not int or not 0 <= value <= 100000000:
        raise BudgetError("token数量必须是范围内的非负整数。")
    return value


def identifier(value):
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[A-Za-z0-9_.:-]{1,200}", value) is None
    ):
        raise BudgetError("编号只能使用字母、数字、下划线、点、冒号或短横线。")
    return value


class BudgetGuard:
    def __init__(
        self,
        path,
        *,
        project_id,
        cap_cny,
        input_rate_per_million,
        output_rate_per_million,
        create=False,
        is_simulated=True,
    ):
        if is_simulated is not True:
            raise BudgetError("本版只允许模拟预算，不能用于真实计费。")

        self.path = Path(path)
        if self.path.suffix != ".jsonl":
            raise BudgetError("预算账本必须使用.jsonl扩展名。")

        self.context = {
            "version": VERSION,
            "project_id": identifier(project_id),
            "currency": "CNY",
            "is_simulated": True,
            "cap_cny": money_text(amount(cap_cny, positive=True)),
            "input_rate_per_million": money_text(
                amount(input_rate_per_million, positive=True)
            ),
            "output_rate_per_million": money_text(
                amount(output_rate_per_million, positive=True)
            ),
        }

        if create:
            entry = self.make_entry(self.context, 0, None)
            # x模式：已经存在就报错，不覆盖。调用者必须复用同一项目账本。
            with self.path.open("x", encoding="utf-8") as file:
                file.write(canonical(entry) + "\n")
                file.flush()
                os.fsync(file.fileno())

        self.summary()

    def cost(self, input_tokens, output_tokens):
        input_tokens = tokens(input_tokens)
        output_tokens = tokens(output_tokens)
        result = (
            Decimal(input_tokens)
            * Decimal(self.context["input_rate_per_million"])
            + Decimal(output_tokens)
            * Decimal(self.context["output_rate_per_million"])
        ) / MILLION
        # 向上取整，避免把很小的正费用舍入为零。
        return result.quantize(UNIT, rounding=ROUND_CEILING)

    @staticmethod
    def make_entry(data, sequence, previous_hash):
        entry = {
            "sequence": sequence,
            "previous_hash": previous_hash,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
        }
        entry["entry_hash"] = fingerprint(entry)
        return entry

    @contextmanager
    def locked_file(self):
        # r+要求账本已存在，不能在续跑时悄悄新建。
        with self.path.open("r+", encoding="utf-8") as file:
            try:
                fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise BudgetError("另一进程正在使用预算账本。") from None
            try:
                yield file
            finally:
                fcntl.flock(file, fcntl.LOCK_UN)

    def snapshot(self, state):
        pending = [
            attempt_id
            for attempt_id, item in state["attempts"].items()
            if item["status"] in ("reserved", "unknown")
        ]
        reserved = sum(
            (
                Decimal(state["attempts"][key]["reserved_cny"])
                for key in pending
            ),
            Decimal("0"),
        )
        available = (
            Decimal(self.context["cap_cny"]) - state["spent"] - reserved
        )
        return {
            "is_simulated": True,
            "cap_cny": self.context["cap_cny"],
            "accounted_cost_cny": money_text(state["spent"]),
            "reserved_cny": money_text(reserved),
            "available_cny": money_text(max(Decimal("0"), available)),
            "pending_attempt_ids": pending,
            "halted": state["halted"],
        }

    def apply_event(self, state, data):
        if not isinstance(data, dict):
            raise BudgetError("预算事件必须是字典。")
        event = data.get("event")
        attempt_id = identifier(data.get("attempt_id"))

        if event == "reserved":
            required = {
                "event", "attempt_id", "max_input_tokens",
                "max_output_tokens", "reserved_cny",
            }
            if set(data) != required:
                raise BudgetError("预留事件字段不正确。")
            if attempt_id in state["attempts"]:
                raise BudgetError("尝试编号已存在，不能重复预留。")
            view = self.snapshot(state)
            if view["halted"] or view["pending_attempt_ids"]:
                raise BudgetError("存在异常或未确认请求，不能预留新费用。")
            expected = self.cost(
                data["max_input_tokens"], data["max_output_tokens"]
            )
            if amount(data["reserved_cny"]) != expected:
                raise BudgetError("预留金额与token上界不一致。")
            if expected > Decimal(view["available_cny"]):
                raise BudgetError("剩余预算不足，禁止开始本次请求。")
            state["attempts"][attempt_id] = {
                **copy.deepcopy(data), "status": "reserved",
            }
            return

        if attempt_id not in state["attempts"]:
            raise BudgetError("尝试编号不存在。")
        item = state["attempts"][attempt_id]
        if item["status"] not in ("reserved", "unknown"):
            raise BudgetError("这次尝试已经核算或发生上界异常。")

        if event == "unknown":
            if set(data) != {"event", "attempt_id"}:
                raise BudgetError("不确定事件字段不正确。")
            if item["status"] != "reserved":
                raise BudgetError("已经标记为不确定，不能重复记录。")
            item["status"] = "unknown"
            return

        if event not in ("settled", "bound_violation"):
            raise BudgetError("未知预算事件。")
        required = {
            "event", "attempt_id", "input_tokens",
            "output_tokens", "accounted_cost_cny", "evidence_hash",
        }
        if set(data) != required:
            raise BudgetError("核算事件字段不正确。")
        input_count = tokens(data["input_tokens"])
        output_count = tokens(data["output_tokens"])
        expected_cost = self.cost(input_count, output_count)
        if amount(data["accounted_cost_cny"]) != expected_cost:
            raise BudgetError("核算费用与token用量不一致。")
        exceeded = (
            input_count > item["max_input_tokens"]
            or output_count > item["max_output_tokens"]
        )
        if exceeded != (event == "bound_violation"):
            raise BudgetError("token上界异常标记不一致。")
        evidence_hash = data["evidence_hash"]
        if evidence_hash is not None and (
            not isinstance(evidence_hash, str)
            or len(evidence_hash) != 64
            or any(c not in "0123456789abcdef" for c in evidence_hash)
        ):
            raise BudgetError("核算证据必须是小写SHA-256或None。")
        state["spent"] += expected_cost
        item["status"] = "violated" if exceeded else "settled"
        item["accounted_cost_cny"] = money_text(expected_cost)
        item["input_tokens"] = input_count
        item["output_tokens"] = output_count
        item["evidence_hash"] = evidence_hash
        if exceeded:
            # 已发生的用量不能隐去；保存费用并暂停后续请求。
            state["halted"] = True

    def read_state(self, file):
        file.seek(0)
        text = file.read()
        if not text or not text.endswith("\n"):
            raise BudgetError("预算账本为空或尾行不完整，不能自动修复。")
        state = {
            "attempts": {}, "spent": Decimal("0"), "halted": False,
        }
        entries = []
        previous_hash = None
        fields = {
            "sequence", "previous_hash", "timestamp", "data", "entry_hash",
        }
        for index, line in enumerate(text.splitlines()):
            entry = json.loads(line)
            if not isinstance(entry, dict) or set(entry) != fields:
                raise BudgetError("预算账本记录字段不正确。")
            payload = dict(entry)
            saved_hash = payload.pop("entry_hash")
            if saved_hash != fingerprint(payload):
                raise BudgetError("预算账本记录指纹不一致。")
            if (
                type(entry["sequence"]) is not int
                or entry["sequence"] != index
                or entry["previous_hash"] != previous_hash
            ):
                raise BudgetError("预算账本顺序或指纹链不一致。")
            if index == 0:
                if entry["data"] != self.context:
                    raise BudgetError("项目、预算、模拟价格或版本发生变化。")
            else:
                self.apply_event(state, entry["data"])
            entries.append(entry)
            previous_hash = saved_hash
        return state, entries

    def summary(self):
        with self.locked_file() as file:
            state, _ = self.read_state(file)
            return self.snapshot(state)

    def inspect(self):
        """锁内校验后提供副本，供运行器核对结果证据与续跑状态。"""
        with self.locked_file() as file:
            state, entries = self.read_state(file)
            return {
                "binding": {
                    "path": str(self.path.resolve()),
                    "context": copy.deepcopy(self.context),
                    "header_hash": entries[0]["entry_hash"],
                },
                "attempts": copy.deepcopy(state["attempts"]),
                "summary": self.snapshot(state),
            }

    def transact(self, event, attempt_id, **values):
        attempt_id = identifier(attempt_id)
        violation = False
        with self.locked_file() as file:
            state, entries = self.read_state(file)
            if event == "reserved":
                data = {
                    "event": event,
                    "attempt_id": attempt_id,
                    **values,
                    "reserved_cny": money_text(self.cost(
                        values["max_input_tokens"], values["max_output_tokens"]
                    )),
                }
            elif event == "settle":
                if attempt_id not in state["attempts"]:
                    raise BudgetError("尝试编号不存在。")
                item = state["attempts"][attempt_id]
                input_count = tokens(values["input_tokens"])
                output_count = tokens(values["output_tokens"])
                violation = (
                    input_count > item["max_input_tokens"]
                    or output_count > item["max_output_tokens"]
                )
                data = {
                    "event": "bound_violation" if violation else "settled",
                    "attempt_id": attempt_id,
                    "input_tokens": input_count,
                    "output_tokens": output_count,
                    "accounted_cost_cny": money_text(
                        self.cost(input_count, output_count)
                    ),
                    "evidence_hash": values.get("evidence_hash"),
                }
            elif event == "unknown":
                data = {"event": event, "attempt_id": attempt_id}
            else:
                raise BudgetError("不支持的预算操作。")

            # 先验证新事件；失败时不追加任何内容。
            next_state = copy.deepcopy(state)
            self.apply_event(next_state, data)
            entry = self.make_entry(data, len(entries), entries[-1]["entry_hash"])
            file.seek(0, os.SEEK_END)
            file.write(canonical(entry) + "\n")
            file.flush()
            os.fsync(file.fileno())
            result = self.snapshot(next_state)

        if violation:
            raise BudgetError(
                "实际token用量超过预留上界；已记录费用并暂停后续请求。"
            )
        return result

    def reserve(self, attempt_id, *, max_input_tokens, max_output_tokens):
        return self.transact(
            "reserved", attempt_id,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
        )

    def settle(
        self, attempt_id, *, input_tokens, output_tokens, evidence_hash=None,
    ):
        return self.transact(
            "settle", attempt_id,
            input_tokens=input_tokens, output_tokens=output_tokens,
            evidence_hash=evidence_hash,
        )

    def mark_unknown(self, attempt_id):
        return self.transact("unknown", attempt_id)
