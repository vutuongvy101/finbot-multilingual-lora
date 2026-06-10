from __future__ import annotations

import pytest

from finbot.parsers import is_unknown_answer, parse_task_mode, validate_field_answer
from finbot.schemas import TaskMode


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1", TaskMode.PLANNING),
        ("planning", TaskMode.PLANNING),
        ("lập kế hoạch", TaskMode.PLANNING),
        ("规划", TaskMode.PLANNING),
        ("2", TaskMode.INVESTMENT),
        ("đầu tư", TaskMode.INVESTMENT),
        ("投资", TaskMode.INVESTMENT),
        ("3", TaskMode.TRADING),
        ("giao dịch", TaskMode.TRADING),
        ("交易", TaskMode.TRADING),
    ],
)
def test_parse_task_mode(text: str, expected: TaskMode) -> None:
    assert parse_task_mode(text) == expected


@pytest.mark.parametrize("text", ["", "4", "crypto", "random"])
def test_parse_task_mode_invalid(text: str) -> None:
    assert parse_task_mode(text) is None


@pytest.mark.parametrize(
    "text",
    ["unknown", "skip", "not sure", "không biết", "bỏ qua", "不知道", "跳过"],
)
def test_is_unknown_answer(text: str) -> None:
    assert is_unknown_answer(text) is True


def test_is_unknown_answer_case_insensitive() -> None:
    assert is_unknown_answer("  SKIP  ") is True
    assert is_unknown_answer("retire early") is False


def test_validate_goal_accepts_non_empty() -> None:
    ok, value = validate_field_answer("GOAL", "  Buy a home  ")
    assert ok is True
    assert value == "Buy a home"


def test_validate_goal_rejects_empty() -> None:
    ok, value = validate_field_answer("GOAL", "   ")
    assert ok is False
    assert value is None


@pytest.mark.parametrize(
    ("raw", "normalized"),
    [
        ("2", "60-120K"),
        ("60k", "60-120K"),
        ("6万以下", "<60K"),
        ("trung bình", "MEDIUM"),
        ("短期", "SHORT"),
    ],
)
def test_validate_field_answer_normalized(raw: str, normalized: str) -> None:
    field = {
        "60-120K": "INCOME_BAND",
        "<60K": "INCOME_BAND",
        "MEDIUM": "RISK_TOLERANCE",
        "SHORT": "TIME_HORIZON",
    }[normalized]
    ok, value = validate_field_answer(field, raw)
    assert ok is True
    assert value == normalized


def test_validate_field_answer_rejects_invalid_choice() -> None:
    ok, value = validate_field_answer("RISK_TOLERANCE", "extreme")
    assert ok is False
    assert value is None
