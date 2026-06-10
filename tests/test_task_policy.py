from __future__ import annotations

from finbot.schemas import TaskMode
from finbot.task_policy import next_unfilled_field, recommendation_ready


def test_next_unfilled_field_planning_order() -> None:
    collected: dict[str, str] = {"GOAL": "Retire early"}
    assert next_unfilled_field(TaskMode.PLANNING, collected) == "INCOME_BAND"


def test_next_unfilled_field_none_when_complete() -> None:
    collected = {
        "GOAL": "Grow savings",
        "INCOME_BAND": "60-120K",
        "CAPITAL_RANGE": "10-50K",
        "TIME_HORIZON": "LONG",
        "RISK_TOLERANCE": "MEDIUM",
    }
    assert next_unfilled_field(TaskMode.PLANNING, collected) is None


def test_recommendation_ready_when_all_fields_filled() -> None:
    collected = {
        "GOAL": "Buy a home",
        "CAPITAL_RANGE": "10-50K",
        "TIME_HORIZON": "MEDIUM",
        "RISK_TOLERANCE": "LOW",
    }
    ready, unknown = recommendation_ready(TaskMode.INVESTMENT, collected)
    assert ready is True
    assert unknown == []


def test_recommendation_ready_rejects_too_many_unknowns() -> None:
    collected = {
        "GOAL": "UNKNOWN",
        "INCOME_BAND": "UNKNOWN",
        "CAPITAL_RANGE": "UNKNOWN",
        "TIME_HORIZON": "LONG",
        "RISK_TOLERANCE": "MEDIUM",
    }
    ready, unknown = recommendation_ready(TaskMode.PLANNING, collected)
    assert ready is False
    assert set(unknown) == {"GOAL", "INCOME_BAND", "CAPITAL_RANGE"}
