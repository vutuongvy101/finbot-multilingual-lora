from __future__ import annotations

from dataclasses import dataclass

from finbot.schemas import TaskMode


@dataclass(frozen=True)
class TaskPolicy:
    required_fields: list[str]
    ask_order: list[str]


TASK_POLICIES: dict[TaskMode, TaskPolicy] = {
    TaskMode.PLANNING: TaskPolicy(
        required_fields=["GOAL", "INCOME_BAND", "CAPITAL_RANGE", "TIME_HORIZON", "RISK_TOLERANCE"],
        ask_order=["GOAL", "INCOME_BAND", "CAPITAL_RANGE", "TIME_HORIZON", "RISK_TOLERANCE"],
    ),
    TaskMode.INVESTMENT: TaskPolicy(
        required_fields=["GOAL", "CAPITAL_RANGE", "TIME_HORIZON", "RISK_TOLERANCE"],
        ask_order=["GOAL", "CAPITAL_RANGE", "TIME_HORIZON", "RISK_TOLERANCE"],
    ),
    TaskMode.TRADING: TaskPolicy(
        required_fields=["GOAL", "CAPITAL_RANGE", "RISK_TOLERANCE", "TIME_HORIZON"],
        ask_order=["GOAL", "CAPITAL_RANGE", "RISK_TOLERANCE", "TIME_HORIZON"],
    ),
}


def get_policy(mode: TaskMode) -> TaskPolicy:
    return TASK_POLICIES[mode]


def next_unfilled_field(mode: TaskMode, collected: dict[str, str]) -> str | None:
    policy = get_policy(mode)
    for field in policy.ask_order:
        if field not in collected:
            return field
    return None


def recommendation_ready(mode: TaskMode, collected: dict[str, str]) -> tuple[bool, list[str]]:
    policy = get_policy(mode)
    unknown_fields = [f for f in policy.required_fields if collected.get(f) == "UNKNOWN"]
    filled_count = sum(1 for f in policy.required_fields if f in collected)
    all_resolved = filled_count == len(policy.required_fields)
    # number of unknown fields must be less than half of the required fields
    too_many_unknown = len(unknown_fields) > (len(policy.required_fields) / 2)
    return all_resolved and not too_many_unknown, unknown_fields