from __future__ import annotations

from finbot.schemas import TaskMode

UNKNOWN_TOKENS: frozenset[str] = frozenset({
    "unknown", "i don't know", "i do not know", "not sure", "unsure", "skip",
    "không biết", "bỏ qua", "不知道", "跳过",
})

TASK_MODE_MAP: dict[str, TaskMode] = {
    "1": TaskMode.PLANNING,
    "planning": TaskMode.PLANNING,
    "lập kế hoạch": TaskMode.PLANNING,
    "规划": TaskMode.PLANNING,
    
    "2": TaskMode.INVESTMENT,
    "investment": TaskMode.INVESTMENT,
    "đầu tư": TaskMode.INVESTMENT,
    "投资": TaskMode.INVESTMENT,
    
    "3": TaskMode.TRADING,
    "trading": TaskMode.TRADING,
    "giao dịch": TaskMode.TRADING,
    "交易": TaskMode.TRADING,
}

_INCOME_BAND_MAP: dict[str, str] = {
    "1": "<60K", "under 60k": "<60K", "dưới 60k": "<60K", "6万以下": "<60K",
    "2": "60-120K", "60k": "60-120K", "60-120k": "60-120K",
    "3": "120K+", "above 120k": "120K+", "từ 120k": "120K+", "12万及以上": "120K+",
}

_CAPITAL_RANGE_MAP: dict[str, str] = {
    "1": "<10K", "under 10k": "<10K", "dưới 10k": "<10K",
    "2": "10-50K", "10k": "10-50K", "10-50k": "10-50K",
    "3": "50-250K", "50k": "50-250K", "50-250k": "50-250K",
    "4": "250K+", "above 250k": "250K+", "từ 250k": "250K+", "25万及以上": "250K+",
}

_TIME_HORIZON_MAP: dict[str, str] = {
    "1": "SHORT", "short": "SHORT", "ngắn hạn": "SHORT", "短期": "SHORT",
    "2": "MEDIUM", "medium": "MEDIUM", "trung hạn": "MEDIUM", "中期": "MEDIUM",
    "3": "LONG", "long": "LONG", "dài hạn": "LONG", "长期": "LONG",
}

_RISK_TOLERANCE_MAP: dict[str, str] = {
    "1": "LOW", "low": "LOW", "thấp": "LOW", "低": "LOW",
    "2": "MEDIUM", "medium": "MEDIUM", "trung bình": "MEDIUM", "中": "MEDIUM",
    "3": "HIGH", "high": "HIGH", "cao": "HIGH", "高": "HIGH",
}

_FIELD_MAPS: dict[str, dict[str, str]] = {
    "INCOME_BAND": _INCOME_BAND_MAP,
    "CAPITAL_RANGE": _CAPITAL_RANGE_MAP,
    "TIME_HORIZON": _TIME_HORIZON_MAP,
    "RISK_TOLERANCE": _RISK_TOLERANCE_MAP,
}


def parse_task_mode(text: str) -> TaskMode | None:
    return TASK_MODE_MAP.get(text.strip().lower())


def is_unknown_answer(text: str) -> bool:
    return text.strip().lower() in UNKNOWN_TOKENS


def validate_field_answer(field: str, text: str) -> tuple[bool, str | None]:
    """Returns (is_valid, normalized_value). GOAL accepts any non-empty text."""
    if field == "GOAL":
        stripped = text.strip()
        return (True, stripped) if stripped else (False, None)
    mapping = _FIELD_MAPS.get(field)
    if mapping is None:
        return True, text.strip()
    normalized = mapping.get(text.strip().lower())
    return (True, normalized) if normalized else (False, None)