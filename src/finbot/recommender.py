from __future__ import annotations

import json
from finbot.schemas import RecommendationPayload
from finbot.llm_adapter import generate


def _extract_json(text: str) -> str | None:
    s = text.find("{")
    e = text.rfind("}")
    if s == -1 or e == -1 or e <= s:
        return None
    return text[s:e + 1]


def _fallback(collected: dict[str, str], unknown_fields: list[str]) -> RecommendationPayload:
    return RecommendationPayload(
        profile_summary=_natural_profile_summary(collected),
        recommendation="Start with diversified, low-cost instruments and phased allocation aligned to your risk and horizon.",
        reasoning="Recommendation is based on provided goals, risk tolerance, and time horizon.",
        risks_caveats=(
            f"Unknown fields: {', '.join(unknown_fields)}." if unknown_fields else "Market risk and execution risk still apply."
        ),
        sources=[],
        disclaimer="Educational information only; not personal financial advice."
    )


def _field_label(field: str) -> str:
    labels = {
        "GOAL": "financial goal",
        "INCOME_BAND": "annual income range",
        "CAPITAL_RANGE": "investment amount",
        "TIME_HORIZON": "time horizon",
        "RISK_TOLERANCE": "risk tolerance",
    }
    return labels.get(field, field.replace("_", " ").lower())


def _natural_profile_summary(collected: dict[str, str]) -> str:
    if not collected:
        return "Your profile information has been captured."

    ordered_fields = ["GOAL", "INCOME_BAND", "CAPITAL_RANGE", "TIME_HORIZON", "RISK_TOLERANCE"]
    parts: list[str] = []
    for field in ordered_fields:
        value = collected.get(field)
        if value:
            parts.append(f"your {_field_label(field)} is {value}")

    if not parts:
        return "Your profile information has been captured."

    if len(parts) == 1:
        return f"Based on your inputs, {parts[0]}."

    return f"Based on your inputs, {', '.join(parts[:-1])}, and {parts[-1]}."


def _looks_machine_like(summary: str) -> bool:
    signals = ("GOAL=", "INCOME_BAND=", "CAPITAL_RANGE=", "TIME_HORIZON=", "RISK_TOLERANCE=")
    return any(token in summary for token in signals)


def generate_recommendation(prompt: str, model_id: str, collected: dict[str, str], unknown_fields: list[str]) -> RecommendationPayload:
    raw = generate(prompt, model_id)
    blob = _extract_json(raw)
    if blob:
        try:
            parsed = RecommendationPayload.model_validate_json(blob)
            if _looks_machine_like(parsed.profile_summary):
                parsed.profile_summary = _natural_profile_summary(collected)
            return parsed
        except Exception:
            pass

    # single repair retry
    repair_prompt = prompt + "\n\nYour previous answer was invalid. Return ONLY valid JSON."
    raw2 = generate(repair_prompt, model_id)
    blob2 = _extract_json(raw2)
    if blob2:
        try:
            parsed = RecommendationPayload.model_validate_json(blob2)
            if _looks_machine_like(parsed.profile_summary):
                parsed.profile_summary = _natural_profile_summary(collected)
            return parsed
        except Exception:
            pass

    return _fallback(collected, unknown_fields)