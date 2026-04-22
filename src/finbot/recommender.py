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


def _parse_payload(raw: object) -> RecommendationPayload | None:
    # Some adapters return structured JSON (dict), others return text.
    if isinstance(raw, dict):
        try:
            return RecommendationPayload.model_validate(raw)
        except Exception:
            return None

    if not isinstance(raw, str):
        return None

    # Fast path for valid JSON strings.
    try:
        return RecommendationPayload.model_validate_json(raw)
    except Exception:
        pass

    # Fallback path for text that may contain one or more JSON objects.
    dec = json.JSONDecoder()
    i = 0
    while i < len(raw):
        if raw[i] != "{":
            i += 1
            continue
        try:
            obj, _end = dec.raw_decode(raw[i:])
            if isinstance(obj, dict):
                return RecommendationPayload.model_validate(obj)
        except Exception:
            pass
        i += 1

    # Last attempt: greedy extraction for legacy cases.
    blob = _extract_json(raw)
    if blob:
        try:
            return RecommendationPayload.model_validate_json(blob)
        except Exception:
            pass

    return None


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
    
    print(f"\n\n=== Raw Answer ===\n {raw}\n\n")

    parsed = _parse_payload(raw)
    if parsed:
        if _looks_machine_like(parsed.profile_summary):
            parsed.profile_summary = _natural_profile_summary(collected)
        return parsed

    # single repair retry
    repair_prompt = prompt + "\n\nYour previous answer was invalid. Return ONLY valid JSON."
    raw2 = generate(repair_prompt, model_id)
    parsed2 = _parse_payload(raw2)
    if parsed2:
        if _looks_machine_like(parsed2.profile_summary):
            parsed2.profile_summary = _natural_profile_summary(collected)
        return parsed2

    return _fallback(collected, unknown_fields)