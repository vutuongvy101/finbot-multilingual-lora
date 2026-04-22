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
        profile_summary=", ".join(f"{k}={v}" for k, v in collected.items()) or "Profile captured.",
        recommendation="Start with diversified, low-cost instruments and phased allocation aligned to your risk and horizon.",
        reasoning="Recommendation is based on provided goals, risk tolerance, and time horizon.",
        risks_caveats=(
            f"Unknown fields: {', '.join(unknown_fields)}." if unknown_fields else "Market risk and execution risk still apply."
        ),
        sources=[],
        disclaimer="Educational information only; not personal financial advice."
    )


def generate_recommendation(prompt: str, model_id: str, collected: dict[str, str], unknown_fields: list[str]) -> RecommendationPayload:
    raw = generate(prompt, model_id)
    blob = _extract_json(raw)
    if blob:
        try:
            return RecommendationPayload.model_validate_json(blob)
        except Exception:
            pass

    # single repair retry
    repair_prompt = prompt + "\n\nYour previous answer was invalid. Return ONLY valid JSON."
    raw2 = generate(repair_prompt, model_id)
    blob2 = _extract_json(raw2)
    if blob2:
        try:
            return RecommendationPayload.model_validate_json(blob2)
        except Exception:
            pass

    return _fallback(collected, unknown_fields)