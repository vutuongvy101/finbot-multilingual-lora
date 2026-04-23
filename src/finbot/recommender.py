from __future__ import annotations

import json
from finbot.schemas import RecommendationPayload, RecommendationError
from finbot.i18n import RECOMMENDATION_FAILED, t
from finbot.llm_adapter import generate

import logging
logger = logging.getLogger(__name__)


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


def generate_recommendation(prompt: str, model_id: str, lang: str) -> RecommendationPayload:
    raw = llm_raw_answer(prompt, model_id)

    parsed = _parse_payload(raw)
    if parsed:
        return parsed

    # single repair retry
    repair_prompt = prompt + "\n\nYour previous answer was invalid. Return ONLY valid JSON."
    raw2 = llm_raw_answer(repair_prompt, model_id)
    parsed2 = _parse_payload(raw2)
    if parsed2:
        return parsed2

    raise RecommendationError(t(RECOMMENDATION_FAILED, lang))


def llm_raw_answer(prompt: str, model_id: str) -> str:
    raw = generate(prompt, model_id)
    logger.info("Raw LLM answer: %s", raw)
    return raw
