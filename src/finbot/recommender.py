from __future__ import annotations

import json
import logging

from pydantic import ValidationError

from finbot.i18n import RECOMMENDATION_FAILED, t
from finbot.llm_adapter import generate_chat
from finbot.schemas import RecommendationError, RecommendationPayload

logger = logging.getLogger(__name__)

_MAX_ERROR_CHARS = 200


def _extract_json(text: str) -> str | None:
    s = text.find("{")
    e = text.rfind("}")
    if s == -1 or e == -1 or e <= s:
        return None
    return text[s:e + 1]


def _parse_payload(raw: object) -> tuple[RecommendationPayload | None, str | None]:
    if isinstance(raw, dict):
        try:
            return RecommendationPayload.model_validate(raw), None
        except ValidationError as exc:
            return None, str(exc)

    if not isinstance(raw, str):
        return None, "Response is not a string or dict."

    last_error: str | None = None

    try:
        return RecommendationPayload.model_validate_json(raw), None
    except ValidationError as exc:
        last_error = str(exc)
    except Exception as exc:
        last_error = str(exc)

    dec = json.JSONDecoder()
    i = 0
    while i < len(raw):
        if raw[i] != "{":
            i += 1
            continue
        try:
            obj, _end = dec.raw_decode(raw[i:])
            if isinstance(obj, dict):
                try:
                    return RecommendationPayload.model_validate(obj), None
                except ValidationError as exc:
                    last_error = str(exc)
        except Exception as exc:
            last_error = str(exc)
        i += 1

    blob = _extract_json(raw)
    if blob:
        try:
            return RecommendationPayload.model_validate_json(blob), None
        except ValidationError as exc:
            last_error = str(exc)
        except Exception as exc:
            last_error = str(exc)

    return None, last_error


def generate_recommendation(
    messages: list[dict], model_id: str, lang: str, adapter_path: str = None,
) -> RecommendationPayload:
    raw = generate_chat(messages, model_id, adapter_path)
    logger.info("Raw LLM answer: %s", raw)

    parsed, err = _parse_payload(raw)
    if parsed:
        return parsed

    truncated = (err or "Invalid JSON.")[:_MAX_ERROR_CHARS]
    logger.warning("Invalid recommendation output. Attempting repair. Error: %s", truncated)

    repair_messages = messages + [
        {"role": "assistant", "content": raw},
        {
            "role": "user",
            "content": (
                f"Your previous output was invalid. Error: {truncated}. "
                "Return ONLY the corrected JSON object that matches the schema. "
                "No markdown, no prose."
            ),
        },
    ]
    raw2 = generate_chat(repair_messages, model_id, adapter_path)
    logger.info("Raw LLM repair answer: %s", raw2)

    parsed2, _ = _parse_payload(raw2)
    if parsed2:
        return parsed2

    raise RecommendationError(t(RECOMMENDATION_FAILED, lang))
