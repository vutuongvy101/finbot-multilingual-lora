from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from finbot.schemas import ChatState, ChatTurnRequest, LanguageCode, ResponseMeta, TaskMode
from finbot.task_policy import next_unfilled_field, recommendation_ready
from finbot.i18n import (
    TASK_PROMPT,
    CLARIFY_SUFFIX,
    TOO_MANY_UNKNOWN_PREFIX,
    t,
    field_question,
    ready_message,
)
from finbot.parsers import parse_task_mode, is_unknown_answer, validate_field_answer
from finbot.safety import redact_pii


@dataclass
class TurnResult:
    session: dict[str, object]
    state: ChatState
    task_mode: str | None
    detected_language: LanguageCode
    assistant_message: str
    next_item: str | None
    ready_for_recommendation: bool
    meta: ResponseMeta


def detect_language(message: str, hint: LanguageCode | None) -> LanguageCode:
    if hint is not None:
        return hint

    vietnamese_chars = set(
        "àáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ"
        "ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ"
    )
    vietnamese_tokens = {"toi", "vui long", "giup", "dau tu", "tai chinh", "ban", "khong", "co"}
    lowered = message.lower()

    if any(ch in vietnamese_chars for ch in message) or any(token in lowered for token in vietnamese_tokens):
        return LanguageCode.VI
    if any("\u4e00" <= ch <= "\u9fff" for ch in message):
        return LanguageCode.ZH
    return LanguageCode.EN

def handle_turn(payload: ChatTurnRequest, session: dict[str, object]) -> TurnResult:
    start = perf_counter()
    lang = detect_language(payload.message, payload.language_hint)
    session["language"] = lang.value

    collected: dict[str, str] = session.setdefault("collected", {})  # type: ignore[assignment]
    clarify_count: dict[str, int] = session.setdefault("clarify_count", {})  # type: ignore[assignment]

    task_mode_raw: str | None = session.get("task_mode")  # type: ignore[assignment]

    # --- Branch 1: task mode not yet selected ---
    if task_mode_raw is None:
        mode = parse_task_mode(payload.message)
        if mode is None:
            return _result(session, start, payload.model_id, lang,
                           ChatState.ASKING, None, t(TASK_PROMPT, lang.value), "TASK_MODE", False)

        session["task_mode"] = mode.value
        # ask the next unfilled (not collected) field
        next_field = next_unfilled_field(mode, collected)
        session["next_item"] = next_field
        return _result(session, start, payload.model_id, lang,
                       ChatState.ASKING, mode.value,
                       field_question(next_field, lang.value) if next_field else "",
                       next_field, False)

    # --- Branch 2: collecting profile fields ---
    mode = TaskMode(task_mode_raw)
    current_field: str | None = session.get("next_item")

    # if no next field, ask the next unfilled (not collected) field
    if current_field is None:
        next_field = next_unfilled_field(mode, collected)
        session["next_item"] = next_field
        return _result(session, start, payload.model_id, lang,
                       ChatState.ASKING, mode.value,
                       field_question(next_field, lang.value) if next_field else "",
                       next_field, False)

    if is_unknown_answer(payload.message):
        count = clarify_count.get(current_field, 0)
        # clarify 2 times, if still unknown, mark as UNKNOWN
        if count == 0:
            clarify_count[current_field] = 1
            msg = field_question(current_field, lang.value) + t(CLARIFY_SUFFIX, lang.value)
            return _result(session, start, payload.model_id, lang,
                           ChatState.ASKING, mode.value, msg, current_field, False)
        collected[current_field] = "UNKNOWN"
        clarify_count.pop(current_field, None)
    else:
        ok, normalized = validate_field_answer(current_field, payload.message)
        # if not valid answer, ask again with the same field
        if not ok:
            return _result(session, start, payload.model_id, lang,
                           ChatState.ASKING, mode.value,
                           field_question(current_field, lang.value), current_field, False)
        if current_field == "GOAL":
            normalized = redact_pii(normalized or "")
        collected[current_field] = normalized  # type: ignore[assignment]
        clarify_count.pop(current_field, None)

    # Advance to next unfilled field
    next_field = next_unfilled_field(mode, collected)
    session["next_item"] = next_field

    if next_field is not None:
        return _result(session, start, payload.model_id, lang,
                       ChatState.ASKING, mode.value,
                       field_question(next_field, lang.value), next_field, False)

    # All fields resolved — check recommendation gate
    ready, unknown_fields = recommendation_ready(mode, collected)
    session["unknown_fields"] = unknown_fields
    session["ready_for_recommendation"] = ready

    if ready:
        return _result(session, start, payload.model_id, lang,
                       ChatState.RECOMMENDING, mode.value,
                       ready_message(lang.value, unknown_fields), None, True)

    # Too many unknowns — re-ask one
    re_ask = unknown_fields[0]
    session["next_item"] = re_ask
    msg = t(TOO_MANY_UNKNOWN_PREFIX, lang.value) + field_question(re_ask, lang.value)
    return _result(session, start, payload.model_id, lang,
                   ChatState.ASKING, mode.value, msg, re_ask, False)


def _result(
    session: dict[str, object],
    start: float,
    model_id: str,
    lang: LanguageCode,
    state: ChatState,
    task_mode: str | None,
    assistant_message: str,
    next_item: str | None,
    ready: bool,
) -> TurnResult:
    return TurnResult(
        session=session,
        state=state,
        task_mode=task_mode,
        detected_language=lang,
        assistant_message=assistant_message,
        next_item=next_item,
        ready_for_recommendation=ready,
        meta=ResponseMeta(
            used_rag=False,
            model_id=model_id,
            latency_ms=(perf_counter() - start) * 1000,
        ),
    )
    