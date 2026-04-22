from __future__ import annotations

from uuid import uuid4

from finbot.schemas import ChatState, LanguageCode


class InMemorySessionStore:
    """
    Minimal in-memory session store for MVP.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, object]] = {}

    def get_or_create(self, session_id: str | None, language: LanguageCode) -> tuple[str, dict[str, object]]:
        if session_id and session_id in self._sessions:
            return session_id, self._sessions[session_id]

        new_id = str(uuid4())
        session = {
            "state": ChatState.ASKING.value,
            "task_mode": None,
            "collected": {},
            "unknown_fields": [],
            "next_item": "TASK_MODE",
            "ready_for_recommendation": False,
            "language": language.value,
        }
        self._sessions[new_id] = session
        return new_id, session

    def save(self, session_id: str, session: dict[str, object]) -> None:
        self._sessions[session_id] = session

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)