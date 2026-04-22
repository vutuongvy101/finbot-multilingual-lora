from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class LanguageCode(str, Enum):
    EN = "en"
    VI = "vi"
    ZH = "zh"


class ChatState(str, Enum):
    ASKING = "ASKING"
    RECOMMENDING = "RECOMMENDING"
    REFINING = "REFINING"


class TaskMode(str, Enum):
    PLANNING = "PLANNING"
    INVESTMENT = "INVESTMENT"
    TRADING = "TRADING"


class ErrorPayload(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ErrorPayload


class ChatTurnRequest(BaseModel):
    """
    Request contract for POST /chat/turn.
    """

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    session_id: str | None = Field(
        default=None,
        description="Null on first turn; server creates a new session_id.",
    )
    message: str = Field(
        ...,
        min_length=1,
        description="Raw user message for this turn.",
    )
    model_id: str = Field(
        default="qwen2.5-1.5b-instruct",
        description="Serving model identifier.",
    )
    language_hint: LanguageCode | None = Field(
        default=None,
        description="Optional frontend hint from browser locale mapping.",
    )


class RecommendationPayload(BaseModel):
    """
    Present only when a recommendation is generated.
    """

    model_config = ConfigDict(extra="forbid")

    profile_summary: str
    recommendation: str
    reasoning: str
    risks_caveats: str
    sources: list[str] = Field(default_factory=list)
    disclaimer: str


class ResponseMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    used_rag: bool = False
    model_id: str
    latency_ms: float = Field(ge=0)


class ChatTurnResponse(BaseModel):
    """
    Response contract for POST /chat/turn.
    """

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    session_id: str
    state: ChatState
    task_mode: TaskMode | None = None
    detected_language: LanguageCode
    assistant_message: str
    next_item: str | None = None
    collected: dict[str, str] = Field(default_factory=dict)
    unknown_fields: list[str] = Field(default_factory=list)
    ready_for_recommendation: bool = False
    recommendation: RecommendationPayload | None = None
    meta: ResponseMeta

