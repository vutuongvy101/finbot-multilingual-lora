from __future__ import annotations

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from finbot.session_store import InMemorySessionStore
from finbot.schemas import ChatTurnRequest, ChatTurnResponse, LanguageCode
from finbot.state_machine import handle_turn
from finbot.policy_loader import load_policies

app = FastAPI(title="Financial Chatbot API", version="0.1.0")

store = InMemorySessionStore()
app.state.policies = load_policies()

origins_env = os.getenv("FRONTEND_ORIGINS", "")
allow_origins = (
    [o.strip() for o in origins_env.split(",") if o.strip()]
    if origins_env.strip()
    else [
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "policies_loaded": str(bool(app.state.policies.output_rules)).lower(),
    }


@app.post("/chat/turn", response_model=ChatTurnResponse)
def chat_turn(payload: ChatTurnRequest) -> ChatTurnResponse:
    # choose a language fallback only for new session creation
    seed_language = (
        LanguageCode(payload.language_hint)
        if payload.language_hint is not None
        else LanguageCode.EN
    )
    session_id, session = store.get_or_create(
        session_id=payload.session_id, 
        language = seed_language
    )
    result = handle_turn(payload, session)
    store.save(session_id, result.session)
    return ChatTurnResponse(
        session_id=session_id,
        state=result.state,
        task_mode=result.task_mode,
        detected_language=result.detected_language,
        assistant_message=result.assistant_message,
        next_item=result.next_item,
        collected=result.session.get("collected", {}),
        unknown_fields=result.session.get("unknown_fields", []),
        ready_for_recommendation=result.ready_for_recommendation,
        recommendation=None,
        meta=result.meta,
    )