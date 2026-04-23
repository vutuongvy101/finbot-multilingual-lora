from __future__ import annotations

import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from finbot.session_store import InMemorySessionStore
from finbot.schemas import (
    ChatTurnRequest,
    ChatTurnResponse,
    LanguageCode,
    ChatState,
    TaskMode,
    ResponseMeta,
    ModelLoadRequest,
    ModelLoadResponse,
    RecommendationError,
)
from finbot.state_machine import handle_turn
from finbot.policy_loader import load_policies
from finbot.prompt_builder import build_recommendation_messages
from finbot.recommender import generate_recommendation
from finbot.llm_adapter import preload_model
from finbot.safety import redact_pii

import logging
import os

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
adapter_path = os.getenv("FINBOT_ADAPTER_PATH")
base_model = os.getenv("FINBOT_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")


logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

app = FastAPI(title="Financial Chatbot API", version="0.1.0")


@app.exception_handler(RecommendationError)
def recommendation_error_handler(request: Request, exc: RecommendationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"error": {"code": "RECOMMENDATION_FAILED", "message": exc.message, "details": {}}},
    )

store = InMemorySessionStore()
app.state.policies = load_policies("src/policies")

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


@app.post("/model/load", response_model=ModelLoadResponse)
def model_load(payload: ModelLoadRequest) -> ModelLoadResponse:
    preload_model(payload.model_id, adapter_path)
    return ModelLoadResponse(status="ok", model_id=payload.model_id)


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
    
    recommendation = None

    if result.state == ChatState.RECOMMENDING and result.ready_for_recommendation and result.task_mode is not None:
        collected = result.session.get("collected", {})
        prompt_collected = dict(collected)
        if "GOAL" in prompt_collected:
            prompt_collected["GOAL"] = redact_pii(str(prompt_collected["GOAL"]))
        recommendation = generate_recommendation(
            messages=build_recommendation_messages(
                task_mode=TaskMode(result.task_mode),
                lang_code=LanguageCode(result.detected_language),
                collected=prompt_collected,
                unknown_fields=result.session.get("unknown_fields", []),
            ),
            model_id=payload.model_id,
            lang=result.detected_language,
            adapter_path=adapter_path,
        )
        
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
        recommendation=recommendation,
        meta=ResponseMeta(
            used_rag=False,  # TODO: no RAG yet
            model_id=result.meta.model_id,
            latency_ms=result.meta.latency_ms,
        ),
    )