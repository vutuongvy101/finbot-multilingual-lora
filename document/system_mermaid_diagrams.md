# FinancialBot System Mermaid Diagrams

## 1) Overall Component Diagram

```mermaid
flowchart LR
    UI[Frontend / User Interface]
    API[API Layer\n`main.py`]
    DIALOGUE[Dialogue Engine\n`state_machine.py` + `task_policy.py` + `parsers.py` + `i18n.py`]
    SAFETY[Safety Layer\n`safety.py` + policy files]
    PROMPT[Prompt & Policy Layer\n`prompt_builder.py` + `policy_loader.py`]
    LLM[LLM Runtime\n`llm_adapter.py`]
    REC[Recommendation Layer\n`recommender.py`]
    SESSION[Session Store\n`session_store.py`]
    SCHEMA[Data Schemas\n`schemas.py`]

    UI --> API
    API --> DIALOGUE
    API --> SESSION
    API --> REC
    DIALOGUE --> SESSION
    DIALOGUE --> SAFETY
    REC --> PROMPT
    PROMPT --> SAFETY
    REC --> LLM
    REC --> SCHEMA
    API --> SCHEMA
```

## 2) API + Session Boundary

```mermaid
flowchart TD
    REQ[POST /chat/turn]
    API[FastAPI Handler\n`main.py`]
    STORE[InMemorySessionStore\n`session_store.py`]
    TURN[handle_turn(...)\n`state_machine.py`]
    GEN[generate_recommendation(...)\n`recommender.py`]
    RESP[ChatTurnResponse\n`schemas.py`]

    REQ --> API
    API --> STORE
    STORE --> API
    API --> TURN
    TURN --> API
    API --> GEN
    GEN --> API
    API --> STORE
    API --> RESP
```

## 3) Dialogue Engine (State + Collection)

```mermaid
flowchart TD
    IN[User Message + language_hint]
    LANG[Language Detection]
    MODE{Task mode set?}
    PARSE[Parse mode\nPlanning/Investment/Trading]
    ASK[Ask next required field]
    VALIDATE[Validate field answer]
    UNKNOWN[Unknown/clarification logic]
    REDACT[Redact PII in GOAL]
    GATE{Recommendation gate\nready?}
    READY[State = RECOMMENDING]
    LOOP[State = ASKING\nre-ask next field]

    IN --> LANG --> MODE
    MODE -- No --> PARSE --> ASK --> LOOP
    MODE -- Yes --> VALIDATE
    VALIDATE --> UNKNOWN
    VALIDATE --> REDACT
    UNKNOWN --> GATE
    REDACT --> GATE
    GATE -- Yes --> READY
    GATE -- No --> LOOP
```

## 4) Prompt + Safety Assembly

```mermaid
flowchart TD
    COLLECTED[Collected profile fields]
    UNKNOWN[unknown_fields]
    POLICYLOAD[load_policies()\n`policy_loader.py`]
    PII[pii_rules.md]
    REFUSAL[refusal_topics.md]
    OUTPUT[output_rules.md]
    BUILD[build_recommendation_prompt()\n`prompt_builder.py`]
    PROMPT[Final Prompt Text\nSYSTEM + PROFILE + INSTRUCTION + JSON schema]

    PII --> POLICYLOAD
    REFUSAL --> POLICYLOAD
    OUTPUT --> POLICYLOAD
    COLLECTED --> BUILD
    UNKNOWN --> BUILD
    POLICYLOAD --> BUILD
    BUILD --> PROMPT
```

## 5) Recommendation + Validation Pipeline

```mermaid
flowchart TD
    PROMPT[Prompt Input]
    GEN[LLM generate()\n`llm_adapter.py`]
    PARSE[_parse_payload()\n`recommender.py`]
    VALID{Valid RecommendationPayload?}
    RETRY[One repair retry]
    FB[Fallback safe payload]
    NATURALIZE[Profile summary cleanup]
    OUT[RecommendationPayload]

    PROMPT --> GEN --> PARSE --> VALID
    VALID -- Yes --> NATURALIZE --> OUT
    VALID -- No --> RETRY --> GEN
    RETRY --> GEN
    RETRY --> PARSE
    VALID -- No again --> FB --> OUT
```

## 6) Safety Controls (Cross-Cutting)

```mermaid
flowchart LR
    USERINPUT[Raw user text]
    REDACT1[Code-level PII redaction\n`safety.py`]
    DIALOGUE[Dialogue state updates]
    PROMPTPOLICY[Policy text injection\nPII/refusal/output rules]
    LLMOUT[LLM output]
    SCHEMAVAL[Schema validation\n`RecommendationPayload`]
    SAFEOUT[Safe response to frontend]

    USERINPUT --> REDACT1 --> DIALOGUE
    DIALOGUE --> PROMPTPOLICY --> LLMOUT --> SCHEMAVAL --> SAFEOUT
```
