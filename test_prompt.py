from __future__ import annotations

import argparse
import json

from finbot.llm_adapter import preload_model
from finbot.policy_loader import load_policies
# from finbot.prompt_builder import build_recommendation_prompt
from finbot.recommender import llm_raw_answer
from finbot.schemas import LanguageCode, TaskMode


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal prompt + model output test")
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Hugging Face model id to load and run",
    )
    args = parser.parse_args()

    policies = load_policies("src/policies")
    collected = {
        "GOAL": "Build long-term retirement savings",
        "INCOME_BAND": "5000-7000 USD/month",
        "CAPITAL_RANGE": "10000-20000 USD",
        "TIME_HORIZON": "10-20 years",
        "RISK_TOLERANCE": "Moderate",
    }
    unknown_fields: list[str] = []
    
    task_mode = TaskMode.PLANNING
    language = LanguageCode.EN
    context_block = "(no external context retrieved)"
    

    prompt = f"""
[SYSTEM]
You are an expert Financial Strategist. Your goal is to provide high-density, mathematical, and specific guidance based on USER PROFILE and FINANCIAL CONTEXT (if provided).

You MUST follow the ANALYSIS PROTOCOL, OUTPUT RULES, REFUSAL TOPICS, PII RULES, and STRICT RULES exactly.


# =========================
# USER INPUT
# =========================

[USER'S PROFILE]
Task: {task_mode.value}
Language: {language.value}
Collected: {json.dumps(collected, ensure_ascii=False)}
Unknown fields: {json.dumps(unknown_fields, ensure_ascii=False)}

[CONTEXT]
{context_block}


# =========================
# ANALYSIS PROTOCOL
# =========================

1. Quantitative Extraction:
   - Identify every number in CONTEXT
   - If missing, assume conservative 5th-percentile industry value
   - Label assumptions as "Assumed X"

2. Stress Test:
   - Run Bear Case scenario (-20% market shift)
   - Recalculate impact on outcome

3. Delta Computation:
   - Compute Δ = Target Value - Current Value
   - If target is missing, infer and state assumption


# =========================
# OUTPUT RULES
# =========================

- Keep guidance educational and risk-aware.
- Include risks and caveats.
- Use clear structure and concise language.
- Add disclaimer that this is not professional financial advice.


# =========================
# REFUSAL TOPICS
# =========================

You MUST refuse or safely redirect for:
- Illegal financial activity
- Fraud, money laundering, tax evasion
- Guaranteed profit claims
- Personalized legal/tax advice beyond scope


# =========================
# PII RULES
# =========================

- Do not request exact identity data.
- Do not store raw phone/email/address when not necessary.
- Prefer bucketed profile fields (enum/range) over exact values.
- If user provides sensitive details, avoid repeating them verbatim.


# =========================
# FINANCIAL INTELLIGENCE RULES
# =========================

- Every recommendation MUST include:
  - at least 1 explicit formula
  - at least 1 numeric substitution
  - at least 1 financial metric (CAGR, risk %, alpha, ratio)
- All actions must be tied to USER PROFILE
- No generic financial advice allowed


# =========================
# TASK LOGIC
# =========================

IF planning:
- Capital allocation framework required
- Savings rate + distribution required

IF investment:
- Portfolio must include ≥3 asset classes
- Must include expected return assumption

IF trading:
- Must include entry, stop-loss %, position sizing formula

# =========================
# NUMERIC TRANSFORMATION RULE (CRITICAL):
# =========================

1. You MUST convert ALL user profile fields into at least one computed financial output.
Example transformations:
- Income → monthly savings capacity
- Capital → portfolio allocation value
- Time horizon → required CAGR
- Risk tolerance → max drawdown constraint
2.You are NOT allowed to describe values without transforming them into a calculation.
3 You MUST choose ONE dominant strateg in requested task mode (planning, investment, trading) AND justify it using a numeric constraint (not just text reasoning).


# =========================
# OUTPUT FORMAT (STRICT JSON)
# =========================

Return ONLY:

{{
  "profile_summary": "...",

  "recommendation": "...",

  "reasoning": "...",

  "risks_caveats": "...",

  "sources": ["..."],

  "disclaimer": "..."
}}


# =========================
# STRICT RULES
# =========================

- Zero vague advice (no “consider”, “might”, “suggest”)
- Use execution language: "Allocate", "Execute", "Set"
- All assumptions must be explicitly labeled
- Do NOT use raw field names in output (GOAL, CAPITAL_RANGE, etc.)
- Must include formula density + numeric reasoning


# =========================
# TONE & STYLE
# =========================

Professional, analytical, and objective.
No motivational language.
No generic financial advice.


# =========================
# FINAL RULE
# =========================

Output ONLY valid JSON.
No markdown.
No explanation outside JSON.

>>> Generate the response now.
""".strip()

    print("=== Prompt Preview ===")
    print(prompt + ("..." if len(prompt) > 800 else ""))
    print()

    print(f"=== Loading model: {args.model_id} ===")
    preload_model(args.model_id)
    print("Model loaded.")
    print()

    print("=== Generating recommendation payload ===")
    result = llm_raw_answer(
        prompt=prompt,
        model_id=args.model_id
    )
    print(result)


if __name__ == "__main__":
    main()
