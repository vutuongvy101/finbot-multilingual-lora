from __future__ import annotations

import json
from finbot.policy_loader import PolicyBundle
from finbot.schemas import LanguageCode, TaskMode


def build_recommendation_prompt(
    *,
    task_mode: TaskMode,
    language: LanguageCode,
    collected: dict[str, str],
    unknown_fields: list[str],
    policies: PolicyBundle,
    rag_context: str = "",
) -> str:
    return f"""
[SYSTEM]
{policies.output_rules}

{policies.refusal_topics}

{policies.pii_rules}

[PROFILE]
Task: {task_mode.value}
Language: {language.value}
Collected: {json.dumps(collected, ensure_ascii=False)}
Unknown fields: {json.dumps(unknown_fields, ensure_ascii=False)}

[CONTEXT]
{rag_context}

[INSTRUCTION]
Return ONLY valid JSON with keys:
profile_summary, recommendation, reasoning, risks_caveats, sources, disclaimer
""".strip()