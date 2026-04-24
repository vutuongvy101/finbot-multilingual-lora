from __future__ import annotations

import re

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\-\s()]{7,}\d)(?!\d)")
PROMPT_INJECTION_RE = re.compile(
    r"(?i)\b("
    r"ignore\s+(all\s+)?(previous|prior|system|developer)\s+instructions?|"
    r"reveal\s+(the\s+)?(system|developer)\s+prompt|"
    r"show\s+(me\s+)?(your\s+)?(hidden|internal)\s+instructions?|"
    r"jailbreak|"
    r"do\s+anything\s+now|"
    r"act\s+as\s+(a\s+)?(system|developer)|"
    r"tool\s+call|"
    r"role:\s*(system|developer)|"
    r"bỏ\s+qua\s+(mọi\s+)?hướng\s+dẫn\s+(trước|trước\s+đó)|"
    r"tiết\s+lộ\s+(prompt|chỉ\s+dẫn)\s+hệ\s+thống|"
    r"cho\s+tôi\s+xem\s+(prompt|chỉ\s+dẫn)\s+ẩn|"
    r"忽略(所有)?(之前|先前|系统|开发者)(指令|说明)|"
    r"显示(系统|开发者)(提示词|指令)|"
    r"泄露(系统|开发者)(提示词|指令)"
    r")\b"
)


def redact_pii(text: str) -> str:
    """Redact common PII patterns in free-text user content."""
    redacted = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    redacted = PHONE_RE.sub("[REDACTED_PHONE]", redacted)
    return redacted


def detect_prompt_injection(text: str) -> bool:
    """Return True when text contains common prompt injection patterns."""
    return bool(PROMPT_INJECTION_RE.search(text))


def sanitize_untrusted_text(text: str) -> str:
    """
    Sanitize untrusted text before placing it in LLM prompt context.

    This keeps user intent while neutralizing obvious instruction-override payloads.
    """
    cleaned = redact_pii(text)
    if not detect_prompt_injection(cleaned):
        return cleaned
    return PROMPT_INJECTION_RE.sub("[BLOCKED_INSTRUCTION]", cleaned)
