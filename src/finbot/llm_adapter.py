from __future__ import annotations

from functools import lru_cache
from transformers import pipeline


@lru_cache(maxsize=3)
def _get_generator(model_id: str):
    return pipeline("text-generation", model=model_id)


def generate(prompt: str, model_id: str, max_new_tokens: int = 320) -> str:
    gen = _get_generator(model_id)
    out = gen(prompt, max_new_tokens=max_new_tokens, do_sample=False)
    return out[0]["generated_text"]