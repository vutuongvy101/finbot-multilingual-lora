from __future__ import annotations

import os
from functools import lru_cache
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer


@lru_cache(maxsize=3)
def _get_generator(model_id: str):
    hf_token = os.getenv("HF_TOKEN") or None
    _cache_dir = os.getenv("HF_CACHE_DIR") or None
    cache_dir = os.path.expanduser(_cache_dir) if _cache_dir else None
    local_files_only = os.getenv("HF_LOCAL_FILES_ONLY", "1").strip().lower() in {"1", "true", "yes", "on"}

    load_kwargs = {
        "token": hf_token,
        "local_files_only": local_files_only,
        **({"cache_dir": cache_dir} if cache_dir else {}),
    }

    tokenizer = AutoTokenizer.from_pretrained(model_id, **load_kwargs)
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

    # Pass already-loaded objects — local_files_only never touches generate()
    return pipeline("text-generation", model=model, tokenizer=tokenizer)


def preload_model(model_id: str) -> None:
    _get_generator(model_id)


def generate(prompt: str, model_id: str, max_new_tokens: int = 800) -> str:
    gen = _get_generator(model_id)
    out = gen(prompt, max_new_tokens=max_new_tokens, do_sample=False)
    return out[0]["generated_text"]