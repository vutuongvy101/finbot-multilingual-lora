from __future__ import annotations

import os
from functools import lru_cache

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


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

    return pipeline("text-generation", model=model, tokenizer=tokenizer)


def preload_model(model_id: str) -> None:
    _get_generator(model_id)


def generate_chat(messages: list[dict], model_id: str, max_new_tokens: int = 1024) -> str:
    gen = _get_generator(model_id)
    tokenizer = gen.tokenizer
    model = gen.model

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    out_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.05,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    new_tokens = out_ids[0, input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)
