from __future__ import annotations

import os
from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel


def _adapter_available(adapter_source: str) -> bool:
    """True when adapter_source is a Hub repo id or an existing local directory."""
    return "/" in adapter_source or os.path.isdir(adapter_source)


@lru_cache(maxsize=3)
def _get_generator(model_id: str, adapter_source: str | None = None):
    hf_token = os.getenv("HF_TOKEN") or None
    _cache_dir = os.getenv("HF_CACHE_DIR") or None
    cache_dir = os.path.expanduser(_cache_dir) if _cache_dir else None
    local_files_only = os.getenv("HF_LOCAL_FILES_ONLY", "0").strip().lower() in {"1", "true", "yes", "on"}

    # M2 MacBook optimizations
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "mps" else torch.float32

    load_kwargs = {
        "token": hf_token,
        "local_files_only": local_files_only,
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
        **({"cache_dir": cache_dir} if cache_dir else {}),
    }
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, **load_kwargs)


    # Load base model
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs).to(device)
    
    # Load LoRA adapter from Hub repo id or local directory
    if adapter_source and _adapter_available(adapter_source):
        model = PeftModel.from_pretrained(model, adapter_source, torch_dtype=torch_dtype)
        print(f"Using fine-tuned model: {adapter_source}")
    else:
        print(f"Using base model: {model_id}")

    return pipeline("text-generation", model=model, tokenizer=tokenizer)


def preload_model(model_id: str, adapter_source: str = None) -> None:
    _get_generator(model_id, adapter_source)


def generate_chat(messages: list[dict], model_id: str, adapter_source: str = None, max_new_tokens: int = 2048) -> str:
    gen = _get_generator(model_id, adapter_source)
    tokenizer = gen.tokenizer
    model = gen.model

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    is_batch_encoding = hasattr(input_ids, "keys") and "input_ids" in input_ids
    source_input_ids = input_ids["input_ids"] if is_batch_encoding else input_ids

    if is_batch_encoding:
        out_ids = model.generate(
            **input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    else:
        out_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = out_ids[0, source_input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)
