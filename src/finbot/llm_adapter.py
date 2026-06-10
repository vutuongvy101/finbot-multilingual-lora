from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from functools import lru_cache

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def _backend() -> str:
    return os.getenv("FINBOT_LLM_BACKEND", "local").strip().lower()


def _adapter_available(adapter_source: str) -> bool:
    """True when adapter_source is a Hub repo id or an existing local directory."""
    return "/" in adapter_source or os.path.isdir(adapter_source)


def _resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        bf16_ok = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        return torch.bfloat16 if bf16_ok else torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def _hf_load_kwargs() -> dict:
    hf_token = os.getenv("HF_TOKEN") or None
    _cache_dir = os.getenv("HF_CACHE_DIR") or None
    cache_dir = os.path.expanduser(_cache_dir) if _cache_dir else None
    local_files_only = os.getenv("HF_LOCAL_FILES_ONLY", "0").strip().lower() in {"1", "true", "yes", "on"}
    return {
        "token": hf_token,
        "local_files_only": local_files_only,
        "trust_remote_code": True,
        **({"cache_dir": cache_dir} if cache_dir else {}),
    }


@lru_cache(maxsize=3)
def _get_generator(model_id: str, adapter_source: str | None = None):
    device = _resolve_device()
    torch_dtype = _resolve_dtype(device)
    load_kwargs = {**_hf_load_kwargs(), "torch_dtype": torch_dtype}

    tokenizer = AutoTokenizer.from_pretrained(model_id, **load_kwargs)
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs).to(device)

    if adapter_source and _adapter_available(adapter_source):
        adapter_kwargs = {"torch_dtype": torch_dtype}
        if "/" in adapter_source:
            adapter_kwargs.update(_hf_load_kwargs())
        model = PeftModel.from_pretrained(model, adapter_source, **adapter_kwargs)
        print(f"Using fine-tuned model: {adapter_source}")
    else:
        print(f"Using base model: {model_id}")

    return pipeline("text-generation", model=model, tokenizer=tokenizer)


def _vllm_model_name(adapter_source: str | None) -> str:
    if adapter_source:
        return os.getenv("FINBOT_VLLM_LORA_MODEL", "finbot")
    return os.getenv("FINBOT_VLLM_MODEL", os.getenv("FINBOT_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"))


def _vllm_base_url() -> str:
    return os.getenv("FINBOT_VLLM_BASE_URL", "http://127.0.0.1:8001/v1").rstrip("/")


def _ollama_base_url() -> str:
    return os.getenv("FINBOT_OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")


def _ollama_model_name() -> str:
    return os.getenv("FINBOT_OLLAMA_MODEL", "finbot")


def _http_json_post(url: str, payload: dict, timeout: float = 300.0) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _vllm_health_check() -> None:
    url = f"{_vllm_base_url()}/models"
    with urllib.request.urlopen(url, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError(f"vLLM health check failed: HTTP {response.status}")


def _ollama_health_check() -> None:
    url = f"{_ollama_base_url()}/api/tags"
    with urllib.request.urlopen(url, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError(f"Ollama health check failed: HTTP {response.status}")


def _generate_local(
    messages: list[dict],
    model_id: str,
    adapter_source: str | None,
    max_new_tokens: int,
) -> str:
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

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "repetition_penalty": 1.05,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if is_batch_encoding:
        out_ids = model.generate(**input_ids, **generate_kwargs)
    else:
        out_ids = model.generate(input_ids, **generate_kwargs)

    new_tokens = out_ids[0, source_input_ids.shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def _generate_vllm(
    messages: list[dict],
    model_id: str,
    adapter_source: str | None,
    max_new_tokens: int,
) -> str:
    payload = {
        "model": _vllm_model_name(adapter_source),
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": 0,
    }
    data = _http_json_post(f"{_vllm_base_url()}/chat/completions", payload)
    return data["choices"][0]["message"]["content"]


def _generate_ollama(messages: list[dict], max_new_tokens: int) -> str:
    payload = {
        "model": _ollama_model_name(),
        "messages": messages,
        "stream": False,
        "options": {"num_predict": max_new_tokens, "temperature": 0},
    }
    data = _http_json_post(f"{_ollama_base_url()}/api/chat", payload)
    return data["message"]["content"]


def preload_model(model_id: str, adapter_source: str | None = None) -> None:
    backend = _backend()
    if backend == "local":
        _get_generator(model_id, adapter_source)
        return
    if backend == "vllm":
        _vllm_health_check()
        return
    if backend == "ollama":
        _ollama_health_check()
        return
    raise ValueError(f"Unsupported FINBOT_LLM_BACKEND: {backend}")


def generate_chat(
    messages: list[dict],
    model_id: str,
    adapter_source: str | None = None,
    max_new_tokens: int = 2048,
) -> str:
    backend = _backend()
    try:
        if backend == "vllm":
            return _generate_vllm(messages, model_id, adapter_source, max_new_tokens)
        if backend == "ollama":
            return _generate_ollama(messages, max_new_tokens)
        return _generate_local(messages, model_id, adapter_source, max_new_tokens)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{backend} backend request failed: {exc}") from exc
