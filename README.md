# FinBot: Multilingual Financial Assistant

A smart financial assistant that provides personalized recommendations for planning, investment, and trading through structured multilingual dialogue. Built with a locally deployed fine-tuned LLM and deterministic dialogue engine.

## Key Features

- **Multilingual Support**: Seamless interaction in English, Vietnamese, and Chinese
- **Structured Dialogue**: Deterministic finite-state machine for reliable information collection  
- **Local Deployment**: Runs entirely locally using Qwen2.5-1.5B with LoRA fine-tuning
- **Privacy-First**: Sensitive data is bucketed and processed locally without external API calls
- **Validated Output**: Robust JSON schema validation with automatic repair mechanisms
- **Safety Hardened**: Multilingual prompt-injection defense and PII redaction

## System Architecture

The system uses a hybrid approach: deterministic components for dialogue management and a single LLM call for final recommendation generation. This design ensures reliability while minimizing computational overhead.

- **Backend**: FastAPI with finite-state dialogue engine
- **Frontend**: Clean browser-based UI with Bootstrap 5
- **Model**: Fine-tuned Qwen2.5-1.5B-Instruct with LoRA adapter — [Adapter available on 🤗 Hub](https://huggingface.co/bibbbu/lora-qwen25-1p5b-finbot-v2)
- **Validation**: Cascading JSON validation and repair pipeline

## Quick Start

### Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

### Configuration
Copy the example env file and set your Hugging Face token (required for first-time model download):
```bash
cp .env.example .env
# edit .env — at minimum set HF_TOKEN
```

The `FINBOT_*` variables already default to the Hub adapter and base model; you only need to change them for custom setups. Adapter available on [🤗 Hub](https://huggingface.co/bibbbu/lora-qwen25-1p5b-finbot-v2). Set `FINBOT_ADAPTER_PATH` only for offline/local use.

### Start Services
```bash
# Backend API
python -m uvicorn finbot.main:app --reload --app-dir src

# Frontend (in another terminal)
cd frontend
python -m http.server 5500
```

Visit `http://localhost:5500` to use the financial assistant.

## Technical Highlights

- **Teacher-Student Distillation**: Fine-tuned using GPT-5.3 and Gemini-3-Flash generated examples
- **100% Schema Validity**: Achieved perfect structured output compliance after fine-tuning
- **Efficient Inference**: 77% reduction in output length and 43% latency improvement (A100 bf16 eval, n=62)
- **Cross-Platform**: Supports both CUDA and Apple Silicon (MPS) deployment

## Project Structure

```
finbot/
├── src/
│   ├── finbot/                  # Core backend modules
│   │   ├── main.py              # FastAPI application
│   │   ├── state_machine.py     # Dialogue flow controller
│   │   ├── parsers.py           # Input/slot parsing utilities
│   │   ├── recommender.py       # Recommendation orchestration
│   │   ├── llm_adapter.py       # Local LLM inference adapter
│   │   ├── prompt_builder.py    # Prompt construction helpers
│   │   ├── policy_loader.py     # Policy file loading
│   │   ├── task_policy.py       # Task-level policy checks
│   │   ├── safety.py            # Safety and refusal pipeline
│   │   ├── i18n.py              # Multilingual response support
│   │   ├── session_store.py     # Conversation/session persistence
│   │   └── schemas.py           # Request/response schemas
│   └── policies/                # Runtime policy documents
├── frontend/                    # Browser-based UI
│   ├── index.html               # Chat interface layout
│   └── app.js                   # Frontend chat logic and API calls
├── notebooks/                   # Data prep, training, and evaluation
│   ├── 1_llm_setup__*.ipynb     # Base model loading and setup experiments
│   ├── 2_llm_evaluation_summary.ipynb
│   ├── 3_finetuning_runbook.ipynb
│   ├── 4_prompting_techniques_ablation.ipynb
│   ├── data/                    # Final SFT train/eval JSONL datasets
│   ├── raw_data/                # Domain/language raw samples
│   ├── prompting/               # Prompting ablation CSV outputs
│   ├── eval/                    # Model evaluation JSON reports
│   └── summary/                 # Aggregated metrics and selection notes
├── demo/                        # Demo screenshots by task/language
├── artifacts/                   # Local adapter cache (weights live on 🤗 Hub)
└── document/                    # Policy/prompt references and diagrams
```

## Performance

Fine-tuning results on held-out evaluation set (n=62, NVIDIA A100 bf16):
- Schema validity: 0% → 100% 
- Safety compliance: 100% maintained
- Output efficiency: 77% token reduction (638 → 150 mean tokens)
- Latency improvement: 43% faster inference (5134 → 2902 ms)

For detailed technical documentation, see the [full report](48706094_Report.pdf).