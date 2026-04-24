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
- **Model**: Fine-tuned Qwen2.5-1.5B-Instruct with LoRA adapter
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
Create `.env` file with your Hugging Face token:
```bash
HF_TOKEN=your_hf_token_here
HF_LOCAL_FILES_ONLY=0
FINBOT_ADAPTER_PATH=artifacts/lora-qwen25-1p5b-finbot-v2
FINBOT_BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct
FINBOT_ADAPTER_MODEL_ID=lora-qwen25-1p5b-finbot-v2
```

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
- **Efficient Inference**: 74% reduction in output length and 35% latency improvement
- **Cross-Platform**: Supports both CUDA and Apple Silicon (MPS) deployment

## Project Structure

```
finbot/
├── src/finbot/           # Core backend modules
│   ├── dialogue/         # State machine and parsers  
│   ├── recommendation/   # LLM adapter and validation
│   └── main.py          # FastAPI application
├── frontend/            # Browser-based UI
├── artifacts/           # LoRA adapter weights
└── notebooks/           # Data processing and evaluation
```

## Performance

Fine-tuning results on held-out evaluation set:
- Schema validity: 0% → 100% 
- Safety compliance: 100% maintained
- Output efficiency: 74% token reduction
- Latency improvement: 35% faster inference

For detailed technical documentation, see the [full report](48706094_Report.pdf).