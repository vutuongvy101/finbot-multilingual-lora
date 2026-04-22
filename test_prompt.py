from __future__ import annotations

import argparse
import json

from finbot.llm_adapter import preload_model
from finbot.policy_loader import load_policies
from finbot.prompt_builder import build_recommendation_prompt
from finbot.recommender import generate_recommendation
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

    prompt = build_recommendation_prompt(
        task_mode=TaskMode.PLANNING,
        language=LanguageCode.EN,
        collected=collected,
        unknown_fields=unknown_fields,
        policies=policies,
    )

    print("=== Prompt Preview ===")
    print(prompt + ("..." if len(prompt) > 800 else ""))
    print()

    print(f"=== Loading model: {args.model_id} ===")
    preload_model(args.model_id)
    print("Model loaded.")
    print()

    print("=== Generating recommendation payload ===")
    result = generate_recommendation(
        prompt=prompt,
        model_id=args.model_id,
        collected=collected,
        unknown_fields=unknown_fields,
    )
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
