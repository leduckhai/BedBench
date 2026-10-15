"""Run T5 with no-example and benchmark-matched-example prompts using the released harness. See README.md for usage."""
import argparse
import asyncio
from pathlib import Path


from .__main__ import load_rows
from .client import evaluate
from .tasks import TASKS

PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts" / "non_thinking"
PROMPTS = ("no_example", "valid_example")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-url", default="http://localhost:8000")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="t5_prompt_results")
    ap.add_argument("--prompts", default=",".join(PROMPTS))
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=100)
    args = ap.parse_args()
    if not set(args.prompts.split(",")).issubset(PROMPTS):
        ap.error("--prompts must be comma-separated no_example,valid_example")
    spec, variant = TASKS["T5"], "color"
    rows = load_rows(args.dataset, spec.config(variant), "train")
    assert len(rows) == 100, len(rows)
    model_dir = args.model.replace("/", "_")
    for run in range(1, args.runs + 1):
        for prompt in args.prompts.split(","):
            out = Path(args.out) / "non_thinking" / "T5" / model_dir / f"run_{run}" / f"{variant}_{prompt}.jsonl"
            if out.exists():
                continue
            template = (PROMPT_DIR / f"Task_5_{prompt}.md").read_text()
            asyncio.run(evaluate(spec, rows, template, out, model=args.model, api_url=args.api_url,
                                 variant=variant, thinking=False, concurrency=args.concurrency))
            print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
