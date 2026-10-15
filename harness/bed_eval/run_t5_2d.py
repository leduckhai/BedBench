"""Run the T5 separated-tile control with the released Non-Thinking payload. See README.md for usage."""
import argparse
import asyncio
import dataclasses
import json
from pathlib import Path

from PIL import Image


from .client import evaluate
from .tasks import TASKS

PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts" / "non_thinking"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-url", default="http://localhost:8000")
    ap.add_argument("--out", default="t5_2d_results")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=100)
    ap.add_argument("--inputs", type=Path, required=True, help="rows.jsonl produced by build_tiles.py")
    args = ap.parse_args()
    spec = dataclasses.replace(TASKS["T5"], images=("tiles",))
    rows = [json.loads(line) for line in args.inputs.read_text().splitlines()]
    for r in rows:
        r["tiles"] = Image.open(args.inputs.parent / r.pop("image")).convert("RGBA")
    assert len(rows) == 100
    template = (PROMPT_DIR / "Task_5_2d.md").read_text()
    for run in range(1, args.runs + 1):
        out = Path(args.out) / "non_thinking" / "T5" / args.model.replace("/", "_") / f"run_{run}" / "color_2d.jsonl"
        if out.exists():
            continue
        asyncio.run(evaluate(spec, rows, template, out, model=args.model, api_url=args.api_url,
                             variant="color_2d", thinking=False, concurrency=args.concurrency))
        print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
