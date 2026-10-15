r"""BedBench evaluation harness.

  python -m bed_eval run   --task T5 --variant color --mode thinking --model Qwen/Qwen3.5-27B \
                           --api-url http://localhost:8000 --dataset DATASET --runs 5 --out results
  python -m bed_eval score results
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from collections import defaultdict
from pathlib import Path

from .client import evaluate
from .scoring import aggregate, score_run
from .tasks import SINGLE_COLORS, TASKS

PROMPTS = Path(__file__).resolve().parents[2] / "prompts"


def load_rows(dataset: str, config: str, split: str) -> list[dict]:
    """A local directory saved with datasets.save_to_disk (DATASET/CONFIG), or a Hugging Face Hub dataset."""
    import datasets
    local = Path(dataset) / config
    ds = datasets.load_from_disk(str(local)) if local.exists() else datasets.load_dataset(dataset, config, split=split)
    return list(ds)


def cmd_run(args) -> None:
    spec = TASKS[args.task]
    variant = args.variant or spec.variants[0]
    if variant not in spec.variants:
        raise SystemExit(f"{args.task} variants: {', '.join(spec.variants)}")
    thinking = args.mode == "thinking"
    rows = load_rows(args.dataset, spec.config(variant), args.split)[: args.limit]
    template = (PROMPTS / args.mode / spec.prompt_file(variant, args.format)).read_text()
    colors = SINGLE_COLORS if args.format == "single_color" else (None,)
    model_dir = args.model.replace("/", "_")
    for run in range(1, args.runs + 1):
        for color in colors:
            name = "_".join(filter(None, [variant, None if args.format == "default" else args.format, color,
                                          None if args.images else "no_image"]))
            out = Path(args.out) / args.mode / args.task / model_dir / f"run_{run}" / f"{name}.jsonl"
            asyncio.run(evaluate(spec, rows, template, out, model=args.model, api_url=args.api_url,
                                 api_key=args.api_key, variant=variant, thinking=thinking, fmt=args.format,
                                 color=color, with_images=args.images,
                                 thinking_budget=None if args.no_thinking_budget else args.thinking_budget,
                                 concurrency=args.concurrency))


def cmd_score(args) -> None:
    """Group result files by everything but the run index, score each run, and report mean ± SD."""
    groups = defaultdict(list)
    for path in sorted(Path(args.results).rglob("*.jsonl")):
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if records and records[0].get("format") != "single_color":
            key = (path.parent.parent.parent.parent.name, *path.parent.parent.parts[-2:], path.stem)
            groups[key].append(score_run(records, mcq=records[0].get("format") == "mcq"))
    summary = {}
    for (mode, task, model, condition), runs in sorted(groups.items()):
        agg = aggregate(runs)
        summary.setdefault(mode, {}).setdefault(model, {}).setdefault(task, {})[condition] = agg
        if "accuracy" not in agg["mean"]:  # some run had no parseable answer
            print(f"{mode:12s} {model:32s} {task:4s} {condition:28s} no parseable answers in at least one run")
            continue
        acc, sd = agg["mean"]["accuracy"], agg["std"]["accuracy"]
        print(f"{mode:12s} {model:32s} {task:4s} {condition:28s} acc {100 * acc:5.1f} ± {100 * sd:4.1f}  ({agg['n_runs']} runs)")
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(prog="bed_eval", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="query a model and save one JSONL file per run")
    run.add_argument("--task", required=True, choices=list(TASKS))
    run.add_argument("--variant", help="visual variant (default: the paper's default setting for the task)")
    run.add_argument("--mode", choices=["non_thinking", "thinking"], required=True)
    run.add_argument("--format", choices=["default", "mcq", "single_color"], default="default",
                     help="mcq: T1 answer-format control; single_color: T5 factorized querying")
    run.add_argument("--no-images", dest="images", action="store_false", help="image-removal control")
    run.add_argument("--model", required=True)
    run.add_argument("--api-url", required=True)
    run.add_argument("--api-key", default="")
    run.add_argument("--dataset", required=True, help="Hugging Face dataset id or local save_to_disk directory")
    run.add_argument("--split", default="train")
    run.add_argument("--runs", type=int, default=5)
    run.add_argument("--thinking-budget", type=int, default=1024)
    run.add_argument("--no-thinking-budget", action="store_true", help="omit the budget field (models without support)")
    run.add_argument("--concurrency", type=int, default=100)
    run.add_argument("--limit", type=int, default=None, help="only the first N questions")
    run.add_argument("--out", default="results")
    run.set_defaults(func=cmd_run)
    score = sub.add_parser("score", help="score saved results")
    score.add_argument("results")
    score.add_argument("--json", help="also write the summary as JSON")
    score.set_defaults(func=cmd_score)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
