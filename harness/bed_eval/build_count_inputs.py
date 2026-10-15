"""Build self-count and oracle-count T7 requests from released fine-tuned T5/T7 outputs. See README.md for usage."""
import argparse
import json
from pathlib import Path

from .scoring import parse  # noqa: E402

def load(outputs: Path, model: str, task: str, variant: str, run: int) -> list[dict]:
    path = outputs / "non_thinking" / task / model / f"run_{run}" / f"{variant}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def paired(outputs: Path, model: str, variant: str, run: int):
    t5, t7 = (load(outputs, model, task, variant, run) for task in ("T5", "T7"))
    by_id = {str(r["row_id"]): r for r in t7}
    if len(by_id) != len(t7) or len({str(r["row_id"]) for r in t5}) != len(t5):
        raise ValueError("Duplicate object ids")
    if {str(r["row_id"]) for r in t5} != set(by_id):
        raise ValueError(f"T5/T7 object ids do not match: {variant}, run {run}")
    return [(r, by_id[str(r["row_id"])]) for r in t5]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", type=Path, required=True, help="released outputs directory")
    ap.add_argument("--model", default="Qwen3.5-27B-T5-LoRA", help="directory name in released outputs")
    ap.add_argument("--out", type=Path, default=Path("counts_inputs.jsonl"))
    args = ap.parse_args()
    out = []
    for run in range(1, 6):
        for variant in ("color", "color_axes"):
            for r5, r7 in paired(args.outputs, args.model, variant, run):
                counts = parse("T5", r5["content"])
                assert counts is not None, (variant, run, r5["row_id"])  # 0 parse errors in the FT runs
                out.append({"condition": "self_count", "variant": variant, "run": run, "row_id": r5["row_id"],
                            "counts": counts, "t5_target": r5["target"], "target": r7["target"]})
        for r5, r7 in paired(args.outputs, args.model, "color", run):
            gt = {c: n for c, n in r5["target"].items() if n}
            out.append({"condition": "oracle_count", "variant": "gt", "run": run, "row_id": r5["row_id"],
                        "counts": gt, "t5_target": r5["target"], "target": r7["target"]})
    path = args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in out))
    print(f"wrote {len(out)} requests to {path}")


if __name__ == "__main__":
    main()
