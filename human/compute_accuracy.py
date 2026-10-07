"""Per-task human accuracy with 95% Wilson intervals.

Usage: python compute_accuracy.py [human_baseline_responses.csv]
"""
import csv
import math
import statistics
import sys


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (center - half), 100 * (center + half)


def main(path: str = "human_baseline_responses.csv") -> None:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    tasks = [c[:-len("_verdict")] for c in rows[0] if c.endswith("_verdict")]
    print(f"{len(rows)} participants")
    print("task  acc(%)  95% Wilson CI")
    accs = []
    for t in tasks:
        k = sum(r[f"{t}_verdict"] == "correct" for r in rows)
        lo, hi = wilson(k, len(rows))
        accs.append(100 * k / len(rows))
        print(f"{t:4s}  {accs[-1]:5.1f}   [{lo:.1f}, {hi:.1f}]")
    print(f"mean over tasks: {statistics.mean(accs):.1f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
