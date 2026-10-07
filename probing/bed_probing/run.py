"""Run all probing analyses over cached features and write results + summary.

Usage:
    python -m bed_probing.run --encoders siglip2 dinov2 clip \
        --cache-dir features --out-dir results
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

from .consistency import run_consistency
from .probes import run_probe
from .targets import TASK_SPECS, build_probe_data, build_t3_probe_data

logger = logging.getLogger(__name__)

KEY_METRIC = {
    "regression": "spearman",
    "binary": "auc",
    "multioutput": "per_color_mae",
    "multilabel": "mean_auc",
}


def run_encoder(encoder: str, cache_dir: Path, rep: str, probe: str,
                n_permutations: int) -> dict:
    """All E1 + E2 analyses for one encoder."""
    results: Dict[str, object] = {"encoder": encoder, "rep": rep, "probe": probe}
    results["consistency"] = run_consistency(cache_dir, encoder, reps=[rep])
    probe_results = []
    for spec in TASK_SPECS:
        data = build_probe_data(spec, cache_dir, encoder, rep=rep)
        probe_results.append(run_probe(data, probe=probe,
                                       n_permutations=n_permutations).to_dict())
    t3_data = build_t3_probe_data(cache_dir, encoder, rep=rep)
    probe_results.append(run_probe(t3_data, probe=probe,
                                   n_permutations=n_permutations).to_dict())
    results["probes"] = probe_results
    return results


def summarize(all_results: List[dict], out_path: Path) -> None:
    """Cross-encoder markdown summary table."""
    lines = ["# Probing summary", ""]
    lines.append("## E2: probe key metrics (vs permutation null p95)")
    lines.append("")
    header = "| Task | kind | " + " | ".join(r["encoder"] for r in all_results) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (2 + len(all_results)))
    for i, probe in enumerate(all_results[0]["probes"]):
        key = KEY_METRIC[probe["kind"]]
        cells = []
        for r in all_results:
            p = r["probes"][i]
            cells.append(f"{p['metrics'][key]:.3f} (null {p['null_p95'][key]:.3f})")
        lines.append(f"| {probe['task']} | {probe['kind']}:{key} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## E1a: cross-view retrieval top-1 (random = 1/n)")
    lines.append("")
    lines.append("| Config | " + " | ".join(r["encoder"] for r in all_results) + " |")
    lines.append("|" + "---|" * (1 + len(all_results)))
    n_retr = len(all_results[0]["consistency"]["retrieval"])
    for i in range(n_retr):
        cfg = all_results[0]["consistency"]["retrieval"][i]["config"]
        cells = [f"{r['consistency']['retrieval'][i]['top1']:.3f}" for r in all_results]
        lines.append(f"| {cfg} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## E1c: zero-shot candidate ranking top-1 (T11/T12/T13, default condition)")
    lines.append("")
    lines.append("| Task | chance | " + " | ".join(r["encoder"] for r in all_results) + " |")
    lines.append("|" + "---|" * (2 + len(all_results)))
    n_rank = len(all_results[0]["consistency"].get("candidate_ranking", []))
    for i in range(n_rank):
        row0 = all_results[0]["consistency"]["candidate_ranking"][i]
        cells = [f"{r['consistency']['candidate_ranking'][i]['top1']:.3f}"
                 for r in all_results]
        lines.append(f"| {row0['task']} | {row0['chance']:.2f} | "
                     + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## E1b: Spearman(embedding distance, structural difference)")
    lines.append("")
    lines.append("| Config | " + " | ".join(r["encoder"] for r in all_results) + " |")
    lines.append("|" + "---|" * (1 + len(all_results)))
    n_dist = len(all_results[0]["consistency"]["distance_vs_difference"])
    for i in range(n_dist):
        cfg = all_results[0]["consistency"]["distance_vs_difference"][i]["config"]
        cells = [
            f"{r['consistency']['distance_vs_difference'][i]['spearman']:.3f}"
            for r in all_results
        ]
        lines.append(f"| {cfg} | " + " | ".join(cells) + " |")
    out_path.write_text("\n".join(lines) + "\n")
    logger.info("Wrote %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoders", nargs="+", default=["siglip2"])
    parser.add_argument("--cache-dir", type=Path, default=Path("features"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--rep", default="pooled", choices=["pooled", "patch_mean"])
    parser.add_argument("--probe", default="linear", choices=["linear", "mlp"])
    parser.add_argument("--n-permutations", type=int, default=1000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[dict] = []
    for encoder in args.encoders:
        results = run_encoder(encoder, args.cache_dir, args.rep, args.probe,
                              args.n_permutations)
        out_json = args.out_dir / f"{encoder}_{args.rep}_{args.probe}.json"
        out_json.write_text(json.dumps(results, indent=2))
        logger.info("Wrote %s", out_json)
        all_results.append(results)
    summarize(all_results, args.out_dir / f"summary_{args.rep}_{args.probe}.md")


if __name__ == "__main__":
    main()
