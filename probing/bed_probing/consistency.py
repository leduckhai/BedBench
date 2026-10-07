"""View-consistency analyses (Experiment E1): no training required.

E1a  Cross-view retrieval: given view 1 of an object, is view 2 its nearest
     neighbor among all view-2 embeddings? Reports top-1 accuracy and MRR
     against the 1/n random baseline.
E1b  Distance vs. structure (T3): does embedding distance between two objects
     track their ground-truth block difference (answer_only_a + answer_only_b)?
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

# Paper conditions for T1 only (the joint color+axes render is not
# analyzed in the paper and is therefore excluded here as well).
RETRIEVAL_CONFIGS: List[str] = [
    "T1_uncolor",
    "T1_color",
    "T1_uncolor_axes",
]

PAIR_CONFIGS: List[str] = [
    "T3_color_axes",
    "T3_uncolor_axes",
]

T3_VIEW_COLS = {
    "a": ("query_a_image_1", "query_a_image_2"),
    "b": ("query_b_image_1", "query_b_image_2"),
}


@dataclass(frozen=True)
class RetrievalResult:
    config: str
    rep: str
    n: int
    top1: float
    mrr: float
    random_top1: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DistanceResult:
    config: str
    rep: str
    n_pairs: int
    spearman: float
    p_value: float

    def to_dict(self) -> dict:
        return asdict(self)


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def _load(cache_dir: Path, encoder: str, config: str) -> dict:
    return dict(np.load(cache_dir / encoder / f"{config}.npz"))


def cross_view_retrieval(cache_dir: Path, encoder: str, config: str,
                         rep: str = "pooled") -> RetrievalResult:
    """Match view-1 embeddings to view-2 embeddings of the same objects."""
    arrays = _load(cache_dir, encoder, config)
    v1 = _l2(arrays[f"{rep}__query_image_1"])
    v2 = _l2(arrays[f"{rep}__query_image_2"])
    sim = v1 @ v2.T  # (n, n) cosine similarity
    n = sim.shape[0]
    # Rank of the true match (diagonal) within each row, 1-based.
    ranks = (sim > np.diag(sim)[:, None]).sum(axis=1) + 1
    result = RetrievalResult(
        config=config, rep=rep, n=n,
        top1=float((ranks == 1).mean()),
        mrr=float((1.0 / ranks).mean()),
        random_top1=1.0 / n,
    )
    logger.info("%s [%s]: top1=%.3f mrr=%.3f (random %.3f)",
                config, rep, result.top1, result.mrr, result.random_top1)
    return result


def distance_vs_difference(cache_dir: Path, encoder: str, config: str,
                           rep: str = "pooled") -> DistanceResult:
    """Correlate object-pair embedding distance with ground-truth block difference."""
    arrays = _load(cache_dir, encoder, config)
    targets = json.loads((cache_dir / encoder / f"{config}.targets.json").read_text())

    emb: Dict[str, np.ndarray] = {}
    for obj, (c1, c2) in T3_VIEW_COLS.items():
        emb[obj] = _l2((arrays[f"{rep}__{c1}"] + arrays[f"{rep}__{c2}"]) / 2.0)
    cosine_dist = 1.0 - (emb["a"] * emb["b"]).sum(axis=1)
    struct_diff = np.asarray(
        [row["answer_only_a"] + row["answer_only_b"] for row in targets], dtype=np.float64
    )
    rho = spearmanr(cosine_dist, struct_diff)
    result = DistanceResult(
        config=config, rep=rep, n_pairs=len(struct_diff),
        spearman=float(rho.statistic), p_value=float(rho.pvalue),
    )
    logger.info("%s [%s]: spearman(dist, diff)=%.3f (p=%.3g)",
                config, rep, result.spearman, result.p_value)
    return result


# E1c: zero-shot candidate ranking for the candidate-selection tasks.
# `rule` encodes the task semantics: the answer indexes the odd/mutated
# candidate ("argmin" similarity to the reference) or the matching one
# ("argmax"). Default paper condition (Uncolored) only.
CANDIDATE_TASKS: Dict[str, dict] = {
    "T11_invalid": {
        "config": "T11_invalid_uncolor", "rule": "argmin",
        "ref_cols": ("query_image_1", "query_image_2"),
        "cand_cols": ("candidate_0", "candidate_1", "candidate_2"),
        "answer_field": "answer",
    },
    "T11_valid": {
        "config": "T11_valid_uncolor", "rule": "argmax",
        "ref_cols": ("query_image_1", "query_image_2"),
        "cand_cols": ("candidate_0", "candidate_1", "candidate_2"),
        "answer_field": "answer",
    },
    "T12": {
        "config": "T12_uncolor", "rule": "argmax",
        "ref_cols": ("silhouette_0", "silhouette_1", "silhouette_2"),
        "cand_cols": ("candidate_0", "candidate_1"),
        "answer_field": "answer_idx",
    },
    "T13_invalid": {
        "config": "T13_invalid_uncolor", "rule": "argmin",
        "ref_cols": ("query_image_1", "query_image_2"),
        "cand_cols": ("candidate_0", "candidate_1", "candidate_2"),
        "answer_field": "answer",
    },
    "T13_valid": {
        "config": "T13_valid_uncolor", "rule": "argmax",
        "ref_cols": ("query_image_1", "query_image_2"),
        "cand_cols": ("candidate_0", "candidate_1", "candidate_2"),
        "answer_field": "answer",
    },
}


@dataclass(frozen=True)
class RankingResult:
    task: str
    config: str
    rep: str
    rule: str
    n: int
    top1: float
    top1_inverted: float  # accuracy under the opposite rule (semantics check)
    chance: float

    def to_dict(self) -> dict:
        return asdict(self)


def candidate_ranking(cache_dir: Path, encoder: str, task: str,
                      spec: dict, rep: str = "pooled") -> RankingResult:
    """Zero-shot: does embedding similarity to the reference pick the answer?"""
    arrays = _load(cache_dir, encoder, spec["config"])
    targets = json.loads(
        (cache_dir / encoder / f"{spec['config']}.targets.json").read_text())
    ref = _l2(np.mean([arrays[f"{rep}__{c}"] for c in spec["ref_cols"]], axis=0))
    sims = np.stack(
        [(ref * _l2(arrays[f"{rep}__{c}"])).sum(axis=1) for c in spec["cand_cols"]],
        axis=1)  # (n, n_candidates)
    answers = np.asarray([int(row[spec["answer_field"]]) for row in targets])
    pick = sims.argmin(axis=1) if spec["rule"] == "argmin" else sims.argmax(axis=1)
    anti = sims.argmax(axis=1) if spec["rule"] == "argmin" else sims.argmin(axis=1)
    result = RankingResult(
        task=task, config=spec["config"], rep=rep, rule=spec["rule"],
        n=len(answers), top1=float((pick == answers).mean()),
        top1_inverted=float((anti == answers).mean()),
        chance=1.0 / len(spec["cand_cols"]),
    )
    logger.info("%s [%s]: top1=%.3f (inverted %.3f, chance %.3f)",
                task, rep, result.top1, result.top1_inverted, result.chance)
    return result


def run_consistency(cache_dir: Path, encoder: str,
                    reps: List[str] = ["pooled", "patch_mean"]) -> dict:
    """Run E1a + E1b + E1c over all configs and representations."""
    retrieval = [cross_view_retrieval(cache_dir, encoder, cfg, rep).to_dict()
                 for cfg in RETRIEVAL_CONFIGS for rep in reps]
    distance = [distance_vs_difference(cache_dir, encoder, cfg, rep).to_dict()
                for cfg in PAIR_CONFIGS for rep in reps]
    ranking = [candidate_ranking(cache_dir, encoder, task, spec, rep).to_dict()
               for task, spec in CANDIDATE_TASKS.items() for rep in reps]
    return {"retrieval": retrieval, "distance_vs_difference": distance,
            "candidate_ranking": ranking}
