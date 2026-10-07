"""Assemble probe design matrices and targets from cached features.

Groups the dataset configurations of one paper task (visual variants merged), builds
X = [view1 features ; view2 features ; covariates] and the task target, and
tags each row with its `sample_id` group for leakage-safe CV.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

import numpy as np

logger = logging.getLogger(__name__)

VIEW_COLS = ("query_image_1", "query_image_2")


@dataclass(frozen=True)
class TaskSpec:
    """One probe task: which configs feed it and how to read its target."""

    name: str
    configs: Sequence[str]
    kind: str  # "regression" | "binary" | "multioutput" | "multilabel"
    target_field: str
    covariate_fields: Sequence[str] = field(default_factory=tuple)
    role: str = ""


TASK_SPECS: List[TaskSpec] = [
    TaskSpec(
        # Paper conditions only: default (uncolor) + single-factor add-ons.
        "T1_volume",
        ["T1_uncolor", "T1_color",
         "T1_uncolor_axes"],
        "regression", "answer", role="VLM-failing",
    ),
    TaskSpec(
        "T2_packaging",
        ["T2_uncolor", "T2_uncolor_axes"],
        "binary", "answer", covariate_fields=("box_dims",), role="VLM-solvable",
    ),
    TaskSpec(
        "T4_color_count",
        ["T4_color", "T4_color_axes"],
        "regression", "answer", role="control",
    ),
    TaskSpec(
        "T5_color_distribution",
        ["T5_color", "T5_color_axes"],
        "multioutput", "color_distribution_json", role="headline failure",
    ),
    TaskSpec(
        "T6_local_structure",
        ["T6_uncolor", "T6_uncolor_axes"],
        "binary", "answer", covariate_fields=("k",), role="VLM-solvable",
    ),
    TaskSpec(
        "T7_color_dominance",
        ["T7_color", "T7_color_axes"],
        "multilabel", "answer", role="positive control",
    ),
    TaskSpec(
        "T8_symmetry",
        ["T8_uncolor", "T8_uncolor_axes"],
        "binary", "answer", role="VLM-near-random",
    ),
    TaskSpec(
        "T9_visible_floors",
        ["T9_uncolor", "T9_uncolor_axes"],
        "regression", "answer", role="VLM-failing",
    ),
    TaskSpec(
        "T10_color_connected",
        ["T10_color", "T10_color_axes"],
        "binary", "answer", role="thinking-responsive",
    ),
]


@dataclass(frozen=True)
class ProbeData:
    """Design matrix + target(s) + CV groups for one task."""

    task: str
    kind: str
    X: np.ndarray
    y: np.ndarray  # (n,) or (n, k) for multioutput/multilabel
    groups: np.ndarray  # sample_id per row
    label_names: Sequence[str] = field(default_factory=tuple)


def _load_config(cache_dir: Path, encoder: str, config: str) -> tuple[dict, list]:
    npz = np.load(cache_dir / encoder / f"{config}.npz")
    targets = json.loads((cache_dir / encoder / f"{config}.targets.json").read_text())
    return dict(npz), targets


def _covariates(rows: List[dict], fields: Sequence[str]) -> np.ndarray:
    cols: List[np.ndarray] = []
    for f in fields:
        vals = [row[f] for row in rows]
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr.reshape(len(rows), -1)
        std = arr.std(axis=0)
        std[std == 0] = 1.0
        cols.append((arr - arr.mean(axis=0)) / std)
    return np.concatenate(cols, axis=1) if cols else np.zeros((len(rows), 0))


def _color_vocab(rows: List[dict], field_name: str, kind: str) -> List[str]:
    vocab: set[str] = set()
    for row in rows:
        raw = row[field_name]
        if kind == "multioutput":
            vocab.update(json.loads(raw).keys())
        else:  # multilabel: answer is already a list of color names
            vocab.update(raw)
    return sorted(vocab)


T3_CONFIGS = ["T3_color_axes", "T3_uncolor_axes"]
T3_VIEW_COLS = ("query_a_image_1", "query_a_image_2",
                "query_b_image_1", "query_b_image_2")
T3_TARGET_FIELDS = ("answer_intersection", "answer_only_a", "answer_only_b")


def build_t3_probe_data(cache_dir: Path, encoder: str, rep: str = "pooled") -> ProbeData:
    """T3 (Difference) probe: 4 view embeddings -> (intersection, only_a, only_b).

    Objects are heavily reused across pairs (the pair graph collapses into a
    few connected components), so leakage-safe CV keeps only an object-disjoint
    subset of pairs: greedily select pairs so that no underlying object appears
    twice, then group by pair_id (both visual variants of a pair stay together).
    """
    feats: List[np.ndarray] = []
    rows: List[dict] = []
    for config in T3_CONFIGS:
        arrays, targets = _load_config(cache_dir, encoder, config)
        views = [arrays[f"{rep}__{col}"] for col in T3_VIEW_COLS]
        feats.append(np.concatenate(views, axis=1))
        rows.extend(targets)
    X_all = np.concatenate(feats, axis=0)

    seen: set[str] = set()
    kept_pairs: set[str] = set()
    for row in sorted(rows, key=lambda r: r["pair_id"]):
        if row["pair_id"] in kept_pairs:
            continue
        a, b = row["sample_a_id"], row["sample_b_id"]
        if a not in seen and b not in seen and a != b:
            seen.update((a, b))
            kept_pairs.add(row["pair_id"])
    keep_idx = [i for i, row in enumerate(rows) if row["pair_id"] in kept_pairs]

    X = X_all[keep_idx]
    kept_rows = [rows[i] for i in keep_idx]
    y = np.asarray([[float(row[f]) for f in T3_TARGET_FIELDS] for row in kept_rows])
    groups = np.asarray([row["pair_id"] for row in kept_rows])
    logger.info("T3_difference: kept %d/%d object-disjoint pairs -> X=%s y=%s",
                len(kept_pairs), len({r['pair_id'] for r in rows}), X.shape, y.shape)
    return ProbeData("T3_difference", "multioutput", X, y, groups,
                     label_names=T3_TARGET_FIELDS)


def build_probe_data(
    spec: TaskSpec, cache_dir: Path, encoder: str, rep: str = "pooled"
) -> ProbeData:
    """Merge a task's configs into one leakage-safe probe dataset."""
    feats: List[np.ndarray] = []
    rows: List[dict] = []
    for config in spec.configs:
        arrays, targets = _load_config(cache_dir, encoder, config)
        views = [arrays[f"{rep}__{col}"] for col in VIEW_COLS]
        feats.append(np.concatenate(views, axis=1))
        rows.extend(targets)
    X = np.concatenate(feats, axis=0)
    X = np.concatenate([X, _covariates(rows, spec.covariate_fields)], axis=1)
    groups = np.asarray([row["sample_id"] for row in rows])

    label_names: Sequence[str] = ()
    if spec.kind == "regression":
        y = np.asarray([float(row[spec.target_field]) for row in rows])
    elif spec.kind == "binary":
        y = np.asarray([int(bool(row[spec.target_field])) for row in rows])
    elif spec.kind == "multioutput":
        label_names = _color_vocab(rows, spec.target_field, spec.kind)
        y = np.zeros((len(rows), len(label_names)))
        for i, row in enumerate(rows):
            for color, count in json.loads(row[spec.target_field]).items():
                y[i, label_names.index(color)] = float(count)
    elif spec.kind == "multilabel":
        label_names = _color_vocab(rows, spec.target_field, spec.kind)
        y = np.zeros((len(rows), len(label_names)), dtype=int)
        for i, row in enumerate(rows):
            for color in row[spec.target_field]:
                y[i, label_names.index(color)] = 1
    else:
        raise ValueError(f"Unknown task kind: {spec.kind}")

    logger.info("%s: X=%s y=%s groups=%d unique", spec.name, X.shape, y.shape,
                len(np.unique(groups)))
    return ProbeData(spec.name, spec.kind, X, y, groups, label_names)
