"""Projection distractors (Algorithm "Generate Projection Distractors")."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .config import GeneratorConfig
from .geometry import Cell, pick
from .voxels import cell_bounds, has_holes, is_connected, is_thin_bar

_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _valid_edit(out: set[Cell], base: set[Cell], cfg: GeneratorConfig) -> bool:
    return (bool(out) and out != base
            and abs(len(out) - len(base)) <= cfg.neg_max_count_delta
            and is_connected(out) and not has_holes(out))


def perturb(rng: np.random.Generator, cells: set[Cell], cfg: GeneratorConfig) -> tuple[set[Cell], str]:
    """Apply random add/remove/shift edits until the result is a valid, different mask."""
    base = set(cells)
    umin, umax, vmin, vmax = cell_bounds(base, pad=cfg.neg_sampling_pad)
    ops = sorted(cfg.neg_ops)
    for _ in range(cfg.neg_edit_tries):
        out = set(base)
        n_edits = int(rng.integers(cfg.neg_edits[0], cfg.neg_edits[1] + 1))
        for op in [ops[i] for i in rng.integers(len(ops), size=n_edits)]:
            if op == "add":
                out.add((int(rng.integers(umin, umax + 1)), int(rng.integers(vmin, vmax + 1))))
            elif op == "remove":
                if len(out) > 1:
                    out.remove(pick(rng, sorted(out)))
            else:  # shift one cell by one step
                c = pick(rng, sorted(out))
                du, dv = pick(rng, _STEPS)
                out.remove(c)
                out.add((c[0] + du, c[1] + dv))
        if _valid_edit(out, base, cfg):
            return out, "proj_perturb"
    # Deterministic fallbacks (not reached for the released split).
    out = set(base)
    c = sorted(out)[0]
    out.remove(c)
    out.add((c[0] + 1, c[1]))
    if _valid_edit(out, base, cfg):
        return out, "proj_perturb_fallback_shift"
    out = set(base)
    out.add((umin - 3, vmin - 3))
    return out, "proj_perturb_last_resort"


def _bitmap(cells: set[Cell], pad: int) -> np.ndarray:
    umin, umax, vmin, vmax = cell_bounds(cells, pad=0)
    img = np.zeros((vmax - vmin + 1 + 2 * pad, umax - umin + 1 + 2 * pad), dtype=bool)
    for u, v in cells:
        img[v - vmin + pad, u - umin + pad] = True
    return img


def _iou_top_left(a: np.ndarray, b: np.ndarray) -> float:
    h, w = max(a.shape[0], b.shape[0]), max(a.shape[1], b.shape[1])
    aa, bb = np.zeros((h, w), bool), np.zeros((h, w), bool)
    aa[:a.shape[0], :a.shape[1]], bb[:b.shape[0], :b.shape[1]] = a, b
    union = int(np.logical_or(aa, bb).sum())
    return 1.0 if union == 0 else int(np.logical_and(aa, bb).sum()) / union


def similarity(a: set[Cell], b: set[Cell], pad: int) -> float:
    """Max IoU of a against the 8 rotations/reflections of b, each normalized to its bounding box."""
    img_a, img_b = _bitmap(a, pad), _bitmap(b, pad)
    variants = [r for k in range(4) for r in (np.rot90(img_b, k), np.fliplr(np.rot90(img_b, k)))]
    return max(_iou_top_left(img_a, v) for v in variants)


def _too_similar(neg: set[Cell], others: list[set[Cell]], cfg: GeneratorConfig) -> bool:
    return any(len(neg ^ p) < cfg.neg_min_symdiff_vs_other
               or similarity(neg, p, cfg.neg_similarity_pad) >= cfg.neg_max_iou_vs_other
               for p in others)


def make_distractor(rng: np.random.Generator, positive: set[Cell], others: list[set[Cell]],
                    cfg: GeneratorConfig) -> tuple[Optional[set[Cell]], str]:
    """Edit one positive projection until it is not too similar to the other two positives."""
    if is_thin_bar(positive, cfg.thin_bar_max_len):
        return None, f"skipped_thinbar_1x{cfg.thin_bar_max_len}"
    cells, status = None, "uninitialized"
    for _ in range(cfg.neg_similarity_tries):
        cells, status = perturb(rng, positive, cfg)
        if not _too_similar(cells, others, cfg):
            return cells, status
    return cells, status + "_maybe_similar_otherpos"
