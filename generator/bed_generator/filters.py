"""Hidden-block filter and sampled two-view ambiguity check."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .config import GeneratorConfig
from .geometry import PLANES, Block, projection_cells, view_limits
from .render import silhouette
from .views import View
from .voxels import sample_object


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int(np.logical_or(a, b).sum())
    return 1.0 if union == 0 else int(np.logical_and(a, b).sum()) / union


def _nearly_unchanged(base: np.ndarray, other: np.ndarray, cfg: GeneratorConfig) -> bool:
    return (int(np.logical_xor(base, other).sum()) < cfg.hidden_block_min_pixels
            and iou(base, other) >= cfg.hidden_block_iou)


def hidden_block_ok(blocks: Sequence[Block], q1: View, q2: View, cfg: GeneratorConfig) -> bool:
    """Reject objects with a block whose removal leaves both silhouettes nearly unchanged."""
    limits = [view_limits(blocks, q, cfg.view_pad) for q in (q1, q2)]
    base = [silhouette(blocks, q, lim, cfg) for q, lim in zip((q1, q2), limits)]
    for k in range(len(blocks)):
        ablated = list(blocks[:k]) + list(blocks[k + 1:])
        if (_nearly_unchanged(base[0], silhouette(ablated, q1, limits[0], cfg), cfg)
                and _nearly_unchanged(base[1], silhouette(ablated, q2, limits[1], cfg), cfg)):
            return False
    return True


def is_ambiguous(rng: np.random.Generator, blocks: Sequence[Block], q1: View, q2: View, cfg: GeneratorConfig) -> bool:
    """Sample alternative objects; ambiguous if one matches both silhouettes but differs in a projection."""
    projections = [projection_cells(blocks, p) for p in PLANES]
    limits = [view_limits(blocks, q, cfg.view_pad) for q in (q1, q2)]
    base = [silhouette(blocks, q, lim, cfg) for q, lim in zip((q1, q2), limits)]
    for _ in range(cfg.collision_trials):
        alt = sample_object(rng, cfg)
        # Rendering draws no random numbers, so stopping at the first mismatching view is safe.
        if all(iou(b, silhouette(alt, q, lim, cfg)) >= cfg.collision_iou
               for b, q, lim in zip(base, (q1, q2), limits)):
            if [projection_cells(alt, p) for p in PLANES] != projections:
                return True
    return False
