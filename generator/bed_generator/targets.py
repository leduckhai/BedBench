"""Task targets derived from one object: extents, container box, color counts."""
from __future__ import annotations

from itertools import permutations
from typing import Sequence

import numpy as np

from .geometry import Block


def extents(blocks: Sequence[Block]) -> tuple[int, int, int]:
    xs, ys, zs = ([getattr(b, a) for b in blocks] for a in "xyz")
    return max(xs) - min(xs) + 1, max(ys) - min(ys) + 1, max(zs) - min(zs) + 1


def fits(ext: Sequence[int], box: Sequence[int]) -> bool:
    return any(all(e <= d for e, d in zip(p, box)) for p in permutations(ext))


def container_box(rng: np.random.Generator, ext: tuple[int, int, int]) -> tuple[tuple[int, int, int], bool]:
    """A box that fits (extents plus 0-2 slack per side) or does not (largest side shrunk), each with p=0.5."""
    if rng.random() > 0.5:
        dims = [e + int(rng.integers(0, 3)) for e in ext]
        rng.shuffle(dims)
        return tuple(dims), True
    small, mid, large = sorted(ext)
    box = [max(1, small + int(rng.integers(-1, 2))),
           max(1, mid + int(rng.integers(-1, 2))),
           max(1, large + int(rng.integers(-3, 0)))]
    if fits(ext, box):
        side = max(1, min(ext) - 1)
        return (side, side, side), False
    rng.shuffle(box)
    return tuple(box), False


def color_counts(blocks: Sequence[Block]) -> dict[str, int]:
    """Per-color block counts, in first-occurrence order."""
    counts: dict[str, int] = {}
    for b in blocks:
        counts[b.color] = counts.get(b.color, 0) + 1
    return counts
