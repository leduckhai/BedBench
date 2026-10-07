"""Adversarial color assignment (Algorithm "Assign Adversarial Colors")."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .geometry import NEIGHBORS_6, Block, pick

# Distinct hues with similar luminance, to weaken brightness-based boundary cues.
COLOR_PAIRS = (
    ("red", "magenta"), ("green", "gold"), ("cyan", "yellow"),
    ("blue", "purple"), ("orange", "orchid"), ("teal", "slategray"),
)


def assign_colors(rng: np.random.Generator, blocks: Sequence[Block]) -> list[Block]:
    """Visit blocks in random order; color each uncolored block and one uncolored neighbor with a pair."""
    index = {(b.x, b.y, b.z): i for i, b in enumerate(blocks)}
    order = list(range(len(blocks)))
    rng.shuffle(order)
    colors: dict[int, str] = {}
    for i in order:
        if i in colors:
            continue
        b = blocks[i]
        neighbor = next((index[p] for p in ((b.x + dx, b.y + dy, b.z + dz) for dx, dy, dz in NEIGHBORS_6)
                         if p in index and index[p] not in colors), None)
        first, second = pick(rng, COLOR_PAIRS)
        if rng.random() > 0.5:
            first, second = second, first
        colors[i] = first
        if neighbor is not None:
            colors[neighbor] = second
    return [Block(b.x, b.y, b.z, colors[i]) for i, b in enumerate(blocks)]
