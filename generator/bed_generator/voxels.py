"""Connected voxel objects and 2D cell-set utilities."""
from __future__ import annotations

import numpy as np

from .config import GeneratorConfig
from .geometry import NEIGHBORS_6, Block, Cell, pick

_XY_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _zyx(b: Block) -> tuple[int, int, int]:
    return (b.z, b.y, b.x)


def grow_object(rng: np.random.Generator, n_blocks: int, cfg: GeneratorConfig) -> list[Block]:
    """Grow a connected, supported object from the origin (Algorithm "Grow a Connected Voxel Object")."""
    occupied = {(0, 0, 0)}
    for _ in range(cfg.growth_attempts):
        if len(occupied) >= n_blocks:
            break
        x, y, z = pick(rng, sorted(occupied, key=lambda p: (p[2], p[1], p[0])))
        if rng.random() < cfg.horizontal_growth_prob:
            dx, dy = pick(rng, _XY_STEPS)
            cand = (x + dx, y + dy, z)
        else:
            cand = (x, y, z + 1)
        cx, cy, cz = cand
        if abs(cx) > cfg.xy_radius or abs(cy) > cfg.xy_radius or not 0 <= cz <= cfg.max_height:
            continue
        if cand in occupied or (cz > 0 and (cx, cy, cz - 1) not in occupied):
            continue
        if any((cx + a, cy + b, cz + c) in occupied for a, b, c in NEIGHBORS_6):
            occupied.add(cand)
    if len(occupied) < n_blocks:
        raise ValueError(f"grew {len(occupied)} of {n_blocks} blocks within the attempt budget")
    return sorted((Block(*p) for p in occupied), key=_zyx)


def sample_object(rng: np.random.Generator, cfg: GeneratorConfig) -> list[Block]:
    n_blocks = int(rng.integers(cfg.n_blocks[0], cfg.n_blocks[1] + 1))
    return grow_object(rng, n_blocks, cfg)


def cell_bounds(cells: set[Cell], pad: int = 1) -> tuple[int, int, int, int]:
    us, vs = [c[0] for c in cells], [c[1] for c in cells]
    return min(us) - pad, max(us) + pad, min(vs) - pad, max(vs) + pad


def is_thin_bar(cells: set[Cell], max_len: int) -> bool:
    """A 1-wide bar of at most max_len cells: too degenerate for a projection distractor."""
    if not cells:
        return True
    umin, umax, vmin, vmax = cell_bounds(cells, pad=0)
    w, h = umax - umin + 1, vmax - vmin + 1
    return min(w, h) == 1 and max(w, h) <= max_len


def _neighbors4(c: Cell) -> list[Cell]:
    return [(c[0] + du, c[1] + dv) for du, dv in _XY_STEPS]


def is_connected(cells: set[Cell]) -> bool:
    if not cells:
        return False
    start = min(cells)
    seen, stack = {start}, [start]
    while stack:
        for nb in _neighbors4(stack.pop()):
            if nb in cells and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(cells)


def has_holes(cells: set[Cell]) -> bool:
    """True if some empty cell inside the padded bounding box is unreachable from its border."""
    if not cells:
        return False
    umin, umax, vmin, vmax = cell_bounds(cells, pad=1)
    border = [(u, v) for u in range(umin, umax + 1) for v in (vmin, vmax)]
    border += [(u, v) for v in range(vmin, vmax + 1) for u in (umin, umax)]
    seen = {c for c in border if c not in cells}
    stack = list(seen)
    while stack:
        for nb in _neighbors4(stack.pop()):
            if umin <= nb[0] <= umax and vmin <= nb[1] <= vmax and nb not in cells and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return any((u, v) not in cells and (u, v) not in seen
               for u in range(umin, umax + 1) for v in range(vmin, vmax + 1))
