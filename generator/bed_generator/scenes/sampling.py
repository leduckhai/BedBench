"""Scene sampling, visibility validation, and scene distractors.

The random stream is consumed in the same order as the generator that produced the released scenes.
`rng.choice(list)` is written as `list[rng.integers(len(list))]`, which draws the same number.
"""
from __future__ import annotations

import copy
from typing import Optional

import numpy as np

from .config import COLORS, PLANES, SceneConfig
from .geometry import Limits, Primitive, boxes_intersect, non_intersecting, rotation, view_limits
from .raster import iou, view_ok

KINDS = ("box", "cylinder", "cube")
ATTACH_MODES = ("top", "side_x+", "side_x-", "side_y+", "side_y-")
MUTATIONS = ("swap", "delete", "move", "resize", "lift")
MOVE_STEPS = (-1.8, -1.2, 1.2, 1.8)


def _pick(rng: np.random.Generator, seq):
    return seq[rng.integers(len(seq))]


def sample_primitive(rng: np.random.Generator, cfg: SceneConfig) -> Primitive:
    kind = _pick(rng, KINDS)
    s = rng.uniform(*cfg.extent_range, size=3)
    if kind == "cube":
        s = np.full(3, rng.uniform(*cfg.cube_edge_range))
    p = Primitive(kind, 0.0, 0.0, 0.0, float(s[0]), float(s[1]), float(s[2]))
    if kind == "cylinder":
        p.axis = _pick(rng, ("x", "y", "z"))
    return p


def place(rng: np.random.Generator, scene: list[Primitive], p: Primitive, cfg: SceneConfig) -> None:
    """Attach p on top of or beside a random existing primitive (small gaps and jitter), or put it on the floor."""
    if not (scene and rng.random() < cfg.attach_prob):
        p.x, p.y, p.z = rng.uniform(-cfg.floor_range, cfg.floor_range), rng.uniform(-cfg.floor_range, cfg.floor_range), 0.0
        return
    base = _pick(rng, scene)
    mode = _pick(rng, ATTACH_MODES)
    jitter = lambda s: rng.uniform(-s, s)  # noqa: E731
    if mode == "top":
        p.x = base.x + jitter(0.35)
        p.y = base.y + jitter(0.35)
        p.z = base.z + base.sz + rng.uniform(0.05, 0.25)
        return
    if mode == "side_x+":
        p.x = base.x + base.sx + rng.uniform(0.05, 0.35)
        p.y = base.y + jitter(0.3)
    elif mode == "side_x-":
        p.x = base.x - p.sx - rng.uniform(0.05, 0.35)
        p.y = base.y + jitter(0.3)
    elif mode == "side_y+":
        p.x = base.x + jitter(0.3)
        p.y = base.y + base.sy + rng.uniform(0.05, 0.35)
    else:
        p.x = base.x + jitter(0.3)
        p.y = base.y - p.sy - rng.uniform(0.05, 0.35)
    p.z = max(0.0, base.z + jitter(0.2))


def grow_scene(rng: np.random.Generator, n: int, cfg: SceneConfig) -> list[Primitive]:
    """Add non-intersecting primitives until the scene has n of them or the attempt budget runs out."""
    scene: list[Primitive] = []
    for _ in range(cfg.object_attempts):
        if len(scene) >= n:
            break
        p = sample_primitive(rng, cfg)
        place(rng, scene, p, cfg)
        if not any(boxes_intersect(p, q) for q in scene):
            scene.append(p)
    return scene


def scene_visible(scene: list[Primitive], cfg: SceneConfig) -> bool:
    """Both query views and all three canonical views pass the visibility checks."""
    for az, el in cfg.query_views:
        R = rotation(az, el)
        if not view_ok(scene, "query", view_limits([scene], "query", R, cfg.pad, cfg.cylinder_segments), cfg, R)[0]:
            return False
    return all(view_ok(scene, plane, view_limits([scene], plane, None, cfg.pad, cfg.cylinder_segments), cfg)[0]
               for plane in PLANES)


def sample_scene(rng: np.random.Generator, cfg: SceneConfig) -> list[Primitive]:
    """A valid scene with colors assigned (colors are drawn after validation)."""
    for _ in range(cfg.scene_attempts):
        n = int(rng.integers(cfg.n_objects[0], cfg.n_objects[1] + 1))
        scene = grow_scene(rng, n, cfg)
        if len(scene) == n and scene_visible(scene, cfg):
            for p in scene:
                p.color = _pick(rng, COLORS)
            return scene
    raise RuntimeError("no valid scene within the attempt budget")


def mutate(rng: np.random.Generator, scene: list[Primitive]) -> list[Primitive]:
    """Swap two primitives' positions, delete one, move one by +-1.2/1.8 along x or y, resize one, or lift one."""
    m = copy.deepcopy(scene)
    mode = _pick(rng, MUTATIONS)
    if mode == "swap":
        i, j = rng.choice(len(m), 2, replace=False)
        (m[i].x, m[i].y, m[i].z), (m[j].x, m[j].y, m[j].z) = (m[j].x, m[j].y, m[j].z), (m[i].x, m[i].y, m[i].z)
    elif mode == "delete":
        del m[rng.integers(len(m))]
    elif mode == "resize":
        p = m[rng.integers(len(m))]
        p.sx *= rng.uniform(1.2, 1.6)
        p.sy *= rng.uniform(0.9, 1.2)
        p.sz *= rng.uniform(0.9, 1.2)
    elif mode == "lift":
        m[rng.integers(len(m))].z += rng.uniform(0.8, 1.6)
    else:
        p = m[rng.integers(len(m))]
        axis = _pick(rng, ("x", "y"))
        step = _pick(rng, MOVE_STEPS)
        if axis == "x":
            p.x += step
        else:
            p.y += step
    return m


def scene_distractor(rng: np.random.Generator, scene: list[Primitive], plane: str,
                     cfg: SceneConfig) -> Optional[tuple[list[Primitive], Limits, float]]:
    """A mutated scene that stays non-intersecting, passes visibility on `plane` in limits shared with the
    original, and has a silhouette IoU with it strictly inside (min_distractor_iou, max_distractor_iou)."""
    for _ in range(cfg.distractor_attempts):
        m = mutate(rng, scene)
        if not non_intersecting(m):
            continue
        lims = view_limits([scene, m], plane, None, cfg.pad, cfg.cylinder_segments)
        ok_pos, r_pos = view_ok(scene, plane, lims, cfg)
        if not ok_pos:
            continue
        ok_neg, r_neg = view_ok(m, plane, lims, cfg)
        if not ok_neg:
            continue
        overlap = iou(r_pos.mask, r_neg.mask)
        if cfg.min_distractor_iou < overlap < cfg.max_distractor_iou:
            return m, lims, overlap
    return None
