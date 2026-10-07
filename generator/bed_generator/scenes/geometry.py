"""Primitives, meshes, and projections."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

Face = tuple[int, ...]
Limits = tuple[float, float, float, float]   # xmin, xmax, ymin, ymax


@dataclass
class Primitive:
    """Axis-aligned box, cube, or cylinder with its minimum corner at (x, y, z) and extents (sx, sy, sz)."""
    kind: str                 # "box", "cube", or "cylinder"
    x: float
    y: float
    z: float
    sx: float
    sy: float
    sz: float
    axis: str = "z"           # cylinder axis
    color: str = "#bfbfbf"


BOX_FACES: list[Face] = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]


def box_mesh(p: Primitive) -> tuple[np.ndarray, list[Face]]:
    x, y, z, sx, sy, sz = p.x, p.y, p.z, p.sx, p.sy, p.sz
    v = np.array([[x, y, z], [x + sx, y, z], [x + sx, y + sy, z], [x, y + sy, z],
                  [x, y, z + sz], [x + sx, y, z + sz], [x + sx, y + sy, z + sz], [x, y + sy, z + sz]], dtype=float)
    return v, BOX_FACES


def cylinder_mesh(p: Primitive, n: int) -> tuple[np.ndarray, list[Face]]:
    """n-gon prism inscribed in the bounding box, along p.axis."""
    x, y, z, sx, sy, sz = p.x, p.y, p.z, p.sx, p.sy, p.sz
    bottom, top = [], []
    for i in range(n):  # scalar cos/sin on purpose: vectorized versions may differ in the last bit
        ang = 2 * np.pi * i / n
        if p.axis == "z":
            px, py = x + sx / 2 + (sx / 2) * np.cos(ang), y + sy / 2 + (sy / 2) * np.sin(ang)
            bottom.append([px, py, z]), top.append([px, py, z + sz])
        elif p.axis == "x":
            py, pz = y + sy / 2 + (sy / 2) * np.cos(ang), z + sz / 2 + (sz / 2) * np.sin(ang)
            bottom.append([x, py, pz]), top.append([x + sx, py, pz])
        else:
            px, pz = x + sx / 2 + (sx / 2) * np.cos(ang), z + sz / 2 + (sz / 2) * np.sin(ang)
            bottom.append([px, y, pz]), top.append([px, y + sy, pz])
    faces = [tuple(range(n)), tuple(range(n, 2 * n))] + [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    return np.array(bottom + top, dtype=float), faces


def mesh(p: Primitive, cylinder_segments: int = 24) -> tuple[np.ndarray, list[Face]]:
    return cylinder_mesh(p, cylinder_segments) if p.kind == "cylinder" else box_mesh(p)


def rotation(azimuth: float, elevation: float) -> np.ndarray:
    az, el = np.deg2rad(azimuth), np.deg2rad(elevation)
    rx = np.array([[1, 0, 0], [0, np.cos(el), -np.sin(el)], [0, np.sin(el), np.cos(el)]], dtype=float)
    rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]], dtype=float)
    return rx @ rz


def project(points: np.ndarray, view: str, R: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
    """2D image coordinates and a depth key (larger values are drawn on top).

    view is "query" (rotation R) or a canonical plane "xy", "yz", "zx" (orthographic, dropping z, x, y).
    """
    P = points.astype(float)
    if view == "query":
        pc = (R @ P.T).T
        return pc[:, :2], -pc[:, 2]
    u, v, d = {"xy": (0, 1, 2), "yz": (1, 2, 0), "zx": (0, 2, 1)}[view]
    return P[:, [u, v]], -P[:, d]


def view_limits(scenes: Sequence[Sequence[Primitive]], view: str, R: Optional[np.ndarray], pad: float,
                cylinder_segments: int = 24) -> Limits:
    """Padded bounds of all projected vertices of one or more scenes (shared limits for scene pairs)."""
    pts = np.vstack([project(mesh(p, cylinder_segments)[0], view, R)[0] for scene in scenes for p in scene])
    return (float(pts[:, 0].min() - pad), float(pts[:, 0].max() + pad),
            float(pts[:, 1].min() - pad), float(pts[:, 1].max() + pad))


def boxes_intersect(a: Primitive, b: Primitive) -> bool:
    """Open-interval overlap of the axis-aligned bounding boxes (touching faces do not intersect)."""
    return not (a.x + a.sx <= b.x or b.x + b.sx <= a.x or a.y + a.sy <= b.y or b.y + b.sy <= a.y
                or a.z + a.sz <= b.z or b.z + b.sz <= a.z)


def non_intersecting(scene: Sequence[Primitive]) -> bool:
    return not any(boxes_intersect(scene[i], scene[j]) for i in range(len(scene)) for j in range(i + 1, len(scene)))
