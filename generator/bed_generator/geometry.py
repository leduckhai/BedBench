"""Blocks, cameras, and projections of unit cubes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

Cell = tuple[int, int]
PLANES = ("zx", "yz", "xy")

CUBE_VERTS = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
], dtype=float)
CUBE_FACES = ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
NEIGHBORS_6 = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


@dataclass(frozen=True)
class Block:
    """A unit cube with its minimum corner at integer (x, y, z)."""

    x: int
    y: int
    z: int
    color: Optional[str] = None


def pick(rng: np.random.Generator, items: Sequence):
    """Uniform choice; consumes the RNG exactly like ``rng.choice(items)``."""
    return items[int(rng.integers(len(items)))]


def camera_matrix(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Oblique orthographic camera: rotate about z by azimuth, then about x by elevation."""
    az, el = np.deg2rad(float(azimuth_deg)), np.deg2rad(float(elevation_deg))
    ca, sa, ce, se = np.cos(az), np.sin(az), np.cos(el), np.sin(el)
    rot_x = np.array([[1, 0, 0], [0, ce, -se], [0, se, ce]], dtype=float)
    rot_z = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], dtype=float)
    return rot_x @ rot_z


def project(points: np.ndarray, view: str, camera: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
    """Screen coordinates and painter depth (draw ascending) for a query or canonical view."""
    p = points.astype(float)
    if view == "query":
        pc = (camera @ p.T).T
        return pc[:, :2], -pc[:, 2]
    if view == "zx":
        return p[:, [0, 2]], -p[:, 1]
    if view == "yz":
        return p[:, [1, 2]], -p[:, 0]
    if view == "xy":
        return p[:, [0, 1]], -p[:, 2]
    raise ValueError(f"unknown view {view!r}")


def cube_vertices(block: Block) -> np.ndarray:
    return CUBE_VERTS + np.array([float(block.x), float(block.y), float(block.z)])


def projected_vertices(blocks: Sequence[Block], view: tuple[float, float]) -> np.ndarray:
    camera = camera_matrix(*view)
    return np.vstack([project(cube_vertices(b), "query", camera)[0] for b in blocks])


def view_limits(blocks: Sequence[Block], view: tuple[float, float], pad: float = 0.8) -> tuple[float, float, float, float]:
    """Fixed viewport around the projected object, padded by max(pad, 6% of the span)."""
    pts = projected_vertices(blocks, view)
    xmin, xmax = float(pts[:, 0].min()), float(pts[:, 0].max())
    ymin, ymax = float(pts[:, 1].min()), float(pts[:, 1].max())
    margin = max(0.8, float(pad), 0.06 * max(xmax - xmin, ymax - ymin))
    return xmin - margin, xmax + margin, ymin - margin, ymax + margin


def _angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    u, v = u / np.linalg.norm(u), v / np.linalg.norm(v)
    return float(np.degrees(np.arccos(float(np.clip(np.dot(u, v), -1.0, 1.0)))))


def axis_separation_ok(azimuth: float, elevation: float, min_deg: float) -> bool:
    """True if the viewing direction is at least min_deg from the -x, -y, and -z axes."""
    direction = camera_matrix(azimuth, elevation).T @ np.array([0.0, 0.0, -1.0])
    direction = direction / np.linalg.norm(direction)  # normalized again in _angle_deg, as in the original
    axes = (np.array([0.0, 0.0, -1.0]), np.array([-1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0]))
    return all(_angle_deg(direction, a) >= float(min_deg) for a in axes)


def projection_cells(blocks: Sequence[Block], plane: str) -> set[Cell]:
    """Occupied cells of the orthographic projection on plane zx, yz, or xy."""
    index = {"zx": (0, 2), "yz": (1, 2), "xy": (0, 1)}[plane]
    return {(int((b.x, b.y, b.z)[index[0]]), int((b.x, b.y, b.z)[index[1]])) for b in blocks}


def frontmost_colors(blocks: Sequence[Block], plane: str) -> dict[Cell, str]:
    """Color of the frontmost block per projection cell (+y for zx, +x for yz, +z for xy)."""
    u, v, depth = {"zx": (0, 2, 1), "yz": (1, 2, 0), "xy": (0, 1, 2)}[plane]
    best: dict[Cell, tuple[int, str]] = {}
    for b in blocks:
        xyz = (b.x, b.y, b.z)
        cell = (int(xyz[u]), int(xyz[v]))
        if cell not in best or xyz[depth] > best[cell][0]:
            best[cell] = (xyz[depth], b.color)
    return {cell: color for cell, (_, color) in best.items()}


def plane_bounds(plane: str, xy_radius: int, max_height: int) -> tuple[int, int, int, int]:
    r = int(xy_radius)
    return (-r, r, -r, r) if plane == "xy" else (-r, r, 0, int(max_height))
