"""Compare generated output against a reference: a voxel split (manifest cells and every PNG pixel) or a
T13 scene pool (primitive parameters and every PNG pixel), detected from manifest.csv in the reference.

Usage: python verify.py GENERATED_DIR REFERENCE_DIR [--n N]
Manifest columns present in only one split are ignored (the original manifest has extra legacy columns).
Reference scenes may use the original field names (face_color, cylinder_axis).
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
from PIL import Image

PRIMITIVE_FIELDS = ("kind", "x", "y", "z", "sx", "sy", "sz")


def compare_pngs(generated: str, reference: str, folders: list[str]) -> tuple[int, list[str], list[str]]:
    """Every PNG under the given folders of `generated`: (count, missing in reference, pixel-different)."""
    n, missing, diffs = 0, [], []
    for folder in folders:
        for dirpath, _, files in os.walk(os.path.join(generated, folder)):
            for name in sorted(f for f in files if f.endswith(".png")):
                rel = os.path.relpath(os.path.join(dirpath, name), generated)
                other = os.path.join(reference, rel)
                n += 1
                if not os.path.exists(other):
                    missing.append(rel)
                elif not np.array_equal(np.asarray(Image.open(os.path.join(generated, rel))), np.asarray(Image.open(other))):
                    diffs.append(rel)
    return n, missing, diffs


def compare_split(generated: str, reference: str, n: int | None) -> list[str]:
    def rows(root):
        with open(os.path.join(root, "manifest.csv"), newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    gen, ref = rows(generated), rows(reference)
    n = min(len(gen), len(ref)) if n is None else n
    shared = [c for c in gen[0] if c in ref[0]]
    diffs = [f"sample {i} {c}" for i in range(n) for c in shared if gen[i][c] != ref[i][c]]
    print(f"samples compared: {n}; manifest columns compared: {len(shared)}; manifest cell differences: {len(diffs)}")
    return diffs + report_pngs(*compare_pngs(generated, reference, [gen[i]["sample_dir"] for i in range(n)]))


def primitives(path: str) -> list[tuple]:
    with open(path, encoding="utf-8") as f:
        prims = json.load(f)
    return [tuple(p[k] for k in PRIMITIVE_FIELDS) + (p.get("color", p.get("face_color")),
            p.get("axis", p.get("cylinder_axis")) if p["kind"] == "cylinder" else None) for p in prims]


def compare_scenes(generated: str, reference: str, n: int | None) -> list[str]:
    scenes = sorted(d for d in os.listdir(generated) if d.startswith("sample_"))[:n]
    diffs = [f"{d}/{rel}" for d in scenes for rel in ["scene.json"] + [f"neg/{p}_scene.json" for p in ("xy", "yz", "zx")]
             if primitives(os.path.join(generated, d, rel)) != primitives(os.path.join(reference, d, rel))]
    print(f"scenes compared: {len(scenes)}; scene files with different primitives: {len(diffs)}")
    return diffs + report_pngs(*compare_pngs(generated, reference, scenes))


def report_pngs(n: int, missing: list[str], diffs: list[str]) -> list[str]:
    print(f"images compared: {n}; missing in reference: {len(missing)}; pixel-different: {len(diffs)}")
    return missing + diffs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("generated")
    ap.add_argument("reference")
    ap.add_argument("--n", type=int, default=None, help="only compare the first N samples or scenes")
    args = ap.parse_args()
    voxel = os.path.exists(os.path.join(args.reference, "manifest.csv"))
    problems = (compare_split if voxel else compare_scenes)(args.generated, args.reference, args.n)
    print("first differences:", problems[:10])
    return int(bool(problems))


if __name__ == "__main__":
    sys.exit(main())
