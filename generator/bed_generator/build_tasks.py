"""Build the T1-T13 evaluation datasets from a generated voxel split and scene pool.

  python -m bed_generator.build_tasks --samples bed_voxels --scenes bed_scenes --out bed_dataset

Writes one datasets.save_to_disk directory per configuration (OUT/T5_color, OUT/T11_invalid_uncolor, ...),
the layout the evaluation harness reads with --dataset OUT. Only the conditions reported in the paper
are built. T1-T12 use the first 100 objects of the split; T13 uses all scenes of the pool.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, Optional

from datasets import Dataset, DatasetDict, Features, Image, Sequence, Value
from PIL import Image as PILImage

N_OBJECTS = 100
N_PAIRS = 100           # T3 object pairs
T6_K = 3                # T6 asks whether some column holds more than k blocks
SEED = 42
PLANES = ("xy", "yz", "zx")
QUERY_STEM = {"uncolor": "query{}_uncolored", "uncolor_axes": "query{}_uncolored_axes",
              "color": "query{}", "color_axes": "query{}_axes"}
JSON_COLUMNS = ("base_blocks_json", "colors_json", "box_dims_json", "color_distribution_json")


def load_objects(samples: Path, n: int = N_OBJECTS) -> list[dict]:
    """First n manifest rows with parsed JSON columns and the absolute sample directory."""
    with open(samples / "manifest.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[:n]
    for row in rows:
        for col in JSON_COLUMNS:
            row[col] = json.loads(row[col])
        row["dir"] = samples / row["sample_dir"]
    return rows


def png(path: Path) -> dict:
    return {"bytes": path.read_bytes(), "path": None}


def query(obj: dict, variant: str) -> dict:
    stem = QUERY_STEM[variant]
    return {f"query_image_{i}": png(obj["dir"] / f"{stem.format(i)}.png") for i in (1, 2)}


# ── per-object targets ────────────────────────────────────────────────────────

def _mirror_symmetric(cells: set, axis: int) -> bool:
    lo, hi = min(c[axis] for c in cells), max(c[axis] for c in cells)
    return {tuple(hi - (v - lo) if a == axis else v for a, v in enumerate(c)) for c in cells} == cells


def any_symmetric_projection(blocks: list) -> bool:
    """T8: some canonical projection is mirror-symmetric about either of its axes."""
    for u, v in ((0, 1), (1, 2), (0, 2)):
        cells = {(b[u], b[v]) for b in blocks}
        if _mirror_symmetric(cells, 0) or _mirror_symmetric(cells, 1):
            return True
    return False


def same_color_neighbors(blocks: list, colors: list) -> bool:
    """T10: two face-adjacent blocks share a color."""
    color = {tuple(b): c for b, c in zip(blocks, colors)}
    return any(color.get((x + dx, y + dy, z + dz)) == c
               for (x, y, z), c in color.items()
               for dx, dy, dz in ((1, 0, 0), (0, 1, 0), (0, 0, 1)))


def dominant_colors(distribution: dict) -> list[str]:
    top = max(distribution.values())
    return [c for c, k in distribution.items() if k == top]


SINGLE = {  # task -> (variants, extra columns, extra features)
    "T1": (("uncolor", "uncolor_axes", "color"),
           lambda o: {"distractors": sorted({max(1, int(o["task_volume"]) - 2), max(1, int(o["task_volume"]) - 1),
                                             int(o["task_volume"]) + 1, int(o["task_volume"]) + 2}),
                      "answer": int(o["task_volume"])},
           {"distractors": Sequence(Value("int32")), "answer": Value("int32")}),
    "T2": (("uncolor", "uncolor_axes"),
           lambda o: {"box_dims": o["box_dims_json"], "answer": o["fits_box"] == "1"},
           {"box_dims": Sequence(Value("int32")), "answer": Value("bool")}),
    "T4": (("color", "color_axes"), lambda o: {"answer": int(o["color_total"])}, {"answer": Value("int32")}),
    "T5": (("color", "color_axes"), lambda o: {"color_distribution_json": json.dumps(o["color_distribution_json"])},
           {"color_distribution_json": Value("string")}),
    "T6": (("uncolor", "uncolor_axes"),
           lambda o: {"k": T6_K, "answer": max(Counter((b[0], b[1]) for b in o["base_blocks_json"]).values()) > T6_K},
           {"k": Value("int32"), "answer": Value("bool")}),
    "T7": (("color", "color_axes"), lambda o: {"answer": dominant_colors(o["color_distribution_json"])},
           {"answer": Sequence(Value("string"))}),
    "T8": (("uncolor", "uncolor_axes"), lambda o: {"answer": any_symmetric_projection(o["base_blocks_json"])},
           {"answer": Value("bool")}),
    "T9": (("uncolor", "uncolor_axes"), lambda o: {"answer": len({b[2] for b in o["base_blocks_json"]})},
           {"answer": Value("int32")}),
    "T10": (("color", "color_axes"), lambda o: {"answer": same_color_neighbors(o["base_blocks_json"], o["colors_json"])},
            {"answer": Value("bool")}),
}
QUERY_FEATURES = {"sample_id": Value("string"), "query_image_1": Image(), "query_image_2": Image()}


def build_single(objects: list[dict]) -> dict[str, Dataset]:
    out = {}
    for task, (variants, target, features) in SINGLE.items():
        for variant in variants:
            rows = [{"sample_id": o["sample_id"], **query(o, variant), **target(o)} for o in objects]
            out[f"{task}_{variant}"] = Dataset.from_list(rows, features=Features({**QUERY_FEATURES, **features}))
    return out


# ── T3: object pairs ──────────────────────────────────────────────────────────

def build_t3(objects: list[dict]) -> dict[str, Dataset]:
    """Shared blocks and blocks unique to each object; the color variant also matches block colors."""
    n = len(objects)
    # ponytail: 300 pairs are drawn and the first 100 kept, as in the original export (same RNG stream)
    pairs = random.Random(SEED).sample([(i, j) for i in range(n) for j in range(i + 1, n)], 300)[:N_PAIRS]
    features = Features({"pair_id": Value("string"), "sample_a_id": Value("string"), "sample_b_id": Value("string"),
                         **{f"query_{s}_image_{i}": Image() for s in "ab" for i in (1, 2)},
                         "answer_intersection": Value("int32"), "answer_only_a": Value("int32"),
                         "answer_only_b": Value("int32")})
    out = {}
    for variant, with_color in (("uncolor_axes", False), ("color_axes", True)):
        rows = []
        for i, j in pairs:
            a, b = objects[i], objects[j]
            keys = [{tuple(blk) + ((c,) if with_color else ()) for blk, c in zip(o["base_blocks_json"], o["colors_json"])}
                    for o in (a, b)]
            images = {f"query_{s}_image_{k}": img for s, o in (("a", a), ("b", b))
                      for k, img in zip((1, 2), query(o, variant).values())}
            rows.append({"pair_id": f"{a['sample_id']}_vs_{b['sample_id']}", "sample_a_id": a["sample_id"],
                         "sample_b_id": b["sample_id"], **images,
                         "answer_intersection": len(keys[0] & keys[1]), "answer_only_a": len(keys[0] - keys[1]),
                         "answer_only_b": len(keys[1] - keys[0])})
        out[f"T3_{variant}"] = Dataset.from_list(rows, features=features)
    return out


# ── T11 / T13: projection matching ───────────────────────────────────────────

def projection_rows(rng: random.Random, sample_id: str, query_images: dict[str, dict], candidate: Callable,
                    variants: Iterable[str]) -> dict[str, list[dict]]:
    """Per target plane: 'invalid' shows two correct projections and the distractor of the target plane (find
    the distractor); 'valid' shows two distractors and the correct target projection. One shuffle per subtask
    and plane (invalid first), shared by all variants. candidate(folder, plane, variant) -> image."""
    rows = {f"{s}_{v}": [] for s in ("invalid", "valid") for v in variants}
    for target in PLANES:
        others = [p for p in PLANES if p != target]
        for subtask, odd, rest in (("invalid", "neg", "pos"), ("valid", "pos", "neg")):
            order = [0, 1, 2]
            rng.shuffle(order)
            cands = [(rest, others[0]), (rest, others[1]), (odd, target)]
            cands = [cands[k] for k in order]
            for variant in variants:
                rows[f"{subtask}_{variant}"].append({
                    "sample_id": sample_id, **query_images[variant],
                    **{f"candidate_{k}": candidate(folder, plane, variant) for k, (folder, plane) in enumerate(cands)},
                    "answer": cands.index((odd, target))})
    return rows


PROJECTION_FEATURES = Features({**QUERY_FEATURES, **{f"candidate_{k}": Image() for k in range(3)},
                                "answer": Value("int32")})
T11_VARIANTS = {"uncolor": "_uncolored", "uncolor_axes": "_uncolored", "color": ""}  # variant -> projection folder suffix
T13_VARIANTS = {"uncolor": "_uncolored", "color": ""}                               # variant -> folder suffix


def _merge(datasets: dict[str, list], rows: dict[str, list]) -> None:
    for name, r in rows.items():
        datasets.setdefault(name, []).extend(r)


def build_t11(objects: list[dict]) -> dict[str, Dataset]:
    """Every object with every plane as the target; objects without distractors on all planes are skipped."""
    rng, rows = random.Random(SEED), {}
    for o in objects:
        if not all((o["dir"] / f / f"{p}.png").exists() for f in ("pos", "neg") for p in PLANES):
            continue
        _merge(rows, projection_rows(rng, o["sample_id"], {v: query(o, v) for v in T11_VARIANTS},
                                     lambda folder, plane, v: png(o["dir"] / f"{folder}{T11_VARIANTS[v]}" / f"{plane}.png"),
                                     T11_VARIANTS))
    return {f"T11_{name}": Dataset.from_list(r, features=PROJECTION_FEATURES) for name, r in rows.items()}


def build_t13(scenes: Path) -> dict[str, Dataset]:
    """Every scene of the pool with every plane as the target."""
    rng, rows = random.Random(SEED), {}
    for d in sorted(scenes.glob("sample_*")):
        queries = {v: {f"query_image_{i}": png(d / f"queries{sfx}" / f"view_{i}.png") for i in (1, 2)}
                   for v, sfx in T13_VARIANTS.items()}
        _merge(rows, projection_rows(rng, str(int(d.name.removeprefix("sample_"))), queries,
                                     lambda folder, plane, v: png(d / f"{folder}{T13_VARIANTS[v]}" / f"{plane}.png"),
                                     T13_VARIANTS))
    return {f"T13_{name}": Dataset.from_list(r, features=PROJECTION_FEATURES) for name, r in rows.items()}


# ── T12: view matching ────────────────────────────────────────────────────────

T12_VARIANTS = {"uncolor": ("pos_uncolored", "uncolor"), "uncolor_axes": ("pos_uncolored", "uncolor_axes"),
                "color": ("pos", "color")}  # variant -> (silhouette folder, candidate query variant)


def build_t12(objects: list[dict]) -> dict[str, Dataset]:
    """Three projections of an object and two query views: one of the object, one of a distractor object
    with a similar block count (+-2). The answer is the index of the object's view."""
    rng = random.Random(SEED)
    by_size: dict[int, list[int]] = {}
    for i, o in enumerate(objects):
        by_size.setdefault(int(o["n_blocks_base"]), []).append(i)
    rows = {v: [] for v in T12_VARIANTS}
    for i, o in enumerate(objects):
        n = int(o["n_blocks_base"])
        pool = [j for d in (0, -1, 1, -2, 2) for j in by_size.get(n + d, []) if j != i]
        distractor = objects[rng.choice(pool)]
        q = 2 if rng.choice([True, False]) else 1
        own_first = rng.choice([True, False])
        order = [0, 1, 2]
        rng.shuffle(order)
        for variant, (folder, cand_variant) in T12_VARIANTS.items():
            own = query(o, cand_variant)[f"query_image_{q}"]
            other = query(distractor, cand_variant)["query_image_1"]
            rows[variant].append({
                "sample_id": o["sample_id"],
                **{f"silhouette_{k}": png(o["dir"] / folder / f"{PLANES[p]}.png") for k, p in enumerate(order)},
                "candidate_0": own if own_first else other, "candidate_1": other if own_first else own,
                "answer_idx": 0 if own_first else 1})
    features = Features({"sample_id": Value("string"), **{f"silhouette_{k}": Image() for k in range(3)},
                         "candidate_0": Image(), "candidate_1": Image(), "answer_idx": Value("int32")})
    return {f"T12_{v}": Dataset.from_list(r, features=features) for v, r in rows.items()}


def build_all(samples: Path, scenes: Optional[Path] = None) -> dict[str, Dataset]:
    objects = load_objects(samples)
    out = {**build_single(objects), **build_t3(objects), **build_t11(objects), **build_t12(objects)}
    return {**out, **build_t13(scenes)} if scenes else out


# ── T5 fine-tuning data ───────────────────────────────────────────────────────

FINETUNE_SPLITS = {"train": (0, 800), "validation": (800, 900)}   # object index ranges of the fine-tuning pool


def _object_key(o: dict) -> frozenset:
    """Blocks with colors, translated so the minimum x and y are 0."""
    blocks = o["base_blocks_json"]
    mx, my = min(b[0] for b in blocks), min(b[1] for b in blocks)
    return frozenset((b[0] - mx, b[1] - my, b[2], c) for b, c in zip(blocks, o["colors_json"]))


def check_finetune_pool(pool: list[dict], benchmark: list[dict]) -> None:
    """The training pool must use the benchmark's color palette and contain no benchmark object."""
    palette = lambda objs: set().union(*(set(o["colors_json"]) for o in objs))  # noqa: E731
    if palette(pool) != palette(benchmark):
        raise ValueError(f"fine-tuning palette {sorted(palette(pool))} differs from the benchmark palette "
                         f"{sorted(palette(benchmark))}; generate the pool with this generator's default coloring")
    leaked = {_object_key(o) for o in pool} & {_object_key(o) for o in benchmark}
    if leaked:
        raise ValueError(f"{len(leaked)} fine-tuning objects equal benchmark objects (blocks and colors)")


def build_t5_finetune(samples: Path, benchmark: list[dict]) -> DatasetDict:
    """T5 (default color views) on a separate pool generated with --seed 100: objects 0-799 for training,
    800-899 for validation. The test set is the benchmark's T5_color. Images are converted to RGB, as in the
    original export (the renders are fully opaque, so pixels are unchanged)."""
    objects = load_objects(samples, n=FINETUNE_SPLITS["validation"][1])
    check_finetune_pool(objects, benchmark)
    features = Features({**QUERY_FEATURES, "color_distribution_json": Value("string")})
    rows = [{"sample_id": o["sample_id"],
             **{f"query_image_{i}": PILImage.open(o["dir"] / f"query{i}.png").convert("RGB") for i in (1, 2)},
             "color_distribution_json": json.dumps(o["color_distribution_json"])} for o in objects]
    return DatasetDict({split: Dataset.from_list(rows[a:b], features=features)
                        for split, (a, b) in FINETUNE_SPLITS.items()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", required=True, help="generator output directory (with manifest.csv)")
    parser.add_argument("--scenes", help="scene generator output directory (T13; omitted: T1-T12 only)")
    parser.add_argument("--finetune-samples", help="fine-tuning pool (python -m bed_generator --n 900 --seed 100); "
                                                   "writes OUT/T5_finetune with train and validation splits")
    parser.add_argument("--out", required=True, help="dataset directory, one subdirectory per configuration")
    args = parser.parse_args()
    for name, ds in build_all(Path(args.samples), Path(args.scenes) if args.scenes else None).items():
        ds.save_to_disk(str(Path(args.out) / name))
        print(f"{name:24s} {len(ds):4d} rows")
    if args.finetune_samples:
        ft = build_t5_finetune(Path(args.finetune_samples), load_objects(Path(args.samples)))
        ft.save_to_disk(str(Path(args.out) / "T5_finetune"))
        print(f"{'T5_finetune':24s} " + ", ".join(f"{k} {len(v)}" for k, v in ft.items()))


if __name__ == "__main__":
    main()
