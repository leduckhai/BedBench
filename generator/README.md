# BedBench generator

Generates the two image pools of BedBench and builds the evaluation datasets from them:

- **Voxel objects** (T1–T12, `bed_generator`): connected voxel objects, two query views, colors,
  canonical projections, projection distractors, all rendered variants, and a `manifest.csv` with the
  targets. The benchmark uses 100 objects. Generation uses a single random stream, so `--n` only sets
  where it stops: the first k objects are the same for any `--n` >= k.
- **Multi-object scenes** (T13, `bed_generator.scenes`): 6–8 non-intersecting boxes, cubes, and
  cylinders, two query views, canonical projections, and one scene distractor per projection plane.
  The pool has its own random stream.

With the default configurations and seed 42, both reproduce the released pools exactly (same manifest
values and primitive parameters, pixel-identical images).

## Usage

```bash
pip install -r requirements.txt
python -m bed_generator --out bed_voxels --n 100 --seed 42          # about 7 s per object
python -m bed_generator.scenes --out bed_scenes --n 100 --seed 42   # about 0.5 s per scene

# evaluation datasets for T1-T13, read by the harness with --dataset bed_dataset
python -m bed_generator.build_tasks --samples bed_voxels --scenes bed_scenes --out bed_dataset

python -m unittest tests.test_reproduction tests.test_scenes   # first 2 objects and scenes vs. the release (~15 s)
python verify.py bed_voxels path/to/released_split              # full manifest + pixel comparison
python verify.py bed_scenes path/to/released_scenes             # primitives + pixel comparison
```

Voxel generation is the slow step: the ambiguity check renders up to 1,000 silhouettes per candidate
object.

## Output

```
bed_voxels/
├── manifest.csv
└── samples/sample_NNNNN/
    ├── query{1,2}.png                    colored
    ├── query{1,2}_axes.png               colored + coordinate axes
    ├── query{1,2}_uncolored.png          uniform gray
    ├── query{1,2}_uncolored_axes.png     uniform gray + coordinate axes
    ├── pos/{xy,yz,zx}.png                colored canonical projections (frontmost block's color)
    ├── pos_uncolored/{xy,yz,zx}.png
    ├── neg/{xy,yz,zx}.png                projection distractors, colored from the nearest positive cell
    └── neg_uncolored/{xy,yz,zx}.png

bed_scenes/
├── manifest.json                         configuration, seed, primitives per scene, distractor IoU per plane
└── sample_NNNN/
    ├── scene.json                        primitives: kind, corner (x, y, z), extents (sx, sy, sz), axis, color
    ├── queries/view_{1,2}.png            colored query views, (azimuth, elevation) = (45°, 30°), (150°, 40°)
    ├── queries_uncolored/view_{1,2}.png  uniform gray
    ├── pos/{xy,yz,zx}.png                canonical projections of the scene
    ├── pos_uncolored/{xy,yz,zx}.png
    ├── neg/{xy,yz,zx}.png                projections of each plane's scene distractor (same view limits as pos/)
    ├── neg/{xy,yz,zx}_scene.json         the distractor scenes
    └── neg_uncolored/{xy,yz,zx}.png
```

Main voxel manifest columns: query views (`query{1,2}_{az,el}`), geometry (`base_blocks_json`),
positive and distractor cells per plane, `colors_json`, and task targets (`task_volume`,
`box_dims_json`, `fits_box`, `color_total`, `color_distribution_json`). The remaining columns record
the filter settings.

## Task datasets

`build_tasks.py` writes one `datasets.save_to_disk` directory per evaluated condition (`T1_uncolor`,
`T5_color`, `T11_invalid_uncolor`, ...; the variant list is in `../harness/README.md`).

- T1, T2, T4–T10, and T12: 100 questions, one per object.
- T3: 100 object pairs, drawn with `random.Random(42)` from all pairs of the 100 objects.
- T11 (objects) and T13 (scenes): 300 questions per subtask, each object or scene with each plane as
  the target. The invalid subtask shows two correct projections and the target plane's distractor
  (find the distractor); the valid subtask shows two distractors and the correct target projection.
- Candidate order in T11, T12, and T13 is drawn from `random.Random(42)`.

Built from regenerated pools, all 34 configurations match the evaluated datasets row for row: every
answer and question field exactly, every image pixel for pixel.

## Code map (paper Appendix C)

| Step | Appendix C | Module |
|---|---|---|
| Grow a connected, supported object (10–15 blocks, \|x\|,\|y\| ≤ 4, 0 ≤ z ≤ 6) | Grow a Connected Voxel Object | `voxels.py` |
| Two query views, ≥ 45° from the canonical axes, footprint checks | Select Two Query Views | `views.py` |
| Reject objects with a block whose removal barely changes both silhouettes | Filter Hidden-Block Objects | `filters.py` |
| Reject objects for which one of 500 sampled alternatives matches both silhouettes (IoU ≥ 0.995) but differs in a projection | Reject Two-View Collisions | `filters.py` |
| Color neighboring blocks with luminance-matched pairs | Assign Adversarial Colors | `colors.py` |
| One-edit projection distractors, filtered against the other planes | Generate Projection Distractors | `distractors.py` |
| Container box and color counts | Construct One Single-Object Sample | `targets.py` |
| Painter's-algorithm rendering of views, projections, silhouettes | (rendering) | `render.py` |
| Sample assembly and manifest | Construct One Single-Object Sample | `pipeline.py` |
| Scene primitives; placement on top of / beside an existing primitive (p = 0.45) or on the floor in [−3, 3]²; reject intersecting placements | Multi-object scenes; Sample and Validate a Multi-Object Scene | `scenes/sampling.py` |
| Scene visibility at 220×220: ≥ 220 visible pixels per primitive in query views, ≥ 180 in canonical views, ≥ 8% of projected area visible | Multi-object scenes | `scenes/raster.py` |
| Scene distractors: swap, delete, move (±1.2/±1.8), resize, lift; non-intersecting, visible in shared limits, silhouette IoU in (0.55, 0.88), 120 attempts per plane | Scene distractors | `scenes/sampling.py` |
| Scene meshes (cylinders as 24-gon prisms), projections, rendering, assembly | (geometry, rendering) | `scenes/geometry.py`, `scenes/render.py`, `scenes/pipeline.py` |
| Task construction | Reusable samples and task construction | `build_tasks.py` |

All parameters live in `config.py` (`GeneratorConfig`) and `scenes/config.py` (`SceneConfig`).

## Notes on exact reproduction

- Both generators consume the random stream in the same order as the code that produced the released
  pools. Some voxel steps therefore keep draws whose outcome is always discarded. For example, the view
  sampler first tries a random second view whose constraints cannot be met when the first view is at
  least 45° from every canonical axis, then falls back to the opposite azimuth with elevation
  `min(e1 + 12°, 55°)`. All released view pairs use this fallback, as described in the paper.
- Scene colors are drawn after a scene passes validation, then distractors plane by plane. The scene
  rasterizer is vectorized per triangle but uses the same floating-point operations as a per-pixel
  loop, so visibility decisions (and therefore the random stream) are unchanged.
- Pixel identity was verified with Python 3.12, NumPy 2.4.4, and Matplotlib 3.10.8 (Agg). Other
  Matplotlib or FreeType versions can change antialiasing slightly while leaving the geometry unchanged.


## T5 separated-tile control

From `generator/` with its requirements installed:

```bash
python -m bed_generator.build_tiles --dataset /path/to/dataset/T5_color --out /path/to/tile_inputs
```

The builder reads a local `datasets.save_to_disk` T5_color dataset. It preserves the original color
dictionary order and shuffles tiles with `random.Random(int(sample_id))`, matching the original
experiment. Each block is one separated square, with the generator's colors, black edges, image
settings, and a gap of 0.6 block units. The builder writes PNGs and a manifest with relative image
paths and unchanged count targets.

Run inference on these images using `bed_eval.run_t5_2d` from `harness/`; see
[additional controls](../harness/controls.md).
