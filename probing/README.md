# BedBench probing

Frozen-encoder probing of the BedBench task targets (paper appendix "Probing frozen encoders"): supervised
linear probes on mean-pooled patch features of 14 frozen encoders, plus three training-free consistency
analyses. The encoders are SigLIP-2 so400m, DINOv2-L, CLIP-L, Depth-Anything-V2-L, and the vision
towers inside Qwen3.5 (0.8B, 2B, 4B, 9B, 27B, 35B-A3B) and Gemma4 (E2B, E4B, 26B-A4B, 31B).

## Usage

```bash
pip install -r requirements.txt

# 1. features (or skip: extract extras/probing_features.tar from the BedBench dataset repo and use --cache-dir probing_features)
python -m bed_probing.extract --encoder siglip2 --dataset ../generator/bed_dataset --cache-dir features
python -m bed_probing.extract --vlm Qwen/Qwen3.5-27B --alias qwen27b-tower \
    --dataset ../generator/bed_dataset --cache-dir features --dtype bfloat16

# 2. probes and consistency analyses (CPU)
python -m bed_probing.run --encoders siglip2 dinov2 clip depthanything \
    qwen0.8b-tower qwen2b-tower qwen4b-tower qwen9b-tower qwen27b-tower qwen35b-tower \
    gemma2b-tower gemma4b-tower gemma26b-tower gemma31b-tower \
    --cache-dir features --out-dir results --rep patch_mean --probe linear
```

`--dataset` is the directory written by `python -m bed_generator.build_tasks` (see `../generator`).
Pretrained aliases: `siglip2` (google/siglip2-so400m-patch14-384), `dinov2` (facebook/dinov2-large),
`clip` (openai/clip-vit-large-patch14), `depthanything` (depth-anything/Depth-Anything-V2-Large-hf).
Tower aliases name the VLM checkpoint they come from: `qwen27b-tower` is Qwen/Qwen3.5-27B,
`gemma31b-tower` is google/gemma-4-31B-it, and so on. Only the tower's weight shards are downloaded.

Output: `results/<encoder>_patch_mean_linear.json` (all metrics) and `results/summary_patch_mean_linear.md`.

## What is computed

| Analysis | Module | Tasks |
|---|---|---|
| Linear probes (ridge / logistic, standardized features) from both query views, 5-fold CV grouped by object, 1,000 label permutations for the null | `targets.py`, `probes.py` | T1, T2, T4–T10 (all analyzed visual conditions pooled), T3 (object-disjoint pairs) |
| Cross-view retrieval: is view 2 of the same object the nearest view-2 embedding of view 1? | `consistency.py` | T1 conditions |
| Embedding distance vs. block difference of object pairs (Spearman) | `consistency.py` | T3 |
| Zero-shot candidate ranking by cosine similarity to the reference views | `consistency.py` | T11, T12, T13 (default, uncolored) |

## Reproducibility

- **Folds.** `bed_probing/folds.json` stores the fold of every object (pair for T3) used in the paper's
  run. scikit-learn's `GroupKFold` breaks ties between equal-size groups with a platform-dependent sort,
  so recomputing folds on another CPU architecture gives different splits; the saved folds make the
  probe results platform-independent.
- **Features.** `extras/probing_features.tar` in the BedBench dataset repo holds the cached features of the
  paper's run (921 MB; unpacks to `probing_features/`, pass it as `--cache-dir`). Running the probes on
  them reproduces the paper's probe results exactly for all 14 encoders (observed metrics, baselines,
  and consistency analyses; for SigLIP-2 also the 1,000-permutation null means and 95th percentiles).
  Re-extracting CLIP-L features from datasets rebuilt with `../generator`
  (macOS arm64, MPS, float32) reproduces the cached features bit for bit (checked on `T1_uncolor` and
  `T11_invalid_uncolor`); other hardware or numeric precision can give slightly different features.


## Token-level and connector probes

[Token probe instructions](token_probes.md) cover `bed_probing.extract_tokens` and
`bed_probing.probe_tokens`: Qwen/Gemma tower tokens, Qwen connector output, attention and token-sum
readouts, and training-size curves. These use the same package as the pooled-feature probes above.

Offline checks (from `probing/`): `python -m unittest discover -s tests -v`.
