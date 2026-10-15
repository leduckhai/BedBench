# Frozen-token probes for T5

From `probing/`, install dependencies and extract tokens:

```bash
pip install -r requirements.txt
python -m bed_probing.extract_tokens --dataset /path/to/dataset \
  --vlm Qwen/Qwen3.5-27B --alias qwen27b-tower --out tokens
python -m bed_probing.extract_tokens --dataset /path/to/dataset \
  --vlm google/gemma-4-31B-it --alias gemma31b-tower --out tokens
python -m bed_probing.extract_tokens --dataset /path/to/dataset \
  --vlm Qwen/Qwen3.5-27B --alias qwen27b-connector --stage connector --out tokens
```

The local dataset directory must contain `T5_finetune/` (800 train, 100 validation) and `T5_color/`
(100 benchmark test objects). The extractor reuses `bed_probing`'s model loading and device selection;
`CUDA_VISIBLE_DEVICES` can select a GPU. It defaults to float32 extraction and stores float16 tokens.
Only the visual weights are loaded. An optional `--paper-features /path/to/probing_features` checks
tower token means against the previously cached pooled features. No such reference exists for the
connector. Existing split files are skipped; choose a new output folder when changing settings.

Qwen tower: 676 × 1152 tokens per view; Gemma tower: 256 × 1152. Qwen connector: 169 × 5120,
read from the vision model's `pooler_output`. Extraction requires a Transformers version supporting
these model classes and the connector output field; GPU extraction was not rerun during packaging.
Saved splits are `tokens/ALIAS/{train,val,test}.pt`, containing both views, count vectors and sample ids.

Train the full-size probes:

```bash
python -m bed_probing.probe_tokens --tokens-dir tokens --out-dir runs/token_probe \
  --encoders qwen27b-tower gemma31b-tower --device cuda --probes mean_ridge attn token_sum
python -m bed_probing.probe_tokens --tokens-dir tokens --out-dir runs/token_probe \
  --encoders qwen27b-connector --device cuda --probes mean_ridge attn
```

For the reported learning curves:

```bash
for n in 200 400; do
  python -m bed_probing.probe_tokens --tokens-dir tokens --out-dir runs/token_probe \
    --encoders qwen27b-tower gemma31b-tower qwen27b-connector \
    --device cuda --probes mean_ridge attn --n-train "$n"
done
```

`--device cpu` or `--device mps` is also supported. Neural probes use three seeds by default,
300 epochs maximum, early stopping after 30 epochs without validation improvement, and the original
smooth-L1 objective. Attention uses eight learned queries; token-sum uses a shared per-token MLP
with nonnegative outputs and the corrected final-layer bias initialization (-7).
RidgeCV fits the combined training and validation examples; neural probes only fit training objects.
`--n-train` selects the first N training objects while retaining the 100 validation objects.

Outputs: `results.json`, `results_n200.json`, `results_n400.json` under `--out-dir`, including native
probe test metrics and three-seed means/SDs. Existing JSON entries for other probes are retained.
The original small-data token-sum runs collapsed before the initialization fix and are not reported;
the learning-curve commands intentionally run ridge and attention only. No token tensors, trained
checkpoints, or historical failed-run results are bundled with this code. Table/figure aggregation
scripts are deferred.
