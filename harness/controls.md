# Additional T5/T7 controls

Run these commands from `harness/`, with `pip install -r requirements.txt` and an
OpenAI-compatible endpoint configured as in [serving.md](serving.md). Modules reuse the standard
request payload and read the original control prompts from `../prompts/non_thinking/`.
The standard `python -m bed_eval run` and `score` commands remain unchanged.
Existing control output files are skipped; use a new output directory when changing inputs/settings.

## T5 prompt-example ablation

```bash
python -m bed_eval.run_t5_prompt --dataset /path/to/dataset \
  --model Qwen/Qwen3.5-27B --api-url http://localhost:8000 --out runs/t5_prompt
```

Repeat for served `Qwen/Qwen3.5-9B`, using the original precision and serving settings.
The dataset argument accepts a local directory containing `T5_color/` or a Hugging Face dataset id.
The two prompts remove the example (`no_example`) or replace it with an 11-block example
(`valid_example`). The paper uses 100 objects × 5 runs × 2 prompts per model.
`--prompts no_example` or `--prompts valid_example` selects one condition.
Outputs: `OUT/non_thinking/T5/MODEL/run_N/color_{no_example,valid_example}.jsonl`.
The original prompt comparator is run by the standard harness, not rerun automatically here.

## T5 separated tiles

Generate the inputs from `generator/` first:

```bash
python -m bed_generator.build_tiles --dataset /path/to/dataset/T5_color --out /path/to/tile_inputs
```

Then, from `harness/`:

```bash
python -m bed_eval.run_t5_2d --inputs /path/to/tile_inputs/rows.jsonl \
  --model Qwen/Qwen3.5-27B --api-url http://localhost:8000 --out runs/t5_2d
```

This sends one tile image instead of two 3D views. The prompt changes only its first sentence;
the Non-Thinking decoding and T5 schema match the baseline. Default: 100 objects × 5 runs.
Outputs: `OUT/non_thinking/T5/MODEL/run_N/color_2d.jsonl`. The saved 3D runs are the comparator.

## T7 from counts

```bash
python -m bed_eval.build_count_inputs --outputs /path/to/outputs --out inputs/t7_counts.jsonl
python -m bed_eval.run_t7_counts --inputs inputs/t7_counts.jsonl \
  --model Qwen3.5-27B-T5-LoRA --api-url http://localhost:8000 --out runs/t7_counts
```

Serve the merged T5-fine-tuned model (see [fine-tuning](../finetune/README.md)). The builder reads
`non_thinking/{T5,T7}/MODEL/run_N/{color,color_axes}.jsonl` from the released outputs, pairing
records by object id and rejecting duplicates or mismatches. Its `--model` selects the source
directory, default `Qwen3.5-27B-T5-LoRA`; the runner's `--model` selects the endpoint model name.

Five runs generate 1,000 self-count inputs (both visual variants) and 500 oracle-count inputs.
The records retain object id, run, variant, source counts, T5 target and T7 target. Unparseable
self-counts stop input construction. The runner supplies text only, using the T7 Non-Thinking
schema and decoding settings. Outputs:
`OUT/self_count/{color,color_axes}/run_N.jsonl` and `OUT/oracle_count/gt/run_N.jsonl`.
The direct visual T7 comparator already exists in the released fine-tuning outputs.

## Coverage

These modules prepare inputs and run inference. New raw control outputs have not yet been added
to the existing outputs archive. Post-hoc comparisons, conditional accuracy, gold/yellow merging,
bootstrap intervals and other analysis scripts are deferred. Token extraction and probe training
are in [probing/](../probing/token_probes.md), with native training/test metrics.
