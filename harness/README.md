# BedBench evaluation harness

Queries a model through any OpenAI-compatible endpoint (vLLM for the open-weight models, see
[serving.md](serving.md)), saves one JSONL record per question, and scores the results the same way as
the paper.

```bash
pip install -r requirements.txt

# one condition, five runs
python -m bed_eval run --task T5 --mode thinking --model Qwen/Qwen3.5-27B \
    --api-url http://localhost:8000 --dataset DATASET --runs 5 --out results

# mean ± SD over runs for everything under results/
python -m bed_eval score results --json summary.json

python -m unittest tests.test_end_to_end       # offline check with a fake server
```

`--dataset` is the Hugging Face dataset id or a local directory saved with `datasets.save_to_disk`
(one subdirectory per configuration, e.g. `DATASET/T5_color`).

## Conditions

| Option | Values |
|---|---|
| `--task` | `T1` … `T13` |
| `--variant` | visual condition: the default setting (first below) or one visual add-on (color or axes), as in the paper |
| `--mode` | `non_thinking`, `thinking` (default reasoning budget 1024 tokens, `--thinking-budget N`; the budget sweep's 0-token point is `--mode non_thinking`) |
| `--format mcq` | T1 five-option answer-format control (correct option always third) |
| `--format single_color` | T5 factorized querying (one query per palette color, Non-Thinking prompts only) |
| `--no-images` | image-removal control |

| Task | Variants (default first) |
|---|---|
| T1 | `uncolor`, `uncolor_axes`, `color` |
| T2, T6, T8, T9 | `uncolor`, `uncolor_axes` |
| T3 | `uncolor_axes`, `color_axes` (color uses the color-aware prompt and targets) |
| T4, T5, T7, T10 | `color`, `color_axes` |
| T11 | `invalid_` or `valid_` × `uncolor`, `uncolor_axes`, `color` |
| T12 | `uncolor`, `uncolor_axes`, `color` |
| T13 | `invalid_uncolor`, `invalid_color`, `valid_uncolor`, `valid_color` |

Results are written to `OUT/{mode}/{task}/{model}/run_{k}/{condition}.jsonl`. Each record holds the
request text (images omitted), the full server response, the answer `content`, and the ground-truth
`target`, so scoring needs no dataset access.

## Requests

Every request uses greedy decoding (`temperature=0`, `top_k=1`), `max_tokens` 8192 (4096 for T5), and
vLLM guided decoding with a task-specific JSON schema (`response_format`). Images are sent before the
prompt, except in T11, where the prompt and a short image legend come first. Prompts are read from
`../prompts/{mode}/`.

## Scoring

An answer that does not parse as the task's JSON answer counts as incorrect and is reported as
`parse_errors`. Accuracy is exact match over all questions (T5: the full color-count dictionary; T7:
the exact set of dominant colors, order-free; T3: all three counts). Secondary metrics are computed over parsed answers, as in the paper: RMSE and MAE for count tasks (T1, T3, T4, T5, T9), and macro
precision, recall, and F1 for decision and candidate tasks. A condition's score is the mean over runs,
with the sample SD.

## Verification

Re-scoring all saved open-weight results with this code reproduces all 7,488 values in the paper's
metric files exactly (accuracies and secondary metrics, means and SDs, for every model, mode, task,
and visual condition).
Rebuilding the request text for 11,400 saved requests (every supported condition) reproduces the stored prompts exactly,
including MCQ options, container sizes, and the T11 image legend.

The released model outputs (`outputs/`, same layout; see its README) cover the main evaluation and original controls,
including the frontier models, the image-removal and answer-format controls, the reasoning-budget sweep,
factorized T5 querying, and the fine-tuned model. Additional controls are documented in [controls.md](controls.md); their outputs and post-hoc
analyses are separate. `bed_eval score` skips factorized single-color records, which need a separate
merge-and-score analysis.


## Additional controls

[controls.md](controls.md) documents the T5 prompt-example ablation, T5 separated-tile inference,
and T7 from self-predicted or oracle counts. All use modules in this `bed_eval` package and prompts
from `../prompts/non_thinking/`. Offline checks: `python -m unittest discover -s tests -v`.
