# BedBench: code release

Code for *Where Do Vision-Language Models Fail at 3D Reasoning?* BedBench poses 13 tasks (T1–T13) on rendered
views of 100 procedurally generated voxel objects (T1–T12) and 100 multi-object scenes (T13).

The data live in the Hugging Face dataset repository [`leduckhai/BedBench`](https://huggingface.co/datasets/leduckhai/BedBench):

| In `leduckhai/BedBench` | Contents |
|---|---|
| configurations `T1_uncolor` … `T13_valid_color` | the 34 evaluation sets (one task under one visual condition) |
| configuration `T5_finetune` | the T5 fine-tuning pool (800 train / 100 validation objects) |
| `extras/samples.tar.gz` | the generator output: 100 objects with all targets and rendered views, 100 scenes |
| `extras/outputs.tar.gz` | main evaluation and original control responses, with ground truth |
| `extras/probing_features.tar` | cached features of the 14 frozen encoders |

## Layout

| Directory | Purpose |
|---|---|
| [`generator/`](generator) | regenerates objects, scenes, evaluation datasets (seed 42), and the 2D tile control |
| [`prompts/`](prompts) | main evaluation and additional control prompts, verbatim |
| [`harness/`](harness) | queries a model through an OpenAI-compatible endpoint and scores the answers as in the paper; vLLM serving settings for the open-weight models |
| [`probing/`](probing) | frozen-encoder feature extraction, pooled and token-level probes, and embedding consistency analyses |
| [`finetune/`](finetune) | QLoRA fine-tuning of Qwen3.5-27B on T5 |
| [`human/`](human) | the 30-participant human baseline responses and the script that scores them |
| [`traces/`](traces) | the 65 annotated reasoning traces of the trace audit, with a browser viewer |

Each directory has its own README and `requirements.txt`. The paper's runs used Python 3.12.

## Reproduce the paper's numbers (no GPU)

```bash
cd harness && pip install -r requirements.txt
huggingface-cli download leduckhai/BedBench extras/outputs.tar.gz --repo-type dataset --local-dir ..
tar xzf ../extras/outputs.tar.gz -C ..
python -m bed_eval score ../outputs --json summary.json
```

`bed_eval score` prints the 5-run mean ± SD accuracy of every model, mode, task, and condition, and
`summary.json` holds all metrics. These cover the exported base evaluations and original controls; see
`outputs/README.md` for the directory layout and the conditions covered.

## Evaluate a model

```bash
cd harness
python -m bed_eval run --task T5 --mode thinking --model Qwen/Qwen3.5-27B \
    --api-url http://localhost:8000 --dataset leduckhai/BedBench --runs 5 --out results
python -m bed_eval score results
```

Omitting `--variant` runs the task's default visual condition; [`harness/README.md`](harness/README.md)
lists the conditions and the controls (`--format mcq`, `--format single_color`, `--no-images`).
Scoring counts an answer that does not parse as incorrect.

Additional controls are documented in [`harness/controls.md`](harness/controls.md), and token probes
in [`probing/token_probes.md`](probing/token_probes.md). Their raw outputs
and post-hoc analysis scripts are not included in the existing outputs export.

## Other parts

- **Data.** `python -m bed_generator --n 100 --seed 42`, `python -m bed_generator.scenes --n 100 --seed 42`,
  and `python -m bed_generator.build_tasks` rebuild `extras/samples` and every configuration
  pixel-identically ([`generator/README.md`](generator/README.md)).
- **Probing.** `extras/probing_features.tar` unpacks to `probing_features/`; pass it as `--cache-dir` to
  rerun the probes on CPU without re-extracting features ([`probing/README.md`](probing/README.md)).
- **Fine-tuning.** `python finetune/train.py --dataset leduckhai/BedBench --output-dir t5_lora` trains on the
  `T5_finetune` configuration ([`finetune/README.md`](finetune/README.md)).
- **Human baseline and traces.** Self-contained in `human/` and `traces/`.

## Developer

Main developer: [Nguyen Kim Hai Bui](https://whynotkimhari.github.io) ([GitHub](https://github.com/whynotkimhari)).

## License

Code: Apache-2.0. Data and human responses: CC BY 4.0.
