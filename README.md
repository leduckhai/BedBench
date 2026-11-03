<div align="center">

<a href="https://leduckhai.github.io/BedBench/">
  <img src="https://arxiv.org/html/2610.39639v1/images/BedBench_icon.png" alt="Official BedBench purple bed icon" width="48" />
</a>

<strong>Where Do Vision Language Models Fail at 3D Reasoning?</strong>
--------

<sub>A controlled benchmark for exact and consistent 3D reasoning in vision-language models</sub>

<br />

<sub>
<a href="https://scholar.google.com/citations?user=DfAzEe0AAAAJ">Khai Le-Duc</a><sup>1,2,3,★</sup> ·
<a href="https://scholar.google.com/citations?user=i8k5OBYAAAAJ">Nguyen Kim Hai Bui</a><sup>1,4,★</sup> ·
<a href="https://scholar.google.com.vn/citations?user=iBqHhg4AAAAJ">An Ngo</a><sup>1,5</sup> ·
<a href="https://scholar.google.com/citations?user=6FYcbj0AAAAJ">Nam D. Hoang</a><sup>†</sup>
<br />
<a href="https://scholar.google.com/citations?user=N_MVG4UAAAAJ">Yueqi Song</a><sup>6</sup> ·
<a href="https://scholar.google.com/citations?user=mSIhZTAAAAAJ">Jiarui Liu</a><sup>6</sup> ·
<a href="https://scholar.google.com/citations?user=3v-x1aQAAAAJ">Thai Quoc Hoang</a><sup>7</sup> ·
<a href="https://scholar.google.com/citations?user=8vHkc5YAAAAJ">Fang Wu</a><sup>8</sup> ·
<a href="https://scholar.google.com/citations?user=JF4kESAAAAAJ">Chris Ngo</a><sup>1</sup> ·
<a href="https://scholar.google.com/citations?user=hFY4t8gAAAAJ">Terry Jingchen Zhang</a><sup>3,9</sup>
<br />
<a href="https://scholar.google.com/citations?user=-y6SIhQAAAAJ">Mona T. Diab</a><sup>6</sup> ·
<a href="https://scholar.google.com/citations?user=sgBB2sUAAAAJ">Ran Xu</a><sup>7</sup> ·
<a href="https://scholar.google.com/citations?user=ImpbxLsAAAAJ">Silvio Savarese</a><sup>7,8</sup> ·
<a href="https://scholar.google.com/citations?user=3KbaUFkAAAAJ">Xiao-Ming Wu</a><sup>4</sup>
</sub>

<p align="center"><sub>
<sup>1</sup> Knovel Engineering &nbsp;·&nbsp; <sup>2</sup> University of Toronto &nbsp;·&nbsp; <sup>3</sup> Vector Institute<br />
<sup>4</sup> The Hong Kong Polytechnic University &nbsp;·&nbsp; <sup>5</sup> Texas A&amp;M University<br />
<sup>6</sup> Carnegie Mellon University &nbsp;·&nbsp; <sup>7</sup> Salesforce AI Research&nbsp;·&nbsp; <sup>8</sup> Stanford University<br />
<sup>9</sup> University of Oxford
<br />
<sup>★</sup> Co-first authors &nbsp;·&nbsp; <sup>†</sup> Independent Researcher
</sub></p>

<p>
  <a href="https://arxiv.org/abs/2610.39639"><img alt="arXiv paper" src="https://img.shields.io/badge/arXiv%3A2610.39639-B31B1B?logo=arxiv&logoColor=white&style=flat" /></a>
  <a href="https://leduckhai.github.io/BedBench/"><img alt="Project website" src="https://img.shields.io/badge/Project-Website-B8A0DF?style=flat&labelColor=8B65C9" /></a>
  <a href="https://github.com/leduckhai/BedBench"><img alt="GitHub BedBench" src="https://img.shields.io/badge/GitHub-BedBench-white?logo=github&logoColor=white&style=flat&labelColor=181717" /></a>
  <a href="https://github.com/leduckhai/BedBench"><img alt="GitHub Stars" src="https://img.shields.io/github/stars/leduckhai/BedBench?style=social" /></a>
  <a href="https://huggingface.co/datasets/leduckhai/BedBench"><img alt="Hugging Face dataset" src="https://img.shields.io/badge/%F0%9F%A4%97-Dataset-FFD21E?style=flat&labelColor=D5A500" /></a>
</p>

<sub>
  <a href="#why-bedbench">Overview</a> &nbsp;·&nbsp;
  <a href="#the-bedbench-benchmark">Benchmark</a> &nbsp;·&nbsp;
  <a href="#key-findings">Findings</a> &nbsp;·&nbsp;
  <a href="#code-release">Reproduce</a> &nbsp;·&nbsp;
  <a href="#citation">Cite</a>
</sub>

<br />

<a href="https://leduckhai.github.io/BedBench/">
  <img src="https://leduckhai.github.io/BedBench/static/tasks/T1.png" alt="T1: Volume — count blocks from two 3D views" height="122" />
</a>&nbsp;
<a href="https://leduckhai.github.io/BedBench/">
  <img src="https://leduckhai.github.io/BedBench/static/tasks/T5.png" alt="T5: Color Distribution — exact per-color block counts" height="122" />
</a>&nbsp;
<a href="https://leduckhai.github.io/BedBench/">
  <img src="https://leduckhai.github.io/BedBench/static/tasks/T13.png" alt="T13: Mixture-of-Shapes — multi-object spatial reasoning" height="122" />
</a>

<sub>From individual block counts to exact color distributions and multi-object reasoning · <a href="https://leduckhai.github.io/BedBench/">Explore interactive examples</a></sub>

</div>

> **TL;DR** — BedBench tests whether vision-language models can recover the **exact properties** of a 3D object, use those properties **consistently** across questions, and actually **depend on the visual evidence**. Individual 3D judgments can look strong even when exact structural reasoning fails.

## Why BedBench?

A model can decide whether an object fits inside a box, or identify its dominant color, without accurately reconstructing everything about that object. BedBench separates these easier judgments from **composite answers**—such as the exact number of blocks of *every* color—and checks whether answers to related questions agree.

All tasks have automatically generated ground truth. T1–T12 use rendered views of **100 procedurally generated voxel objects**, while T13 uses **100 multi-object scenes**. The study evaluates **12 open-weight and 4 frontier VLMs**, alongside **30 human participants** and simple baselines.

| **13** tasks | **16** VLMs | **30** human participants |
| :---: | :---: | :---: |
| Exact 3D questions | 12 open-weight + 4 frontier | Human baseline |

## The BedBench Benchmark

Thirteen tasks cover five spatial-reasoning demands. These examples come from the [project website](https://leduckhai.github.io/BedBench/) and show each task's input, goal, and correct answer.

| **Aggregate quantity & support** | **Compositional decomposition** | **Relational structure** |
| :---: | :---: | :---: |
| <a href="https://leduckhai.github.io/BedBench/"><img src="https://leduckhai.github.io/BedBench/static/tasks/T4.png" height="112" alt="T4 · Color Count" /></a> | <a href="https://leduckhai.github.io/BedBench/"><img src="https://leduckhai.github.io/BedBench/static/tasks/T5.png" height="112" alt="T5 · Color Distribution" /></a> | <a href="https://leduckhai.github.io/BedBench/"><img src="https://leduckhai.github.io/BedBench/static/tasks/T10.png" height="112" alt="T10 · Color Connected" /></a> |
| T1 · T2 · T4 · T7 · T9 | T3 · T5 · T6 | T8 · T10 |

| **Cross-view geometry** | **Multi-object scene reasoning** |
| :---: | :---: |
| <a href="https://leduckhai.github.io/BedBench/"><img src="https://leduckhai.github.io/BedBench/static/tasks/T11.png" height="126" alt="T11 · Projection" /></a> | <a href="https://leduckhai.github.io/BedBench/"><img src="https://leduckhai.github.io/BedBench/static/tasks/T13.png" height="126" alt="T13 · Mixture-of-Shapes" /></a> |
| T11 · T12 | T13 |

<details>
<summary><b>View all 13 task examples</b></summary>

<br />

| **T1 · Volume** | **T2 · Packaging** | **T3 · Difference** |
| :---: | :---: | :---: |
| <img src="https://leduckhai.github.io/BedBench/static/tasks/T1.png" height="106" alt="T1 · Volume" /> | <img src="https://leduckhai.github.io/BedBench/static/tasks/T2.png" height="106" alt="T2 · Packaging" /> | <img src="https://leduckhai.github.io/BedBench/static/tasks/T3.png" height="106" alt="T3 · Difference" /> |

| **T4 · Color Count** | **T5 · Color Distribution** | **T6 · Local Structure** |
| :---: | :---: | :---: |
| <img src="https://leduckhai.github.io/BedBench/static/tasks/T4.png" height="106" alt="T4 · Color Count" /> | <img src="https://leduckhai.github.io/BedBench/static/tasks/T5.png" height="106" alt="T5 · Color Distribution" /> | <img src="https://leduckhai.github.io/BedBench/static/tasks/T6.png" height="106" alt="T6 · Local Structure" /> |

| **T7 · Color Dominance** | **T8 · Silhouette Symmetry** | **T9 · Visible Floors** |
| :---: | :---: | :---: |
| <img src="https://leduckhai.github.io/BedBench/static/tasks/T7.png" height="106" alt="T7 · Color Dominance" /> | <img src="https://leduckhai.github.io/BedBench/static/tasks/T8.png" height="106" alt="T8 · Silhouette Symmetry" /> | <img src="https://leduckhai.github.io/BedBench/static/tasks/T9.png" height="106" alt="T9 · Visible Floors" /> |

| **T10 · Color Connected** | **T11 · Projection** | **T12 · Reconstruct** |
| :---: | :---: | :---: |
| <img src="https://leduckhai.github.io/BedBench/static/tasks/T10.png" height="106" alt="T10 · Color Connected" /> | <img src="https://leduckhai.github.io/BedBench/static/tasks/T11.png" height="106" alt="T11 · Projection" /> | <img src="https://leduckhai.github.io/BedBench/static/tasks/T12.png" height="106" alt="T12 · Reconstruct" /> |

<p align="center"><b>T13 · Mixture-of-Shapes</b></p>
<p align="center"><img src="https://leduckhai.github.io/BedBench/static/tasks/T13.png" height="150" alt="T13 · Mixture-of-Shapes" /></p>

</details>

## Key Findings

### 1. Exact color counting is the hardest gap

On **T5 · Color Distribution**, the answer is marked correct only when *every* per-color block count is correct. Humans are far more reliable than the strongest evaluated VLM on this task.

| **93.3%** | **7.2%** | **3.0% → 68.6%** |
| :---: | :---: | :---: |
| Human T5 accuracy | Best VLM T5 accuracy | Qwen3.5-27B T5, before → after fine-tuning |

Fine-tuning on T5 greatly improves exact counting, but the model still struggles to use its own counts consistently when answering related questions (such as T7 · Color Dominance).

### 2. The same blocks are easier to count in 2D than in 3D

For Qwen3.5-27B, displaying the same colored blocks as separate **2D tiles** raises T5 exact-match accuracy from **3.0% in 3D to 52.4% in 2D**.

| 3D rendering · View 1 | 3D rendering · View 2 | 2D tiles · Same blocks |
| :---: | :---: | :---: |
| <img src="https://leduckhai.github.io/BedBench/static/t5_2d/query_image_1.png" alt="3D view 1" width="110" /> | <img src="https://leduckhai.github.io/BedBench/static/t5_2d/query_image_2.png" alt="3D view 2" width="110" /> | <img src="https://leduckhai.github.io/BedBench/static/t5_2d/tiles.png" alt="Same colored blocks separated into 2D tiles" width="110" /> |

### 3. Information can be present without being used

A probe at Qwen3.5-27B's language-model input recovers exact per-color counts on **65.0%** of objects, versus **3.0%** in the VLM's own answers. And image-removal controls show that some correct-looking answers are achievable without any image. These results motivate reporting **exact correctness, consistency, and visual dependence**, rather than only aggregate accuracy.

<p align="center">
  <a href="https://leduckhai.github.io/BedBench/"><b>Explore the figures, results, and interactive examples on the project website →</b></a>
</p>

---

## Code release
BedBench poses 13 tasks (T1–T13) on rendered
views of 100 procedurally generated voxel objects (T1–T12) and 100 multi-object scenes (T13).

The data live in the Hugging Face dataset repository [`leduckhai/BedBench`](https://huggingface.co/datasets/leduckhai/BedBench):

| In `leduckhai/BedBench` | Contents |
|---|---|
| configurations `T1_uncolor` … `T13_valid_color` | the 34 evaluation sets (one task under one visual condition) |
| configuration `T5_finetune` | the T5 fine-tuning pool (800 train / 100 validation objects) |
| `extras/samples.tar.gz` | the generator output: 100 objects with all targets and rendered views, 100 scenes |
| `extras/outputs.tar.gz` | main evaluation and original control responses, with ground truth |
| `extras/probing_features.tar` | cached features of the 14 frozen encoders |

### Layout

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

### Reproduce the paper's numbers (no GPU)

```bash
cd harness && pip install -r requirements.txt
huggingface-cli download leduckhai/BedBench extras/outputs.tar.gz --repo-type dataset --local-dir ..
tar xzf ../extras/outputs.tar.gz -C ..
python -m bed_eval score ../outputs --json summary.json
```

`bed_eval score` prints the 5-run mean ± SD accuracy of every model, mode, task, and condition, and
`summary.json` holds all metrics. These cover the exported base evaluations and original controls; see
`outputs/README.md` for the directory layout and the conditions covered.

### Evaluate a model

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

### Other parts

- **Data.** `python -m bed_generator --n 100 --seed 42`, `python -m bed_generator.scenes --n 100 

## Citation

If BedBench is useful for your research, please cite:

```bibtex
@article{leduc2026bedbench,
  title   = {Where Do Vision Language Models Fail at 3D Reasoning?},
  author  = {Le-Duc, Khai and Bui, Nguyen Kim Hai and Ngo, An and Hoang, Nam D.
             and Song, Yueqi and Liu, Jiarui and Hoang, Thai Quoc and Wu, Fang
             and Ngo, Chris and Zhang, Terry Jingchen and Diab, Mona T.
             and Xu, Ran and Savarese, Silvio and Wu, Xiao-Ming},
  journal = {arXiv preprint arXiv:2610.39639},
  year    = {2026}
}
```
