# Prompts

Every prompt used in the evaluation, verbatim (these files are also rendered in the paper's Prompts
appendix). `non_thinking/` holds the concise task prompts; `thinking/` holds the same tasks with
task-specific decomposition instructions. Placeholders in double braces (e.g. `{{box_dims}}`) are
filled at inference time.

File names use the paper's task numbers (`Task_11.md` is T11, and so on).

| File | Task | Used for |
|---|---|---|
| `Task_1.md` | T1: Volume (open-ended) | main evaluation |
| `Task_1_mcq.md` | T1: Volume (five-option MCQ) | answer-format control |
| `Task_2.md` | T2: Packaging | main evaluation |
| `Task_3.md` | T3: Difference | main evaluation |
| `Task_3_color_aware.md` | T3: Difference, color-aware variant | visual-cue comparison |
| `Task_4.md` | T4: Color Count | main evaluation |
| `Task_5.md` | T5: Color Distribution | main evaluation |
| `Task_5_single_color_extraction.md` | T5: Color Distribution (Single Color Extraction), non-thinking only | factorized querying |
| `Task_6.md` | T6: Local Structure | main evaluation |
| `Task_7.md` | T7: Color Dominance | main evaluation |
| `Task_8.md` | T8: Silhouette Symmetry | main evaluation |
| `Task_9.md` | T9: Visible Floors | main evaluation |
| `Task_10.md` | T10: Color Connected | main evaluation |
| `Task_11.md` | T11: Projection, identify the invalid projection | main evaluation |
| `Task_11_detect_positive.md` | T11: Projection, identify the valid projection | main evaluation |
| `Task_12.md` | T12: Reconstruct | main evaluation |
| `Task_13.md` | T13: Mixture-of-Shapes, identify the invalid silhouette | main evaluation |
| `Task_13_detect_positive.md` | T13: Mixture-of-Shapes, identify the valid silhouette | main evaluation |


Additional control prompts in `non_thinking/`, copied verbatim from the experiments:

| File | Control |
|---|---|
| `Task_5_no_example.md` | T5 with the example removed |
| `Task_5_valid_example.md` | T5 with the 11-block example |
| `Task_5_2d.md` | T5 on separated tiles |
| `Task_7_counts.md` | T7 from explicit counts (`{{counts}}` placeholder) |

See [harness controls](../harness/controls.md) for execution.
