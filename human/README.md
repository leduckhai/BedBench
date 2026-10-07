# Human baseline

`human_baseline_responses.csv` holds one row per participant (30 volunteers). Each participant answered
one randomly drawn question per task under the task's default visual condition.

| Column | Content |
|---|---|
| `participant` | anonymous ID assigned by the study interface |
| `T{k}_item` | row index of the question in the released dataset for task T{k} (see below) |
| `T{k}_answer` | the participant's answer, as JSON (`null` if left blank; scored incorrect) |
| `T{k}_verdict` | `correct` or `incorrect` (exact match with the ground truth) |

Questions were drawn from the following task variants. For T1, T2, T4–T10, and T12 the row index equals
the object's `sample_id`; for T3 it indexes an object pair, and for T11 and T13 it indexes one of the
three projection-plane questions of an object or scene.

| Task | Variant shown to participants |
|---|---|
| T1 | Volume, open-ended, uncolored |
| T2 | Packaging, uncolored |
| T3 | Difference, uncolored with axes |
| T4, T5, T7, T10 | colored |
| T6, T8, T9 | uncolored |
| T11 | Projection, identify the invalid projection, uncolored |
| T12 | Reconstruct, uncolored |
| T13 | Mixture-of-Shapes, identify the invalid silhouette, uncolored |

As stated in the paper's Ethics Statement, the file contains only anonymous per-task answers,
verdicts, and item identifiers. Demographics are reported only in aggregate in the paper; per-person
demographics and session times are not released, and no names, contact details, or IP addresses were
collected.

`python compute_accuracy.py` reproduces the Human row of the main results table and the 95% Wilson
intervals of the human-versus-model comparison.
