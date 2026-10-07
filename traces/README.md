# Annotated reasoning traces

**To read the traces, open `viewer.html` in any browser** (self-contained, no server needed). It shows each task's input images, prompt, and ground truth, and every model's trace with the labeled spans highlighted; labels can be toggled, and the arrow keys move between traces. `python build_viewer.py` regenerates it from `annotated_traces.json`, `inputs/`, and `../prompts/thinking/`.

`annotated_traces.json` contains the 65 reasoning traces of the trace audit (one sampled question per
task; for each, the trace of one open-weight model and the four frontier models), with all 457
hand-labeled spans. The paper's qualitative appendix shows 15 of these traces; this file is the
complete set on which the reported span counts are based.

Each entry:

| Field | Content |
|---|---|
| `trace_id` | `{task}_{model}` |
| `task` | T1–T13 |
| `item` | row of the question in the task's default released configuration (an object pair `a_vs_b` for T3) |
| `model`, `model_type` | model name; `open-weight` or `frontier` |
| `trace_source` | `raw thinking tokens` (open-weight); `reasoning summary returned by the API` (Grok-4.20); `meta-reasoning summary collected manually through the chat interface` (GPT-5, Gemini-3-Flash, Gemini-3.1-Pro, whose evaluation endpoints did not expose reasoning) |
| `ground_truth`, `model_answer` | parsed answers |
| `correct` | whether the scored answer is correct |
| `text` | the trace exactly as annotated |
| `spans` | `{start, end, label}` character offsets into `text` (Python string indices, end exclusive) |

## Span labels

| Label | Meaning |
|---|---|
| Tool Use | invokes code execution to inspect pixel values or crop the image |
| Good Perception | the stated reading of the scene is correct |
| Good Tactic | a valid problem-solving strategy or check, applied correctly |
| Good Intuition | a correct-magnitude estimate reached by heuristic guess rather than derivation |
| Reiterating | restates or redoes a step already computed, with no new evidence |
| Prior Bias | anchors the answer on the prompt's example or output template |
| Wrong Perception | the stated reading of the scene is incorrect |
| Color Confusion | confuses two similar or same-family colors |
| Harnessing | the reasoning derails into the system's own meta-instructions |
| Misc | uninterpretable artifacts, e.g. a stray non-English token |
| Budget Hit | the trace is cut at the token limit before reaching a decision |

Labels apply to individual spans; most text carries no label. The paper's two whole-trace verdicts
(Correct by Luck, Think–Answer Mismatch) were assigned in a separate reading and are not stored here.

Example:

```python
import json
traces = json.load(open("annotated_traces.json"))
t = traces[0]
for s in t["spans"]:
    print(s["label"], "|", t["text"][s["start"]:s["end"]][:80])
```
