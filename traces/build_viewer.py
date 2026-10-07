"""Build viewer.html: a self-contained page for reading the annotated traces.

Usage: python build_viewer.py   (stdlib only; reads annotated_traces.json, inputs/, ../prompts/thinking/)
"""
import base64
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROMPTS = HERE.parent / "prompts" / "thinking"

TASK_NAMES = {
    "T1": "Volume", "T2": "Packaging", "T3": "Difference", "T4": "Color Count", "T5": "Color Distribution",
    "T6": "Local Structure", "T7": "Color Dominance", "T8": "Silhouette Symmetry", "T9": "Visible Floors",
    "T10": "Color Connected", "T11": "Projection", "T12": "Reconstruct", "T13": "Mixture-of-Shapes",
}
# Image order and roles as presented to the models (see the prompts).
ROLE_ORDER = [
    ("query_image_1", "View 1"), ("query_image_2", "View 2"),
    ("query_a_image_1", "Object A, view 1"), ("query_a_image_2", "Object A, view 2"),
    ("query_b_image_1", "Object B, view 1"), ("query_b_image_2", "Object B, view 2"),
    ("silhouette_0", "Silhouette"), ("silhouette_1", "Silhouette"), ("silhouette_2", "Silhouette"),
    ("candidate_0", "Candidate 0"), ("candidate_1", "Candidate 1"), ("candidate_2", "Candidate 2"),
]
LABELS = [  # (label, group, color, meaning)
    ("Good Perception", "good", "#7dd3a0", "The stated reading of the scene is correct."),
    ("Good Tactic", "good", "#5fd4c6", "A valid strategy or check, applied correctly."),
    ("Good Intuition", "good", "#a8c7fa", "A correct-magnitude estimate reached by heuristic guess."),
    ("Tool Use", "good", "#8ecbff", "Runs code to inspect pixel values or crop the image."),
    ("Wrong Perception", "bad", "#ff9e8f", "The stated reading of the scene is incorrect."),
    ("Reiterating", "bad", "#ffd966", "Restates or redoes a step already computed, with no new evidence."),
    ("Color Confusion", "bad", "#f2a6d8", "Confuses two similar or same-family colors."),
    ("Prior Bias", "bad", "#d8b4fe", "Anchors the answer on the prompt's example or output template."),
    ("Harnessing", "bad", "#ffb07c", "Derails into the system's own meta-instructions."),
    ("Misc", "bad", "#c8c8c8", "Uninterpretable artifact, e.g. a stray non-English token."),
    ("Budget Hit", "artifact", "#c9a66b", "Cut at the token limit before reaching a decision."),
]


def task_inputs(task: str) -> list:
    images = []
    for key, role in ROLE_ORDER:
        path = HERE / "inputs" / f"{task}_{key}.png"
        if path.exists():
            data = base64.b64encode(path.read_bytes()).decode()
            images.append({"role": role, "src": f"data:image/png;base64,{data}"})
    for i, image in enumerate(images, 1):
        image["caption"] = f"Image {i} · {image['role']}"
    return images


def main() -> None:
    traces = json.loads((HERE / "annotated_traces.json").read_text())
    tasks = {}
    for t in traces:
        task = t["task"]
        if task not in tasks:
            prompt = PROMPTS / f"Task_{task[1:]}.md"
            tasks[task] = {"name": TASK_NAMES[task], "item": t["item"], "ground_truth": t["ground_truth"],
                           "prompt": prompt.read_text(), "images": task_inputs(task), "traces": []}
        tasks[task]["traces"].append({k: t[k] for k in ("trace_id", "model", "model_type", "trace_source",
                                                        "model_answer", "correct", "text", "spans")})
    order = sorted(tasks, key=lambda k: int(k[1:]))
    payload = {"tasks": [{"id": k, **tasks[k]} for k in order],
               "labels": [{"name": n, "group": g, "color": c, "meaning": m} for n, g, c, m in LABELS]}
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    html = (HERE / "viewer_template.html").read_text().replace("__DATA__", data)
    (HERE / "viewer.html").write_text(html, encoding="utf-8")
    print(f"wrote viewer.html ({len(html) // 1024} KB, {len(traces)} traces)")


if __name__ == "__main__":
    main()
