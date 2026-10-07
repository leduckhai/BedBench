"""Per-task specification: dataset variants, prompt files, message layout, answer schema."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

SINGLE_COLORS = ("yellow", "cyan", "green", "magenta", "red", "orange",
                 "orchid", "teal", "slategray", "gold", "blue", "purple")

_INT = {"type": "integer"}
_BOOL = {"type": "boolean"}


def _schema(title: str, props: dict) -> dict:
    """Same JSON schema the original runs sent (pydantic model_json_schema output)."""
    return {"properties": {k: {**v, "title": k.replace("_", " ").title()} for k, v in props.items()},
            "required": list(props), "title": title, "type": "object"}


@dataclass(frozen=True)
class TaskSpec:
    task: str                      # paper id, e.g. "T5"
    variants: tuple[str, ...]      # default setting first, then single visual add-ons (paper Table "visual add-ons")
    images: tuple[str, ...]        # dataset image columns, in the order shown to the model
    schema: dict                   # JSON schema for guided decoding
    prompt: str = ""               # prompt file name (defaults to Task_{n}.md)
    text_first: bool = False       # T11 puts the prompt (and an image legend) before the images
    max_tokens: int = 8192
    fill: Callable[[str, dict], str] = field(default=lambda prompt, row: prompt)

    def prompt_file(self, variant: str, fmt: str = "default") -> str:
        n = self.task[1:]
        if fmt == "mcq":
            return f"Task_{n}_mcq.md"
        if fmt == "single_color":
            return f"Task_{n}_single_color_extraction.md"
        if self.task == "T3" and variant.startswith("color"):
            return "Task_3_color_aware.md"
        if self.task in ("T11", "T13") and variant.startswith("valid"):
            return f"Task_{n}_detect_positive.md"
        return self.prompt or f"Task_{n}.md"

    def config(self, variant: str) -> str:
        """Dataset configuration name of one variant."""
        return f"{self.task}_{variant}"


def _fill_mcq(prompt: str, row: dict) -> str:
    """Five options with the correct volume always third (answer-format control)."""
    options = [row["distractors"][0], row["distractors"][1], row["answer"], row["distractors"][2], row["distractors"][3]]
    for i, value in enumerate(options, 1):
        prompt = prompt.replace(f"{{{{option_{i}_number}}}}", str(value))
    return prompt


def _replace(placeholder: str, column: str) -> Callable[[str, dict], str]:
    return lambda prompt, row: prompt.replace(placeholder, str(row[column])) if placeholder in prompt else prompt


QUERY = ("query_image_1", "query_image_2")
CANDIDATES_3 = ("candidate_0", "candidate_1", "candidate_2")
U, UA, C, CA = "uncolor", "uncolor_axes", "color", "color_axes"

TASKS: dict[str, TaskSpec] = {t.task: t for t in [
    TaskSpec("T1", (U, UA, C), QUERY, _schema("TotalBlocksResponse", {"total_blocks": _INT})),
    TaskSpec("T2", (U, UA), QUERY, _schema("Task2Response", {"answer": _BOOL}), fill=_replace("{{box_dims}}", "box_dims")),
    TaskSpec("T3", (UA, CA), ("query_a_image_1", "query_a_image_2", "query_b_image_1", "query_b_image_2"),
             _schema("Task3Response", {"answer_intersection": _INT, "answer_only_a": _INT, "answer_only_b": _INT})),
    TaskSpec("T4", (C, CA), QUERY, _schema("Task4Response", {"answer": _INT})),
    TaskSpec("T5", (C, CA), QUERY, _schema("Task5Response", {"answer": {"additionalProperties": _INT, "type": "object"}}),
             max_tokens=4096),
    TaskSpec("T6", (U, UA), QUERY, _schema("Task6Response", {"answer": _BOOL}), fill=_replace("{{k}}", "k")),
    TaskSpec("T7", (C, CA), QUERY, _schema("Task7Response", {"answer": {"items": {"type": "string"}, "type": "array"}})),
    TaskSpec("T8", (U, UA), QUERY, _schema("Task8Response", {"answer": _BOOL})),
    TaskSpec("T9", (U, UA), QUERY, _schema("Task9Response", {"answer": _INT})),
    TaskSpec("T10", (C, CA), QUERY, _schema("Task10Response", {"answer": _BOOL})),
    TaskSpec("T11", tuple(f"{s}_{v}" for s in ("invalid", "valid") for v in (U, UA, C)),
             QUERY + CANDIDATES_3, _schema("Task11Response", {"candidate_idx": _INT}), text_first=True),
    TaskSpec("T12", (U, UA, C), ("silhouette_0", "silhouette_1", "silhouette_2", "candidate_0", "candidate_1"),
             _schema("Task12Response", {"candidate_idx": {"enum": [0, 1], "type": "integer"}})),
    TaskSpec("T13", ("invalid_uncolor", "invalid_color", "valid_uncolor", "valid_color"), QUERY + CANDIDATES_3,
             _schema("Task13Response", {"candidate_idx": {"enum": [0, 1, 2], "type": "integer"}})),
]}


def image_legend(n_candidates: int = 3) -> str:
    """T11 image legend, sent after the prompt."""
    cands = ", ".join(f"({3 + k}) 2D silhouette candidate {k}" for k in range(n_candidates))
    return f"Attached images appear in this fixed order: (1) first 3D query view, (2) second 3D query view, {cands}."


def fill_prompt(spec: TaskSpec, template: str, row: dict, fmt: str = "default",
                color: Optional[str] = None) -> str:
    if fmt == "mcq":
        return _fill_mcq(template, row)
    if fmt == "single_color":
        return template.replace("{{color}}", str(color))
    return spec.fill(template, row)
