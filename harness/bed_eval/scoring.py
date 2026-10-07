"""Answer parsing and metrics. Unparseable answers count as incorrect for accuracy; secondary metrics
(RMSE, MAE, precision, recall, F1) are computed over parsed answers. Condition scores are the mean (and
sample SD) of per-run metrics."""
from __future__ import annotations

import json
from typing import Any, Optional

from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, recall_score)

PALETTE = ["yellow", "cyan", "green", "magenta", "red", "orange", "orchid", "teal", "slategray", "gold", "blue", "purple"]
T3_FIELDS = ("answer_intersection", "answer_only_a", "answer_only_b")


def _json(content: str) -> Optional[dict]:
    try:
        value = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _int(value) -> Optional[int]:
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _candidate_from_text(content: str) -> Optional[int]:
    """T11: the JSON object spanning the first '{' to the last '}' (trailing text allowed)."""
    s = (content or "").strip()
    start, end = s.find("{"), s.rfind("}")
    obj = _json(s[start:end + 1]) if start != -1 and end > start else None
    value = obj.get("candidate_idx") if obj else None
    if isinstance(value, bool):
        return None
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return value if isinstance(value, int) else None


def parse(task: str, content: str) -> Any:
    """Parsed answer, or None if the response does not follow the task's JSON format."""
    if task == "T11":
        return _candidate_from_text(content)
    d = _json(content or "")
    if d is None:
        return None
    if task == "T1":
        return _int(d["total_blocks"]) if "total_blocks" in d else None
    if task == "T3":
        values = [_int(d.get(k)) for k in T3_FIELDS]
        return None if None in values else dict(zip(T3_FIELDS, values))
    if task in ("T12", "T13"):
        v = _int(d.get("candidate_idx")) if "candidate_idx" in d else None
        return v if v in ((0, 1) if task == "T12" else (0, 1, 2)) else None
    if "answer" not in d:
        return None
    value = d["answer"]
    if task in ("T2", "T6", "T8", "T10"):
        return int(value) if isinstance(value, bool) else None
    if task in ("T4", "T9"):
        return _int(value)
    if task == "T5":
        if not isinstance(value, dict):
            return None
        counts = {}
        for color, n in value.items():
            try:
                counts[color] = int(float(n))
            except (ValueError, TypeError):
                pass
        return counts or None
    if task == "T7":
        return [str(x) for x in value] if isinstance(value, list) else None
    raise ValueError(f"unknown task {task}")


def _classification(y_true, y_pred) -> dict:
    kw = {"average": "macro", "zero_division": 0}
    return {"accuracy": accuracy_score(y_true, y_pred), "precision": precision_score(y_true, y_pred, **kw),
            "recall": recall_score(y_true, y_pred, **kw), "f1": f1_score(y_true, y_pred, **kw)}


def _regression(y_true, y_pred) -> dict:
    return {"rmse": mean_squared_error(y_true, y_pred) ** 0.5, "mae": mean_absolute_error(y_true, y_pred)}


def run_metrics(task: str, targets: list, preds: list, mcq: bool = False) -> dict:
    """Metrics of one run over its parsed answers (targets[i] is the ground truth of preds[i])."""
    if not preds:
        return {}
    exact = sum(t == p for t, p in zip(targets, preds)) / len(preds)
    if task == "T1":
        return _classification(targets, preds) if mcq else {"accuracy": exact, **_regression(targets, preds)}
    if task in ("T4", "T9"):
        return {"accuracy": exact, **_regression(targets, preds)}
    if task == "T3":
        per_field = [_regression([t[k] for t in targets], [p[k] for p in preds]) for k in T3_FIELDS]
        return {"accuracy": exact, "rmse": _avg([m["rmse"] for m in per_field]), "mae": _avg([m["mae"] for m in per_field])}
    if task == "T5":
        vec = lambda d: [d.get(c, 0) for c in PALETTE]
        return {"accuracy": exact, **_regression([vec(t) for t in targets], [vec(p) for p in preds])}
    if task == "T7":
        # Accuracy: exact set match (order-free, no missing or extra colors, case-insensitive).
        # Precision/recall/F1: micro-averaged over palette indicator vectors.
        norm = lambda colors: frozenset(str(c).strip().lower() for c in colors)  # noqa: E731
        ind = lambda colors: [int(c in set(colors)) for c in PALETTE]  # noqa: E731
        yt, yp = [ind(t) for t in targets], [ind(p) for p in preds]
        kw = {"average": "micro", "zero_division": 0}
        exact_set = sum(norm(t) == norm(p) for t, p in zip(targets, preds)) / len(preds)
        return {"accuracy": exact_set, "precision": precision_score(yt, yp, **kw),
                "recall": recall_score(yt, yp, **kw), "f1": f1_score(yt, yp, **kw)}
    return _classification(targets, preds)  # T2, T6, T8, T10 (binary); T11–T13 (candidate index)


def score_run(records: list[dict], mcq: bool = False) -> dict:
    """Parse the records of one run and compute its metrics; reports parse failures."""
    task = records[0]["task"]
    targets, preds, failures = [], [], 0
    for r in records:
        p = parse(task, r.get("content"))
        if p is None:
            failures += 1
            continue
        targets.append(r["target"])
        preds.append(p)
    metrics = run_metrics(task, targets, preds, mcq)
    # Unparseable answers are incorrect: rescale accuracy from the parsed answers to all records.
    metrics["accuracy"] = metrics.get("accuracy", 0.0) * len(preds) / len(records)
    return {**metrics, "n": len(records), "parse_errors": failures}


def _avg(values: list) -> float:
    return sum(values) / len(values)


def aggregate(runs: list[dict]) -> dict:
    """Mean and sample SD of each metric across runs (plain float formulas, as in the paper's tables)."""
    keys = [k for k in runs[0] if k not in ("n", "parse_errors") and all(k in r for r in runs)]
    means = {k: _avg([r[k] for r in runs]) for k in keys}
    stds = {k: (sum((r[k] - means[k]) ** 2 for r in runs) / (len(runs) - 1)) ** 0.5 if len(runs) > 1 else 0.0
            for k in keys}
    return {"mean": means, "std": stds, "n_runs": len(runs)}
