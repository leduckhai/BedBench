"""Group-aware cross-validated probes with permutation nulls.

All probes are linear by default (ridge / logistic); an optional MLP probe
lower-bounds nonlinear decodability. Group 5-fold CV by `sample_id` prevents the
same underlying object (rendered in several visual variants) from appearing
in both train and test folds. The fold of every group is read from folds.json
(the GroupKFold assignment of the paper's run): GroupKFold breaks ties between
equal-size groups with a platform-dependent sort, so recomputing it on another
CPU architecture can give different folds and slightly different metrics.
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .targets import ProbeData

logger = logging.getLogger(__name__)

N_FOLDS = 5
N_PERMUTATIONS = 1000
RIDGE_ALPHAS = np.logspace(-3, 5, 17)
SEED = 42
FOLDS: Dict[str, Dict[str, int]] = json.loads((Path(__file__).with_name("folds.json")).read_text())


def _splits(task: str, groups: np.ndarray):
    """(train_idx, test_idx) per fold from the saved group -> fold assignment of `task`."""
    fold = np.asarray([FOLDS[task][str(g)] for g in groups])
    for k in range(N_FOLDS):
        yield np.flatnonzero(fold != k), np.flatnonzero(fold == k)


@dataclass(frozen=True)
class ProbeResult:
    """Cross-validated probe metrics plus permutation-null calibration."""

    task: str
    kind: str
    probe: str
    n: int
    n_groups: int
    metrics: Dict[str, float]
    baseline: Dict[str, float]
    null_mean: Dict[str, float]
    null_p95: Dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


def _make_estimator(kind: str, probe: str) -> "object":
    if kind in ("regression", "multioutput"):
        if probe == "mlp":
            model = MLPRegressor(hidden_layer_sizes=(256,), max_iter=2000, random_state=SEED)
        else:
            model = RidgeCV(alphas=RIDGE_ALPHAS)
        return make_pipeline(StandardScaler(), model)
    if probe == "mlp":
        model = MLPClassifier(hidden_layer_sizes=(256,), max_iter=2000, random_state=SEED)
    else:
        model = LogisticRegression(C=1.0, max_iter=5000)
    return make_pipeline(StandardScaler(), model)


def _cv_predict(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                kind: str, probe: str, task: str) -> np.ndarray:
    """Out-of-fold predictions under the saved group folds (scores for binary)."""
    preds = np.zeros_like(y, dtype=np.float64)
    for train_idx, test_idx in _splits(task, groups):
        if kind == "binary" and len(np.unique(y[train_idx])) < 2:
            # Degenerate fold (rare label, e.g. an uncommon T5/T7 color):
            # fall back to the constant class instead of crashing.
            preds[test_idx] = float(y[train_idx][0])
            continue
        est = _make_estimator(kind, probe)
        est.fit(X[train_idx], y[train_idx])
        if kind == "binary":
            preds[test_idx] = est.predict_proba(X[test_idx])[:, 1]
        else:
            preds[test_idx] = est.predict(X[test_idx])
    return preds


def _regression_metrics(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    rho = spearmanr(y, pred).statistic
    return {
        "r2": float(r2_score(y, pred)),
        "spearman": float(0.0 if np.isnan(rho) else rho),
        "mae": float(mean_absolute_error(y, pred)),
    }


def _binary_metrics(y: np.ndarray, score: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(((score >= 0.5).astype(int) == y).mean()),
        "auc": float(roc_auc_score(y, score)) if len(np.unique(y)) > 1 else 0.5,
    }


def _multioutput_metrics(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """T5-style metrics: per-color MAE, rounded exact match, palette accuracy."""
    rounded = np.clip(np.round(pred), 0, None)
    exact = float((rounded == y).all(axis=1).mean())
    palette = float(((rounded > 0) == (y > 0)).all(axis=1).mean())
    return {
        "per_color_mae": float(np.abs(pred - y).mean()),
        "exact_match": exact,
        "palette_accuracy": palette,
    }


def _multilabel_metrics(y: np.ndarray, score: np.ndarray) -> Dict[str, float]:
    pred = (score >= 0.5).astype(int)
    aucs = [roc_auc_score(y[:, j], score[:, j])
            for j in range(y.shape[1]) if len(np.unique(y[:, j])) > 1]
    return {
        "subset_accuracy": float((pred == y).all(axis=1).mean()),
        "mean_auc": float(np.mean(aucs)) if aucs else 0.5,
    }


def _evaluate(data: ProbeData, probe: str, rng: np.random.Generator,
              permute: bool = False) -> Dict[str, float]:
    y = data.y.copy()
    if permute:
        y = y[rng.permutation(len(y))]
    if data.kind in ("multioutput", "multilabel"):
        kind_each = "regression" if data.kind == "multioutput" else "binary"
        preds = np.column_stack([
            _cv_predict(data.X, y[:, j], data.groups, kind_each, probe, data.task)
            for j in range(y.shape[1])
        ])
        fn = _multioutput_metrics if data.kind == "multioutput" else _multilabel_metrics
        return fn(y, preds)
    preds = _cv_predict(data.X, y, data.groups, data.kind, probe, data.task)
    fn = _regression_metrics if data.kind == "regression" else _binary_metrics
    return fn(y, preds)


def _baseline(data: ProbeData) -> Dict[str, float]:
    if data.kind == "regression":
        return {"mae": float(np.abs(data.y - data.y.mean()).mean())}
    if data.kind == "binary":
        return {"accuracy": float(max(data.y.mean(), 1 - data.y.mean()))}
    if data.kind == "multioutput":
        mean_pred = np.tile(data.y.mean(axis=0), (len(data.y), 1))
        return _multioutput_metrics(data.y, mean_pred)
    majority = np.tile((data.y.mean(axis=0) >= 0.5).astype(int), (len(data.y), 1))
    return {"subset_accuracy": float((majority == data.y).all(axis=1).mean())}


def run_probe(data: ProbeData, probe: str = "linear",
              n_permutations: int = N_PERMUTATIONS) -> ProbeResult:
    """Run one probe with permutation-null calibration."""
    rng = np.random.default_rng(SEED)
    metrics = _evaluate(data, probe, rng)
    nulls: List[Dict[str, float]] = [
        _evaluate(data, probe, rng, permute=True) for _ in range(n_permutations)
    ]
    null_mean = {k: float(np.mean([n[k] for n in nulls])) for k in metrics}
    null_p95 = {k: float(np.percentile([n[k] for n in nulls], 95)) for k in metrics}
    result = ProbeResult(
        task=data.task, kind=data.kind, probe=probe, n=len(data.y),
        n_groups=int(len(np.unique(data.groups))),
        metrics=metrics, baseline=_baseline(data),
        null_mean=null_mean, null_p95=null_p95,
    )
    logger.info("%s [%s]: %s (null p95: %s)", data.task, probe, metrics, null_p95)
    return result
