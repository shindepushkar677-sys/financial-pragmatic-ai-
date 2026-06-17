"""Metric computation helpers for model comparison."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def compute_metrics(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict:
    accuracy = float(accuracy_score(y_true, y_pred))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    macro_precision = float(
        precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )[0]
    )
    macro_recall = float(
        precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )[1]
    )
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_class = {}
    for i, label in enumerate(labels):
        per_class[label] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def delta_metrics(baseline: Dict, ours: Dict) -> Dict[str, float]:
    return {
        "accuracy_delta": float(ours["accuracy"] - baseline["accuracy"]),
        "macro_f1_delta": float(ours["macro_f1"] - baseline["macro_f1"]),
        "macro_precision_delta": float(ours["macro_precision"] - baseline["macro_precision"]),
        "macro_recall_delta": float(ours["macro_recall"] - baseline["macro_recall"]),
    }


def to_numpy_confusion(metrics: Dict) -> np.ndarray:
    return np.asarray(metrics["confusion_matrix"], dtype=np.int64)
