"""Generate evaluation diagrams for finbert_intent_v3."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[4]
BACKEND_ROOT = PROJECT_ROOT / "backend"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from financial_pragmatic_ai.evaluation.better_than_fin.utils import SIGNAL_LABELS
from financial_pragmatic_ai.evaluation.better_than_fin.visualize import (
    save_agreement_bar_chart,
    save_model_comparison_chart,
    save_normalized_confusion_matrix,
    save_per_class_f1_chart,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_RESULTS = ROOT / "better_than_fin" / "results" / "metrics_summary.json"
OUTPUT_DIR = Path(__file__).resolve().parent


def _load_metrics() -> dict:
    with open(SOURCE_RESULTS, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _per_class_ordered(metrics: dict) -> dict:
    return {label: metrics["per_class"][label] for label in SIGNAL_LABELS}


def _save_class_distribution_chart(
    labels: list[str],
    y_true_counts: list[int],
    y_finbert_counts: list[int],
    y_ours_counts: list[int],
    output_path: Path,
) -> None:
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, y_true_counts, width=width, label="Ground Truth", color="#6b7280")
    ax.bar(x, y_finbert_counts, width=width, label="FinBERT", color="#9ca3af")
    ax.bar(x + width, y_ours_counts, width=width, label="Our System", color="#2563eb")
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = _load_metrics()

    finbert = metrics["finbert"]
    ours = metrics["our_system"]

    save_normalized_confusion_matrix(
        labels=SIGNAL_LABELS,
        confusion_matrix=np.asarray(ours["confusion_matrix"], dtype=np.int64),
        title="FinBERT Intent v3 - Our System Confusion Matrix",
        output_path=OUTPUT_DIR / "confusion_matrix_ours_normalized.png",
    )
    save_normalized_confusion_matrix(
        labels=SIGNAL_LABELS,
        confusion_matrix=np.asarray(finbert["confusion_matrix"], dtype=np.int64),
        title="FinBERT Intent v3 - FinBERT Baseline Confusion Matrix",
        output_path=OUTPUT_DIR / "confusion_matrix_finbert_normalized.png",
    )
    save_model_comparison_chart(
        finbert_metrics=finbert,
        our_metrics=ours,
        output_path=OUTPUT_DIR / "model_comparison.png",
    )
    save_per_class_f1_chart(
        labels=SIGNAL_LABELS,
        finbert_metrics={"per_class": _per_class_ordered(finbert)},
        our_metrics={"per_class": _per_class_ordered(ours)},
        output_path=OUTPUT_DIR / "per_class_f1.png",
    )
    save_agreement_bar_chart(
        agreement_count=metrics["comparison"]["agreement_count"],
        disagreement_count=metrics["comparison"]["disagreement_count"],
        output_path=OUTPUT_DIR / "agreement_bar.png",
    )
    _save_class_distribution_chart(
        labels=SIGNAL_LABELS,
        y_true_counts=[metrics["dataset"]["sample_class_counts"][label] for label in SIGNAL_LABELS],
        y_finbert_counts=[
            int(sum(finbert["confusion_matrix"][row][col] for row in range(3)))
            for col in range(3)
        ],
        y_ours_counts=[
            int(sum(ours["confusion_matrix"][row][col] for row in range(3)))
            for col in range(3)
        ],
        output_path=OUTPUT_DIR / "class_distribution.png",
    )

    print(f"Saved diagrams to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
