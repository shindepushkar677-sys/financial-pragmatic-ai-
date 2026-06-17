"""Visualization helpers for better-than-FinBERT reporting."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _normalize_confusion_matrix(cm: np.ndarray) -> np.ndarray:
    cm = cm.astype(np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return cm / row_sums


def save_normalized_confusion_matrix(
    labels: List[str],
    confusion_matrix: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    normalized = _normalize_confusion_matrix(confusion_matrix)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for row in range(normalized.shape[0]):
        for col in range(normalized.shape[1]):
            pct = normalized[row, col] * 100.0
            ax.text(col, row, f"{pct:.1f}%", ha="center", va="center", fontsize=9)

    colorbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Percentage")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_model_comparison_chart(
    finbert_metrics: Dict,
    our_metrics: Dict,
    output_path: Path,
) -> None:
    metrics = ["Accuracy", "Macro F1"]
    fin_values = [finbert_metrics["accuracy"], finbert_metrics["macro_f1"]]
    our_values = [our_metrics["accuracy"], our_metrics["macro_f1"]]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.bar(x - width / 2, fin_values, width, label="FinBERT", color="#6b7280")
    ax.bar(x + width / 2, our_values, width, label="Our System", color="#2563eb")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_per_class_f1_chart(
    labels: List[str],
    finbert_metrics: Dict,
    our_metrics: Dict,
    output_path: Path,
) -> None:
    fin_values = [finbert_metrics["per_class"][label]["f1"] for label in labels]
    our_values = [our_metrics["per_class"][label]["f1"] for label in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, fin_values, width, label="FinBERT", color="#9ca3af")
    ax.bar(x + width / 2, our_values, width, label="Our System", color="#3b82f6")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("F1")
    ax.set_title("Per-Class F1 Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_agreement_bar_chart(
    agreement_count: int,
    disagreement_count: int,
    output_path: Path,
) -> None:
    labels = ["Agreement", "Disagreement"]
    values = [agreement_count, disagreement_count]
    colors = ["#16a34a", "#dc2626"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Count")
    ax.set_title("Agreement vs Disagreement")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for i, value in enumerate(values):
        ax.text(i, value + max(values) * 0.01 if max(values) > 0 else 0.01, str(value), ha="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_class_distribution_chart(
    labels: List[str],
    y_true: List[str],
    y_finbert: List[str],
    y_ours: List[str],
    output_path: Path,
) -> None:
    counts_true = [sum(1 for value in y_true if value == label) for label in labels]
    counts_finbert = [sum(1 for value in y_finbert if value == label) for label in labels]
    counts_ours = [sum(1 for value in y_ours if value == label) for label in labels]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, counts_true, width=width, label="Ground Truth", color="#6b7280")
    ax.bar(x, counts_finbert, width=width, label="FinBERT", color="#9ca3af")
    ax.bar(x + width, counts_ours, width=width, label="Our System", color="#2563eb")
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
