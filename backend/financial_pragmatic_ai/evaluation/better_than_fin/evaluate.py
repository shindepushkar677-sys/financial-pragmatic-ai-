"""Run full evaluation: FinBERT baseline vs custom financial NLP system."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from sklearn.metrics import classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from financial_pragmatic_ai.analysis.financial_signal_engine import (
    compute_confidence,
    compute_intent_distribution,
    compute_risk_score,
    detect_volatility,
    derive_signal,
)
from financial_pragmatic_ai.analysis.insight_engine import extract_key_drivers
from financial_pragmatic_ai.analysis.market_predictor import predict_market_outlook
from financial_pragmatic_ai.analysis.transcript_analyzer import TranscriptAnalyzer
from financial_pragmatic_ai.evaluation.better_than_fin.metrics import (
    compute_metrics,
    delta_metrics,
    to_numpy_confusion,
)
from financial_pragmatic_ai.evaluation.better_than_fin.utils import (
    SIGNAL_LABELS,
    agreement_rate,
    average_confidence_per_class,
    baseline_sentiment_to_signal,
    build_balanced_signal_sample,
    build_ground_truth_signals,
    clear_results_dir,
    ensure_results_dir,
    explain_our_decision,
    load_evaluation_dataset,
    normalize_confidence_to_percent,
    snippet,
)
from financial_pragmatic_ai.evaluation.better_than_fin.visualize import (
    save_agreement_bar_chart,
    save_class_distribution_chart,
    save_model_comparison_chart,
    save_normalized_confusion_matrix,
    save_per_class_f1_chart,
)


def _normalize_finbert_label(raw_label: str, pred_idx: int) -> str:
    label = str(raw_label).lower()
    if "positive" in label:
        return "positive"
    if "negative" in label:
        return "negative"
    if "neutral" in label:
        return "neutral"
    fallback = {0: "positive", 1: "negative", 2: "neutral"}
    return fallback.get(pred_idx, "neutral")


def run_finbert_baseline(
    texts: List[str],
    model_name: str = "ProsusAI/finbert",
    batch_size: int = 32,
    max_length: int = 128,
) -> Dict[str, List]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    sentiments: List[str] = []
    signals: List[str] = []
    confidences: List[float] = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)
            pred_ids = torch.argmax(probs, dim=-1)
            id2label = getattr(model.config, "id2label", {}) or {}

            for row, pred_id_tensor in enumerate(pred_ids):
                pred_id = int(pred_id_tensor.item())
                raw_label = id2label.get(pred_id, str(pred_id))
                sentiment = _normalize_finbert_label(raw_label, pred_id)
                signal = baseline_sentiment_to_signal(sentiment)
                confidence = normalize_confidence_to_percent(float(probs[row, pred_id].item()))

                sentiments.append(sentiment)
                signals.append(signal)
                confidences.append(confidence)

            step = start // batch_size + 1
            if step % 20 == 0:
                print(
                    f"[FinBERT baseline] processed "
                    f"{min(start + len(batch_texts), len(texts))}/{len(texts)}"
                )

    return {"sentiments": sentiments, "signals": signals, "confidences": confidences}


def _safe_analyze(transcript_analyzer: TranscriptAnalyzer, text: str) -> List[dict]:
    segments = transcript_analyzer.analyze(text)
    return segments


def run_custom_system(texts: List[str]) -> List[Dict]:
    analyzer = TranscriptAnalyzer()
    outputs: List[Dict] = []

    for index, text in enumerate(texts, start=1):
        segments = _safe_analyze(analyzer, text)
        if not segments:
            segments = [{"speaker": "EXECUTIVE", "text": text, "intent": "GENERAL_UPDATE"}]

        score = compute_risk_score(segments)
        signal = derive_signal(score)
        confidence = float(compute_confidence(segments))
        volatility = detect_volatility(segments)
        distribution = compute_intent_distribution(segments)
        prediction = predict_market_outlook(
            signal=signal,
            risk_score=score,
            volatility=volatility,
            intent_distribution=distribution,
        )
        drivers = extract_key_drivers(segments)

        outputs.append(
            {
                "segments": segments,
                "signal": signal,
                "prediction": prediction["prediction"],
                "prediction_explanation": prediction["explanation"],
                "confidence": normalize_confidence_to_percent(confidence),
                "volatility": volatility,
                "score": score,
                "drivers": drivers,
            }
        )

        if index % 20 == 0:
            print(f"[Our system] processed {index}/{len(texts)}")

    return outputs


def _build_case_rows(
    sampled_df: pd.DataFrame,
    y_true: List[str],
    baseline: Dict[str, List],
    ours: List[Dict],
) -> pd.DataFrame:
    rows = []

    for idx in range(len(sampled_df)):
        true_signal = y_true[idx]
        finbert_signal = baseline["signals"][idx]
        our_signal = ours[idx]["signal"]

        case_type = "agreement"
        if finbert_signal != our_signal:
            if our_signal == true_signal and finbert_signal != true_signal:
                case_type = "ours_correct_finbert_wrong"
            elif finbert_signal == true_signal and our_signal != true_signal:
                case_type = "finbert_correct_ours_wrong"
            else:
                case_type = "disagreement_other"

        text = str(sampled_df.iloc[idx]["text"])
        rows.append(
            {
                "index": idx,
                "text": text,
                "text_snippet": snippet(text),
                "ground_truth_signal": true_signal,
                "finbert_signal": finbert_signal,
                "our_signal": our_signal,
                "finbert_sentiment": baseline["sentiments"][idx],
                "finbert_confidence": baseline["confidences"][idx],
                "our_confidence": ours[idx]["confidence"],
                "our_prediction": ours[idx]["prediction"],
                "our_volatility": ours[idx]["volatility"],
                "our_score": ours[idx]["score"],
                "our_explanation": explain_our_decision(ours[idx]["segments"], our_signal),
                "case_type": case_type,
            }
        )

    return pd.DataFrame(rows)


def run_evaluation(
    dataset_path: str | Path | None = None,
    per_class_target: int = 80,
    results_dir: str | Path | None = None,
    batch_size: int = 32,
) -> Dict:
    results_path = ensure_results_dir(results_dir)
    clear_results_dir(results_path)

    full_df = load_evaluation_dataset(dataset_path=dataset_path)
    full_df["ground_truth_signal"] = build_ground_truth_signals(full_df["intent"].tolist())
    full_counts = {
        label: int((full_df["ground_truth_signal"] == label).sum()) for label in SIGNAL_LABELS
    }

    sampled_df = build_balanced_signal_sample(
        full_df,
        per_class_target=per_class_target,
        random_state=42,
    )
    if sampled_df.empty:
        raise ValueError("Balanced sampling produced an empty dataset.")

    sampled_counts = {
        label: int((sampled_df["ground_truth_signal"] == label).sum()) for label in SIGNAL_LABELS
    }

    texts = sampled_df["text"].tolist()
    y_true = sampled_df["ground_truth_signal"].tolist()

    print(f"Full dataset size: {len(full_df)}")
    print(f"Full per-class counts: {full_counts}")
    print(f"Balanced sample size: {len(sampled_df)}")
    print(f"Balanced per-class counts: {sampled_counts}")

    print("Running FinBERT baseline...")
    baseline = run_finbert_baseline(texts=texts, batch_size=batch_size)

    print("Running custom system...")
    ours = run_custom_system(texts=texts)

    y_pred_finbert = baseline["signals"]
    y_pred_ours = [item["signal"] for item in ours]
    confidence_ours = [item["confidence"] for item in ours]

    finbert_distribution = Counter(y_pred_finbert)
    our_distribution = Counter(y_pred_ours)
    print(f"FinBERT prediction distribution: {dict(finbert_distribution)}")
    print(f"Our system prediction distribution: {dict(our_distribution)}")

    if y_pred_ours:
        dominant_share = max(our_distribution.values()) / len(y_pred_ours)
        if dominant_share > 0.80:
            print("[ERROR] Model collapse detected")

    finbert_metrics = compute_metrics(y_true=y_true, y_pred=y_pred_finbert, labels=SIGNAL_LABELS)
    our_metrics = compute_metrics(y_true=y_true, y_pred=y_pred_ours, labels=SIGNAL_LABELS)
    deltas = delta_metrics(finbert_metrics, our_metrics)

    agree_rate = agreement_rate(y_pred_finbert, y_pred_ours)
    agreement_count = int(round(agree_rate * len(y_true)))
    disagreement_count = int(len(y_true) - agreement_count)

    confidence_comparison = {
        "finbert": average_confidence_per_class(
            y_pred_finbert, baseline["confidences"], labels=SIGNAL_LABELS
        ),
        "our_system": average_confidence_per_class(
            y_pred_ours, confidence_ours, labels=SIGNAL_LABELS
        ),
    }

    cases_df = _build_case_rows(
        sampled_df=sampled_df,
        y_true=y_true,
        baseline=baseline,
        ours=ours,
    )
    disagreement_df = cases_df[cases_df["finbert_signal"] != cases_df["our_signal"]].copy()
    ours_correct_df = cases_df[
        (cases_df["our_signal"] == cases_df["ground_truth_signal"])
        & (cases_df["finbert_signal"] != cases_df["ground_truth_signal"])
    ].copy()
    finbert_correct_df = cases_df[
        (cases_df["finbert_signal"] == cases_df["ground_truth_signal"])
        & (cases_df["our_signal"] != cases_df["ground_truth_signal"])
    ].copy()

    summary = {
        "dataset": {
            "full_size": int(len(full_df)),
            "sample_size": int(len(sampled_df)),
            "full_class_counts": full_counts,
            "sample_class_counts": sampled_counts,
            "per_class_target": int(per_class_target),
            "sampling_random_state": 42,
        },
        "finbert": finbert_metrics,
        "our_system": our_metrics,
        "improvement": deltas,
        "comparison": {
            "agreement_rate": round(float(agree_rate), 6),
            "agreement_count": agreement_count,
            "disagreement_count": int(len(disagreement_df)),
            "ours_correct_finbert_wrong_count": int(len(ours_correct_df)),
            "finbert_correct_ours_wrong_count": int(len(finbert_correct_df)),
        },
        "confidence_comparison": confidence_comparison,
        "results_dir": str(results_path.resolve()),
    }

    with open(results_path / "metrics_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    report_lines = [
        "=== DATASET ===",
        f"Full dataset size: {len(full_df)}",
        f"Balanced sample size: {len(sampled_df)}",
        f"Full per-class counts: {full_counts}",
        f"Balanced per-class counts: {sampled_counts}",
        "",
        "=== FINBERT CLASSIFICATION REPORT ===",
        classification_report(
            y_true,
            y_pred_finbert,
            labels=SIGNAL_LABELS,
            target_names=SIGNAL_LABELS,
            digits=4,
            zero_division=0,
        ),
        "",
        "=== OUR SYSTEM CLASSIFICATION REPORT ===",
        classification_report(
            y_true,
            y_pred_ours,
            labels=SIGNAL_LABELS,
            target_names=SIGNAL_LABELS,
            digits=4,
            zero_division=0,
        ),
        "",
        "=== AGREEMENT ANALYSIS ===",
        f"Agreement rate: {agree_rate:.4f}",
        f"Disagreement count: {len(disagreement_df)}",
        f"Ours correct, FinBERT wrong: {len(ours_correct_df)}",
        f"FinBERT correct, Ours wrong: {len(finbert_correct_df)}",
    ]
    report_text = "\n".join(report_lines)
    with open(results_path / "classification_report.txt", "w", encoding="utf-8") as file:
        file.write(report_text)

    disagreement_df.to_csv(results_path / "disagreement_cases.csv", index=False)
    ours_correct_df.to_csv(results_path / "ours_correct_finbert_wrong.csv", index=False)

    save_normalized_confusion_matrix(
        labels=SIGNAL_LABELS,
        confusion_matrix=to_numpy_confusion(our_metrics),
        title="Our System - Normalized Confusion Matrix",
        output_path=results_path / "confusion_matrix_ours_normalized.png",
    )
    save_normalized_confusion_matrix(
        labels=SIGNAL_LABELS,
        confusion_matrix=to_numpy_confusion(finbert_metrics),
        title="FinBERT - Normalized Confusion Matrix",
        output_path=results_path / "confusion_matrix_finbert_normalized.png",
    )
    save_model_comparison_chart(
        finbert_metrics=finbert_metrics,
        our_metrics=our_metrics,
        output_path=results_path / "model_comparison.png",
    )
    save_per_class_f1_chart(
        labels=SIGNAL_LABELS,
        finbert_metrics=finbert_metrics,
        our_metrics=our_metrics,
        output_path=results_path / "per_class_f1.png",
    )
    save_agreement_bar_chart(
        agreement_count=agreement_count,
        disagreement_count=disagreement_count,
        output_path=results_path / "agreement_bar.png",
    )
    save_class_distribution_chart(
        labels=SIGNAL_LABELS,
        y_true=y_true,
        y_finbert=y_pred_finbert,
        y_ours=y_pred_ours,
        output_path=results_path / "class_distribution.png",
    )

    print("\n=== MODEL COMPARISON ===\n")
    print(f"Sample size: {len(sampled_df)}")
    print(f"Per-class counts: {sampled_counts}")
    print("")
    print("FinBERT:")
    print(f"- Accuracy: {finbert_metrics['accuracy']:.4f}")
    print(f"- F1: {finbert_metrics['macro_f1']:.4f}")
    print("")
    print("Our System:")
    print(f"- Accuracy: {our_metrics['accuracy']:.4f}")
    print(f"- F1: {our_metrics['macro_f1']:.4f}")
    print("")
    print("Improvement:")
    print(f"- Δ Accuracy: {deltas['accuracy_delta']:.4f}")
    print(f"- Δ F1: {deltas['macro_f1_delta']:.4f}")

    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate custom pipeline vs FinBERT baseline.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Optional path to dataset CSV.",
    )
    parser.add_argument(
        "--per-class-target",
        type=int,
        default=80,
        help="Target samples per signal class for balanced evaluation.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Optional output directory for evaluation artifacts.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="FinBERT baseline inference batch size.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    run_evaluation(
        dataset_path=args.dataset_path,
        per_class_target=args.per_class_target,
        results_dir=args.results_dir,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
