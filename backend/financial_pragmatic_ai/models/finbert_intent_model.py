from __future__ import annotations

import hashlib
import re
import string
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset as HFDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from financial_pragmatic_ai.evaluation.better_than_fin.utils import build_ground_truth_signals


MODEL_NAME = "yiyanghkust/finbert-tone"
HF_INTENT_REPO = "SarcoNarco/finbert_intent_v3"
INTENT_LABELS = [
    "EXPANSION",
    "COST_PRESSURE",
    "STRATEGIC_PROBING",
    "GENERAL_UPDATE",
]
INTENT_TO_INDEX = {label: idx for idx, label in enumerate(INTENT_LABELS)}
INDEX_TO_INTENT = {idx: label for label, idx in INTENT_TO_INDEX.items()}
LABEL2ID = {
    "EXPANSION": 0,
    "COST_PRESSURE": 1,
    "STRATEGIC_PROBING": 2,
    "GENERAL_UPDATE": 3,
}
ID2LABEL = {
    0: "EXPANSION",
    1: "COST_PRESSURE",
    2: "STRATEGIC_PROBING",
    3: "GENERAL_UPDATE",
}

DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "pragmatic_intent_dataset_clean.csv"
)
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[0] / "finbert_intent_v3"

_SIGNAL_TO_INTENT = {
    "GROWTH": "EXPANSION",
    "RISK": "COST_PRESSURE",
    "NEUTRAL": "GENERAL_UPDATE",
}

_ANALYST_QUESTION_PATTERN = re.compile(
    r"\?|\bcould you\b|\bcan you\b|\bhow\b|\bwhat\b|\bwhy\b|\bguidance\b",
    re.IGNORECASE,
)
_PROBING_START_PATTERN = re.compile(
    r"^\s*(?:can you|what is|how|could you)\b",
    re.IGNORECASE,
)
RISK_KEYWORDS = [
    "decline",
    "decrease",
    "drop",
    "fall",
    "weakness",
    "pressure",
    "margin pressure",
    "cost increase",
    "inflation",
    "headwind",
    "uncertainty",
    "slowdown",
    "challenging",
    "soft demand",
    "lower revenue",
    "compression",
    "volatility",
    "risk",
    "exposure",
]
_TRANSCRIPT_ID_COLUMNS = (
    "transcript_id",
    "call_id",
    "conversation_id",
    "document_id",
    "doc_id",
    "transcript",
    "transcript_text",
)


def _resolve_output_dir(output_path: Path | str | None) -> Path:
    if output_path is None:
        return DEFAULT_MODEL_DIR

    output = Path(output_path)
    if output.suffix:
        return output.with_suffix("")
    return output


def _resolve_dataset_path(dataset_path: Path | str | None) -> Path:
    if dataset_path is None:
        return DEFAULT_DATASET_PATH
    return Path(dataset_path)


def _normalize_for_hash(text: str) -> str:
    normalized = str(text or "").lower()
    normalized = normalized.translate(str.maketrans({char: " " for char in string.punctuation}))
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _text_hash(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def load_raw_dataset(dataset_path: Path | str | None = None) -> pd.DataFrame:
    resolved_dataset_path = _resolve_dataset_path(dataset_path)
    if not resolved_dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {resolved_dataset_path}")

    raw = pd.read_csv(resolved_dataset_path)
    rename_map = {column: column.strip().lower() for column in raw.columns}
    raw = raw.rename(columns=rename_map)

    if "text" not in raw.columns:
        raise ValueError("Dataset must include 'text' column")
    if "intent" not in raw.columns and "signal" not in raw.columns:
        raise ValueError("Dataset must include either 'intent' or 'signal' column")

    label_source_col = "intent" if "intent" in raw.columns else "signal"

    transcript_id_col = None
    for candidate in _TRANSCRIPT_ID_COLUMNS:
        if candidate in raw.columns:
            transcript_id_col = candidate
            break

    dataset = pd.DataFrame(
        {
            "text": raw["text"].fillna("").astype(str),
            "speaker": raw["speaker"].fillna("EXECUTIVE").astype(str).str.upper()
            if "speaker" in raw.columns
            else "EXECUTIVE",
            "source_label": raw[label_source_col].fillna("GENERAL_UPDATE").astype(str).str.upper(),
        }
    )

    if transcript_id_col is not None:
        dataset["transcript_id"] = raw[transcript_id_col].fillna("").astype(str)
    else:
        # Assumption fallback when transcript IDs are unavailable in the source dataset.
        dataset["transcript_id"] = dataset["text"].map(_normalize_for_hash)

    dataset = dataset[dataset["text"].str.strip().str.len() > 0].reset_index(drop=True)
    if dataset.empty:
        raise ValueError("Raw dataset is empty after cleaning text.")
    return dataset


def split_dataset_transcript_level(dataset: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if dataset.empty:
        raise ValueError("Cannot split empty dataset.")

    transcript_table = dataset[["transcript_id"]].drop_duplicates().copy()
    transcript_table["split"] = transcript_table["transcript_id"].map(
        lambda value: "train" if int(_text_hash(value), 16) % 100 < 80 else "eval"
    )

    merged = dataset.merge(transcript_table, on="transcript_id", how="left")
    train_raw = merged[merged["split"] == "train"].copy().reset_index(drop=True)
    eval_raw = merged[merged["split"] == "eval"].copy().reset_index(drop=True)

    if train_raw.empty or eval_raw.empty:
        raise ValueError(
            "Transcript-level split produced an empty split. "
            f"train_rows={len(train_raw)} eval_rows={len(eval_raw)}"
        )

    train_hashes = set(train_raw["text"].map(_normalize_for_hash).map(_text_hash))
    eval_hashes = set(eval_raw["text"].map(_normalize_for_hash).map(_text_hash))
    duplicate_hashes = train_hashes & eval_hashes
    if duplicate_hashes:
        print(
            "[WARNING] Duplicate normalized segment texts detected across train/eval splits: "
            f"{len(duplicate_hashes)}"
        )

    return {
        "train_raw": train_raw,
        "eval_raw": eval_raw,
        "duplicate_hashes": pd.DataFrame({"hash": list(duplicate_hashes)}),
    }


def _apply_intent_mapping(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    signals = build_ground_truth_signals(prepared["source_label"].tolist())
    prepared["intent"] = [_SIGNAL_TO_INTENT[signal.upper()] for signal in signals]

    text_lower = prepared["text"].astype(str).str.lower()
    risk_mask = text_lower.apply(
        lambda text: any(keyword in text for keyword in RISK_KEYWORDS)
    )
    risk_override_count = int(risk_mask.sum())
    prepared.loc[risk_mask, "intent"] = "COST_PRESSURE"

    question_mask = prepared["text"].str.contains(_ANALYST_QUESTION_PATTERN, regex=True)
    starter_mask = prepared["text"].str.contains(_PROBING_START_PATTERN, regex=True)
    probing_mask = question_mask | starter_mask
    prepared.loc[probing_mask, "intent"] = "STRATEGIC_PROBING"

    final_distribution = prepared["intent"].value_counts().to_dict()
    print(f"Risk keyword override count: {risk_override_count}")
    print(f"Final class distribution after mapping: {final_distribution}")

    return prepared[["text", "speaker", "transcript_id", "intent"]]


def build_train_set(split_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    mapped_train = _apply_intent_mapping(split_data["train_raw"])
    return _balance_intent_frame(mapped_train, random_state=42)


def build_eval_set(split_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return _apply_intent_mapping(split_data["eval_raw"])


def _balance_intent_frame(frame: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    counts = frame["intent"].value_counts().to_dict()
    missing = [label for label in INTENT_LABELS if counts.get(label, 0) == 0]
    if missing:
        raise ValueError(
            f"Cannot train 4-class model; missing labels after mapping/heuristics: {missing}. "
            f"Counts: {counts}"
        )

    target_size = min(counts[label] for label in INTENT_LABELS)
    if target_size == 0:
        raise ValueError(f"Invalid balanced target size 0. Counts: {counts}")

    balanced_parts = []
    for label in INTENT_LABELS:
        subset = frame[frame["intent"] == label]
        balanced_parts.append(subset.sample(n=target_size, random_state=random_state))

    balanced = pd.concat(balanced_parts, ignore_index=True)
    balanced = balanced.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return balanced


def _build_hf_train_dataset(frame: pd.DataFrame, tokenizer, max_length: int = 128) -> HFDataset:
    # Required HF fields: text, label
    hf_source = pd.DataFrame(
        {
            "text": frame["text"].astype(str),
            "label": frame["intent"].map(INTENT_TO_INDEX).astype(int),
        }
    )

    dataset = HFDataset.from_pandas(hf_source, preserve_index=False)

    def _tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized = dataset.map(_tokenize_batch, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized = tokenized.remove_columns(["text"])
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized


# ---------------------------------------------------------------------------
# Models are loaded from HuggingFace Hub.
# No local model files required.
# Primary repo: SarcoNarco/finbert_intent_v3
# ---------------------------------------------------------------------------
class FinBERTIntentModel:

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        model_dir: Path | str | None = None,
        device: torch.device | None = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Loading FinBERT intent model from HuggingFace: {HF_INTENT_REPO}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(HF_INTENT_REPO)
            self.model = AutoModelForSequenceClassification.from_pretrained(HF_INTENT_REPO)
        except Exception as exc:
            print(f"[ERROR] Failed to load FinBERT intent model from HuggingFace: {exc}")
            raise

        self.model.to(self.device)
        self.model.eval()
        print(f"[INFO] Loaded FinBERT intent model num_labels={self.model.config.num_labels}")

    def save_pretrained(self, output_dir: Path | str = DEFAULT_MODEL_DIR):
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(target)
        self.tokenizer.save_pretrained(target)

    def predict(self, text: str, max_length: int = 128):
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.no_grad():
            outputs = self.model(
                **encoded,
                output_hidden_states=True,
                return_dict=True,
            )
            logits = outputs.logits.squeeze(0)
            cls_embedding = outputs.hidden_states[-1][:, 0, :].squeeze(0)

        logits_cpu = logits.detach().cpu()
        probs = torch.softmax(logits_cpu, dim=-1)
        pred_class = int(torch.argmax(probs).item())
        label_map = {
            0: "EXPANSION",
            1: "COST_PRESSURE",
            2: "STRATEGIC_PROBING",
            3: "GENERAL_UPDATE",
        }
        intent = label_map.get(pred_class, "GENERAL_UPDATE")
        print("LOGITS:", logits_cpu.tolist())
        print("PRED CLASS:", pred_class)
        print(f"[DEBUG] CLASS → INTENT: {pred_class} → {intent}")

        return {
            "intent": intent,
            "logits": logits_cpu,
            "embedding": cls_embedding.detach().cpu().float(),
            "confidence": float(probs[pred_class].item()),
        }


def train_finbert_intent_model(
    dataset_path: Path | str | None = None,
    output_path: Path | str | None = None,
    max_length: int = 128,
    batch_size: int = 16,
    epochs: int = 3,
    learning_rate: float = 2e-5,
):
    output_dir = _resolve_output_dir(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_dataset_path = _resolve_dataset_path(dataset_path)
    print(f"Loading evaluation dataset source: {resolved_dataset_path}")
    raw_dataset = load_raw_dataset(dataset_path=resolved_dataset_path)
    split_data = split_dataset_transcript_level(raw_dataset)
    train_frame = build_train_set(split_data)
    eval_frame = build_eval_set(split_data)

    if train_frame.empty:
        raise ValueError("Training dataset is empty after mapping and balancing.")
    if eval_frame.empty:
        raise ValueError("Evaluation dataset is empty after transcript-level split.")

    print(
        f"Train transcripts: {split_data['train_raw']['transcript_id'].nunique()} | "
        f"Eval transcripts: {split_data['eval_raw']['transcript_id'].nunique()}"
    )
    print(
        f"Train segments: {len(split_data['train_raw'])} | "
        f"Eval segments: {len(split_data['eval_raw'])}"
    )
    class_counts = train_frame["intent"].value_counts().reindex(INTENT_LABELS, fill_value=0).to_dict()
    eval_counts = eval_frame["intent"].value_counts().reindex(INTENT_LABELS, fill_value=0).to_dict()
    print(f"Prepared training rows (balanced): {len(train_frame)}")
    print(f"Training class distribution: {class_counts}")
    print(f"Evaluation class distribution: {eval_counts}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = _build_hf_train_dataset(train_frame, tokenizer=tokenizer, max_length=max_length)
    eval_dataset = _build_hf_train_dataset(eval_frame, tokenizer=tokenizer, max_length=max_length)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=4,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir / "trainer_artifacts"),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        logging_steps=100,
        save_strategy="no",
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved 4-class FinBERT intent model to: {output_dir}")

    wrapper_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return FinBERTIntentModel(model_dir=output_dir, device=wrapper_device)


if __name__ == "__main__":
    train_finbert_intent_model()
