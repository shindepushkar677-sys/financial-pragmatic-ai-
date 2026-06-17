"""Train FinancialPragmaticTransformer on pragmatic intent data."""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from financial_pragmatic_ai.models.financial_pragmatic_transformer_v2 import (
    FinancialPragmaticTransformer,
)
from financial_pragmatic_ai.models.speaker_embedding import get_speaker_embedding


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training device:", device)


INTENT_TO_INDEX = {
    "EXPANSION": 0,
    "COST_PRESSURE": 1,
    "STRATEGIC_PROBING": 2,
    "GENERAL_UPDATE": 3,
}

DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "pragmatic_intent_dataset_clean.csv"
)
MODEL_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "pragmatic_transformer_trained.pt"
)


class PragmaticIntentDataset(Dataset):
    """Lazy-tokenization dataset using raw text and labels."""

    def __init__(self, frame: pd.DataFrame, tokenizer, max_length: int = 256) -> None:
        frame = frame.reset_index(drop=True)
        self.texts = frame["text"].tolist()
        self.speakers = frame["speaker"].tolist()
        self.label_ids = frame["intent"].map(INTENT_TO_INDEX).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict:
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "speaker": self.speakers[idx],
            "label": torch.tensor(self.label_ids[idx], dtype=torch.long),
        }


def collate_batch(batch: List[Dict]) -> Dict:
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "speakers": [item["speaker"] for item in batch],
        "labels": torch.stack([item["label"] for item in batch]),
    }


def build_speaker_batch(speakers: List[str], target_device: torch.device) -> torch.Tensor:
    embeddings = [get_speaker_embedding(speaker).squeeze(0) for speaker in speakers]
    return torch.stack(embeddings).to(target_device)


def compute_class_weights(train_df: pd.DataFrame) -> torch.Tensor:
    label_ids = train_df["intent"].map(INTENT_TO_INDEX).to_numpy()
    counts = np.bincount(label_ids, minlength=len(INTENT_TO_INDEX)).astype(np.float32)
    counts = np.clip(counts, a_min=1.0, a_max=None)
    weights = counts.sum() / (len(INTENT_TO_INDEX) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def evaluate(
    model: FinancialPragmaticTransformer,
    dataloader: DataLoader,
    criterion: nn.Module,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            speaker_embeddings = build_speaker_batch(batch["speakers"], device)

            logits = model(
                {"input_ids": input_ids, "attention_mask": attention_mask},
                speaker_embeddings,
            )
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    accuracy = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    macro_f1 = (
        f1_score(all_labels, all_preds, average="macro", zero_division=0)
        if all_labels
        else 0.0
    )
    return avg_loss, accuracy, macro_f1


def main() -> None:
    df = pd.read_csv(DATASET_PATH)
    df["text"] = df["text"].fillna("").astype(str)
    df["speaker"] = df["speaker"].fillna("EXECUTIVE").astype(str)
    df["intent"] = df["intent"].fillna("GENERAL_UPDATE").astype(str)
    df = df[df["intent"].isin(INTENT_TO_INDEX)].reset_index(drop=True)

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["intent"],
    )

    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    train_dataset = PragmaticIntentDataset(train_df, tokenizer=tokenizer)
    val_dataset = PragmaticIntentDataset(val_df, tokenizer=tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
    )

    model = FinancialPragmaticTransformer()
    model.to(device)

    class_weights = compute_class_weights(train_df)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    epochs = 3
    for epoch in range(epochs):
        model.train()
        train_loss_total = 0.0

        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            speaker_embeddings = build_speaker_batch(batch["speakers"], device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(
                    {"input_ids": input_ids, "attention_mask": attention_mask},
                    speaker_embeddings,
                )
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if step % 200 == 0:
                print(
                    f"Epoch {epoch+1} Step {step}/{len(train_loader)} Loss {loss.item():.4f}"
                )

            train_loss_total += loss.item()

        train_loss = train_loss_total / len(train_loader) if len(train_loader) > 0 else 0.0
        val_loss, val_accuracy, val_f1 = evaluate(model, val_loader, criterion)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_accuracy:.4f}")
        print(f"Val F1: {val_f1:.4f}")

    torch.save(model.state_dict(), MODEL_OUTPUT_PATH)
    print(f"Saved trained model to: {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
