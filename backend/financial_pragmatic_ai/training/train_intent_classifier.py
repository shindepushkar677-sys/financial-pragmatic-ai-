"""Train ExecutiveIntentClassifier on FinBERT CLS embeddings."""

import csv
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from financial_pragmatic_ai.models.finbert_base import load_finbert
from financial_pragmatic_ai.models.intent_classifier import (
    INTENT_LABELS,
    ExecutiveIntentClassifier,
)
from financial_pragmatic_ai.utils.device import get_torch_device


def load_dataset(csv_path: Path) -> Tuple[List[str], List[int]]:
    """Load text samples and mapped intent indices from CSV."""
    label_to_idx = {label: idx for idx, label in enumerate(INTENT_LABELS)}
    texts: List[str] = []
    labels: List[int] = []

    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            text = (row.get("text") or "").strip()
            intent = (row.get("intent") or "").strip()
            if not text or not intent:
                continue
            if intent not in label_to_idx:
                raise ValueError(f"Unknown intent label in dataset: {intent}")
            texts.append(text)
            labels.append(label_to_idx[intent])

    if not texts:
        raise ValueError("Dataset is empty or invalid.")

    return texts, labels


def build_embeddings(
    texts: List[str], tokenizer, finbert_model, device: torch.device
) -> torch.Tensor:
    """Generate CLS embeddings for each text sample using FinBERT."""
    embeddings: List[torch.Tensor] = []

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = finbert_model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()
            embeddings.append(cls_embedding)

    return torch.stack(embeddings)


def main() -> None:
    device = get_torch_device()
    tokenizer, finbert_model = load_finbert(device=device.type)

    project_root = Path(__file__).resolve().parents[2]
    dataset_path = project_root / "financial_pragmatic_ai" / "data" / "intent_dataset.csv"
    model_out_path = (
        project_root
        / "financial_pragmatic_ai"
        / "models"
        / "intent_classifier.pt"
    )

    texts, labels = load_dataset(dataset_path)
    x_embeddings = build_embeddings(texts, tokenizer, finbert_model, device)
    y_labels = torch.tensor(labels, dtype=torch.long)

    train_dataset = TensorDataset(x_embeddings, y_labels)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    classifier = ExecutiveIntentClassifier(num_intent_classes=len(INTENT_LABELS)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(classifier.parameters(), lr=1e-3)

    epochs = 10
    for epoch in range(1, epochs + 1):
        classifier.train()
        total_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = classifier(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")

    torch.save(classifier.state_dict(), model_out_path)
    print(f"Saved model to: {model_out_path}")


if __name__ == "__main__":
    main()
