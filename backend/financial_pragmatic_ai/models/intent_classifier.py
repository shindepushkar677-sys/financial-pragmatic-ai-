"""Intent classifier for executive statements based on FinBERT embeddings."""

from typing import List

import torch
from torch import nn

INTENT_LABELS = [
    "EXPANSION",
    "COST_PRESSURE",
    "STRATEGIC_PROBING",
    "GENERAL_UPDATE",
]


class ExecutiveIntentClassifier(nn.Module):
    """Small feed-forward network for intent classification."""

    def __init__(self, num_intent_classes: int = len(INTENT_LABELS)) -> None:
        super().__init__()
        self.num_intent_classes = num_intent_classes
        self.intent_labels: List[str] = INTENT_LABELS[:num_intent_classes]

        self.network = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_intent_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits for each intent class."""
        return self.network(x)

    def predict(self, x: torch.Tensor) -> str:
        """Return the label with the highest predicted probability."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
            pred_idx = int(torch.argmax(probs, dim=-1).item())
        return self.intent_labels[pred_idx]


if __name__ == "__main__":
    model = ExecutiveIntentClassifier()
    sample_embedding = torch.randn(1, 768)
    predicted_intent = model.predict(sample_embedding)
    print(f"Predicted intent: {predicted_intent}")
