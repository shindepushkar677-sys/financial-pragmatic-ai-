"""Integrated model: FinBERT encoder + pragmatic attention + intent classifier."""

import torch
from torch import nn

from financial_pragmatic_ai.models.finbert_base import load_finbert
from financial_pragmatic_ai.models.intent_classifier import (
    INTENT_LABELS,
    ExecutiveIntentClassifier,
)
from financial_pragmatic_ai.models.pragmatic_attention import PragmaticAttention
from financial_pragmatic_ai.utils.device import get_torch_device


class FinancialPragmaticTransformer(nn.Module):
    """End-to-end model for intent prediction from financial text."""

    def __init__(self) -> None:
        super().__init__()
        self.device = get_torch_device()
        self.tokenizer, self.finbert = load_finbert(device=self.device.type)
        self.finbert.gradient_checkpointing_enable()
        self.attention_input_projection = nn.Linear(768, 512).to(self.device)
        self.pragmatic_attention = PragmaticAttention().to(self.device)
        self.attention_output_projection = nn.Linear(512, 768).to(self.device)
        self.intent_classifier = ExecutiveIntentClassifier(
            num_intent_classes=len(INTENT_LABELS)
        ).to(self.device)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Tensor of token ids.
            attention_mask: Tensor mask for valid tokens.

        Returns:
            Intent logits of shape (batch_size, num_intent_classes).
        """
        outputs = self.finbert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_embeddings = outputs.last_hidden_state
        attention_inputs = self.attention_input_projection(sequence_embeddings)
        attended_embedding, _ = self.pragmatic_attention(attention_inputs)
        aggregated_embedding = self.attention_output_projection(attended_embedding)
        logits = self.intent_classifier(aggregated_embedding)
        return logits

    def predict(self, text: str) -> str:
        """Tokenize text, run inference, and return predicted intent label."""
        self.eval()
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.forward(input_ids=input_ids, attention_mask=attention_mask)
            pred_idx = int(torch.argmax(logits, dim=-1).item())
        return INTENT_LABELS[pred_idx]


if __name__ == "__main__":
    model = FinancialPragmaticTransformer()

    samples = [
        "We expect margin compression next quarter.",
        "We are expanding our operations in Asia.",
    ]

    for sentence in samples:
        intent = model.predict(sentence)
        print(f"Text: {sentence}")
        print(f"Predicted intent: {intent}")
