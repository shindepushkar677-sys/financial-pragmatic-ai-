"""Financial pragmatic transformer v2: FinBERT + speaker + pragmatic layers."""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from financial_pragmatic_ai.models.speaker_embedding import get_speaker_embedding
from financial_pragmatic_ai.models.pragmatic_input_layer import PragmaticInputLayer
from financial_pragmatic_ai.models.pragmatic_attention import PragmaticAttention


MODEL_NAME = "ProsusAI/finbert"


class FinancialPragmaticTransformer(nn.Module):
    """Integrates FinBERT with pragmatic speaker-aware modeling."""

    def __init__(self, num_intents: int = 4) -> None:
        super().__init__()
        self.intent_labels = [
            "EXPANSION",
            "COST_PRESSURE",
            "STRATEGIC_PROBING",
            "GENERAL_UPDATE",
        ]
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.finbert = AutoModel.from_pretrained(MODEL_NAME)
        self.pragmatic_input = PragmaticInputLayer()
        self.attention = PragmaticAttention()
        self.classifier = nn.Linear(512, num_intents)

    def forward(
        self, text_tokens: dict[str, torch.Tensor], speaker_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            text_tokens: Tokenized text dict for FinBERT.
            speaker_embedding: Tensor shaped (batch, 32) or (batch, seq_len, 32).

        Returns:
            Logits tensor shaped (batch, num_intents).
        """
        finbert_output = self.finbert(**text_tokens)
        token_embeddings = finbert_output.last_hidden_state  # (batch, seq_len, 768)

        if speaker_embedding.dim() == 2:
            speaker_embedding = speaker_embedding.unsqueeze(1).expand(
                -1, token_embeddings.size(1), -1
            )
        elif speaker_embedding.dim() == 3 and speaker_embedding.size(1) != token_embeddings.size(1):
            raise ValueError("speaker_embedding seq_len must match token_embeddings seq_len")

        pragmatic_tokens = self.pragmatic_input(token_embeddings, speaker_embedding)
        attention_output, _ = self.attention(pragmatic_tokens)
        logits = self.classifier(attention_output)
        return logits

    def predict(
        self,
        text: str,
        speaker: str,
        target_device: torch.device | None = None,
    ) -> str:
        if target_device is None:
            target_device = next(self.parameters()).device

        if speaker.upper() == "UNKNOWN":
            speaker = "EXECUTIVE"

        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(target_device)
        attention_mask = encoded["attention_mask"].to(target_device)
        speaker_embedding = get_speaker_embedding(speaker).to(target_device)

        with torch.no_grad():
            logits = self(
                {"input_ids": input_ids, "attention_mask": attention_mask},
                speaker_embedding,
            )

        text_lower = text.lower()
        cost_keywords = [
            "cost",
            "costs",
            "expense",
            "expenses",
            "margin",
            "pressure",
            "decline",
            "increase in costs",
            "compression",
        ]

        if speaker.upper() == "CFO" and any(k in text_lower for k in cost_keywords):
            return "COST_PRESSURE"

        if speaker.upper() == "ANALYST" and any(
            k in text_lower for k in ["margin", "risk", "decline", "impact"]
        ):
            return "STRATEGIC_PROBING"

        pred_idx = int(torch.argmax(logits, dim=-1).item())
        confidence = float(torch.softmax(logits, dim=-1)[0][pred_idx].item())
        intent = self.intent_labels[pred_idx]

        if confidence < 0.6:
            intent = "GENERAL_UPDATE"

        if intent == "GENERAL_UPDATE":
            if "growth" in text_lower:
                return "EXPANSION"
            if "cost" in text_lower:
                return "COST_PRESSURE"

        return intent


if __name__ == "__main__":
    model = FinancialPragmaticTransformer(num_intents=4)

    sample_text = "We expect margin compression next quarter but demand remains strong."
    predicted_intent = model.predict(sample_text, speaker="CEO")
    print(f"Predicted intent: {predicted_intent}")
