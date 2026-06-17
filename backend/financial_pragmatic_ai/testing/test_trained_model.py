import torch
from transformers import AutoTokenizer

from financial_pragmatic_ai.models.financial_pragmatic_transformer_v2 import (
    FinancialPragmaticTransformer,
)
from financial_pragmatic_ai.models.speaker_embedding import get_speaker_embedding

MODEL_PATH = "financial_pragmatic_ai/models/pragmatic_transformer_trained.pt"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

INTENT_LABELS = [
    "EXPANSION",
    "COST_PRESSURE",
    "STRATEGIC_PROBING",
    "GENERAL_UPDATE",
]

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")


def load_model():
    model = FinancialPragmaticTransformer()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict(model, text, speaker="EXECUTIVE"):
    encoded = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    speaker_embedding = get_speaker_embedding(speaker).to(device)

    with torch.no_grad():
        logits = model(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            speaker_embedding,
        )

    pred_idx = torch.argmax(logits, dim=-1).item()
    return INTENT_LABELS[pred_idx]


if __name__ == "__main__":
    model = load_model()

    samples = [
        ("We expect margin compression next quarter.", "CFO"),
        ("We are expanding our operations in Asia.", "CEO"),
        ("Could you elaborate on restructuring costs?", "ANALYST"),
        ("Operating expenses increased due to supply chain issues.", "CFO"),
    ]

    for text, speaker in samples:
        intent = predict(model, text, speaker)
        print(f"{speaker}: {text}")
        print("Predicted intent:", intent)
        print()