import torch
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from transformers import AutoTokenizer

from financial_pragmatic_ai.models.financial_pragmatic_transformer_v2 import (
    FinancialPragmaticTransformer,
)
from financial_pragmatic_ai.models.speaker_embedding import get_speaker_embedding

MODEL_PATH = "financial_pragmatic_ai/models/pragmatic_transformer_trained.pt"
DATA_PATH = "financial_pragmatic_ai/data/pragmatic_intent_dataset_clean.csv"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

INTENT_LABELS = [
    "EXPANSION",
    "COST_PRESSURE",
    "STRATEGIC_PROBING",
    "GENERAL_UPDATE",
]

INTENT_TO_INDEX = {k: i for i, k in enumerate(INTENT_LABELS)}

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")


def load_model():
    model = FinancialPragmaticTransformer()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict(model, text, speaker):
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
    return pred_idx


def main():

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    df = df[df["intent"].isin(INTENT_LABELS)]

    model = load_model()

    y_true = []
    y_pred = []

    print("Running evaluation...")

    for _, row in df.sample(2000).iterrows():

        text = row["text"]
        speaker = row["speaker"]
        true_label = INTENT_TO_INDEX[row["intent"]]

        pred_label = predict(model, text, speaker)

        y_true.append(true_label)
        y_pred.append(pred_label)

    print("\nConfusion Matrix\n")

    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    print("\nClassification Report\n")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=INTENT_LABELS,
        )
    )


if __name__ == "__main__":
    main()