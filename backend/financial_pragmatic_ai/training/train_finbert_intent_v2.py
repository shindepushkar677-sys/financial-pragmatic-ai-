"""Train and save a full 4-class FinBERT intent model (HF format)."""

from __future__ import annotations

from pathlib import Path

from financial_pragmatic_ai.models.finbert_intent_model import train_finbert_intent_model


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PACKAGE_ROOT / "data" / "pragmatic_intent_dataset_clean.csv"
OUTPUT_DIR = PACKAGE_ROOT / "models" / "finbert_intent_v3"


def main() -> None:
    train_finbert_intent_model(
        dataset_path=DATASET_PATH,
        output_path=OUTPUT_DIR,
        max_length=128,
        batch_size=16,
        epochs=3,
        learning_rate=2e-5,
    )


if __name__ == "__main__":
    main()
