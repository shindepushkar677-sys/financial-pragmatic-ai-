"""Clean pragmatic intent dataset by removing earnings-call boilerplate."""

from pathlib import Path
import re

import pandas as pd


INPUT_PATH = Path(__file__).resolve().parent / "pragmatic_intent_dataset.csv"
OUTPUT_PATH = Path(__file__).resolve().parent / "pragmatic_intent_dataset_clean.csv"

BOILERPLATE_PHRASES = [
    "listen-only mode",
    "question-and-answer session",
    "investor relations",
    "forward-looking statements",
    "press release",
    "webcast",
    "earnings release",
    "risk factors",
    "sec filing",
]


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    original_size = len(df)

    if "text" not in df.columns:
        raise ValueError("Expected a 'text' column in pragmatic_intent_dataset.csv")

    df["text"] = df["text"].fillna("").astype(str).str.strip()

    boilerplate_regex = "|".join(re.escape(phrase) for phrase in BOILERPLATE_PHRASES)
    mask_boilerplate = df["text"].str.contains(
        boilerplate_regex, case=False, na=False, regex=True
    )
    df = df.loc[~mask_boilerplate]

    df = df.loc[df["text"].str.len() >= 20]
    df = df.drop_duplicates(subset=["text"])

    cleaned_size = len(df)
    rows_removed = original_size - cleaned_size

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Original dataset size: {original_size}")
    print(f"Cleaned dataset size: {cleaned_size}")
    print(f"Rows removed: {rows_removed}")


if __name__ == "__main__":
    main()
