"""Build sentence-level pragmatic intent training data from transcript JSONL."""

import csv
import json
import re
from pathlib import Path
from typing import Dict, List

import nltk
from nltk.tokenize import sent_tokenize


INPUT_PATH = (
    Path(__file__).resolve().parent
    / "pragmatic_training_dataset"
    / "combined_pragmatic_transcripts.jsonl"
)
OUTPUT_PATH = Path(__file__).resolve().parent / "pragmatic_intent_dataset.csv"

CEO_INDICATORS = [
    "ceo",
    "chief executive officer",
    "president",
    "lisa",
    "executive",
]
CFO_INDICATORS = [
    "cfo",
    "chief financial officer",
    "finance",
    "margin",
    "balance sheet",
]
ANALYST_INDICATORS = ["could you", "can you", "how do you", "what about"]

EXPANSION_KEYWORDS = ["growth", "increase", "demand", "expansion", "strong revenue"]
COST_PRESSURE_KEYWORDS = ["margin decline", "cost", "expenses", "supply cost"]


def ensure_nltk_punkt() -> None:
    """Ensure punkt tokenizer model is available."""
    try:
        sent_tokenize("Test sentence.")
    except LookupError:
        nltk.download("punkt", quiet=True)
        # Some NLTK versions separate these resources.
        try:
            sent_tokenize("Test sentence.")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)


def load_records(path: Path) -> List[Dict]:
    """Load JSONL records into memory."""
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def detect_speaker(sentence: str) -> str:
    """Infer speaker role from sentence-level heuristics."""
    text = sentence.lower().strip()

    if text.endswith("?") or any(phrase in text for phrase in ANALYST_INDICATORS):
        return "ANALYST"
    if any(indicator in text for indicator in CFO_INDICATORS):
        return "CFO"
    if any(indicator in text for indicator in CEO_INDICATORS):
        return "CEO"
    return "EXECUTIVE"


def assign_intent(speaker: str, sentence: str) -> str:
    """Assign pragmatic intent using rule-based heuristics."""
    text = sentence.lower()
    if any(keyword in text for keyword in EXPANSION_KEYWORDS):
        return "EXPANSION"
    if any(keyword in text for keyword in COST_PRESSURE_KEYWORDS):
        return "COST_PRESSURE"
    if speaker == "ANALYST":
        return "STRATEGIC_PROBING"
    return "GENERAL_UPDATE"


def build_dataset(input_path: Path, output_path: Path) -> int:
    """Build and save sentence-level pragmatic dataset. Returns row count."""
    ensure_nltk_punkt()
    records = load_records(input_path)
    rows: List[Dict[str, str]] = []

    for record in records:
        text = str(record.get("text", "")).strip()
        if not text:
            continue

        for sentence in sent_tokenize(text):
            clean_sentence = re.sub(r"\s+", " ", sentence).strip()
            if len(clean_sentence) < 10:
                continue

            speaker = detect_speaker(clean_sentence)
            intent = assign_intent(speaker, clean_sentence)
            rows.append({"text": clean_sentence, "speaker": speaker, "intent": intent})

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["text", "speaker", "intent"])
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def main() -> None:
    dataset_size = build_dataset(INPUT_PATH, OUTPUT_PATH)
    print(f"Saved dataset to: {OUTPUT_PATH}")
    print(f"Dataset size: {dataset_size}")


if __name__ == "__main__":
    main()
