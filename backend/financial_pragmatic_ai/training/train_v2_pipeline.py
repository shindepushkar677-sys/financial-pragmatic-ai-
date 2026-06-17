"""Unified V2 training pipeline for intent + conversation attention models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, TypedDict

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from financial_pragmatic_ai.models.conversation_attention_model import (
    ConversationAttentionModel,
    SIGNAL_TO_INDEX,
)
from financial_pragmatic_ai.models.finbert_intent_model import (
    INTENT_LABELS,
    FinBERTIntentModel,
    train_finbert_intent_model,
)
from financial_pragmatic_ai.models.speaker_embedding import SPEAKER_TO_INDEX


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PACKAGE_ROOT / "data" / "pragmatic_intent_dataset_clean.csv"
FINBERT_INTENT_PATH = PACKAGE_ROOT / "models" / "finbert_intent_v3"
CONVERSATION_ATTENTION_PATH = PACKAGE_ROOT / "models" / "conversation_attention.pt"
EMBEDDINGS_DIR = Path("./embeddings")

VALID_INTENTS = set(INTENT_LABELS)


class ConversationSequence(TypedDict):
    texts: List[str]
    intents: List[str]
    speakers: List[str]
    signal: str


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings_dir: Path):
        self.files = sorted(embeddings_dir.glob("*.pt"))
        if not self.files:
            raise ValueError(f"No embedding files found in: {embeddings_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        payload = torch.load(self.files[index], map_location="cpu")
        return payload["X"], payload["y"]


def _clean_row(row: pd.Series) -> Dict[str, str] | None:
    text = str(row.get("text", "")).strip()
    speaker = str(row.get("speaker", "")).strip().upper()
    intent = str(row.get("intent", "")).strip().upper()

    if not text or not speaker or not intent:
        return None
    if intent not in VALID_INTENTS:
        return None
    return {"text": text, "speaker": speaker, "intent": intent}


def _assign_signal(ceo_intent: str, cfo_intent: str, analyst_intent: str) -> str:
    if cfo_intent == "COST_PRESSURE":
        return "risk"
    if ceo_intent == "EXPANSION" and analyst_intent == "STRATEGIC_PROBING":
        return "neutral"
    if ceo_intent == "EXPANSION":
        return "growth"
    return "neutral"


def build_conversation_sequences(df: pd.DataFrame) -> List[ConversationSequence]:
    sequences: List[ConversationSequence] = []

    cleaned_rows = []
    for _, row in df.iterrows():
        cleaned = _clean_row(row)
        if cleaned is not None:
            cleaned_rows.append(cleaned)

    for i in range(len(cleaned_rows) - 2):
        window = cleaned_rows[i : i + 3]

        speakers = [row["speaker"] for row in window]

        if not all(role in speakers for role in ["CEO", "CFO", "ANALYST"]):
            continue

        texts = [row["text"] for row in window]
        intents = [row["intent"] for row in window]

        role_map = {row["speaker"]: row for row in window}
        ceo_intent = role_map["CEO"]["intent"]
        cfo_intent = role_map["CFO"]["intent"]
        analyst_intent = role_map["ANALYST"]["intent"]

        signal = _assign_signal(ceo_intent, cfo_intent, analyst_intent)

        sequences.append(
            {
                "texts": texts,
                "intents": intents,
                "speakers": speakers,
                "signal": signal,
            }
        )

    return sequences


def _speaker_vector_3d(speaker: str) -> torch.Tensor:
    _ = SPEAKER_TO_INDEX.get(speaker.upper(), SPEAKER_TO_INDEX["EXECUTIVE"])
    if speaker.upper() == "CEO":
        return torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    if speaker.upper() == "CFO":
        return torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    if speaker.upper() == "ANALYST":
        return torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    return torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)


def _prepare_embedding_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    for file in path.glob("*.pt"):
        file.unlink()


def stream_embeddings_to_disk(
    sequences: List[ConversationSequence],
    finbert_wrapper: FinBERTIntentModel,
    embeddings_dir: Path,
    batch_size: int = 32,
    max_length: int = 64,
):
    valid_sequences = [
        seq
        for seq in sequences
        if len(seq["texts"]) == 3 and len(seq["speakers"]) == 3 and seq["signal"] in SIGNAL_TO_INDEX
    ]

    _prepare_embedding_dir(embeddings_dir)

    encoder = finbert_wrapper.model
    tokenizer = finbert_wrapper.tokenizer
    encoder_device = finbert_wrapper.encoder_device

    encoder.to(encoder_device)
    encoder.eval()

    total = len(valid_sequences)
    if total == 0:
        return 0

    print(f"Streaming embeddings to disk: {total} sequences")
    file_count = 0

    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            seq_batch = valid_sequences[start:end]

            texts_flat: List[str] = []
            speakers_flat: List[str] = []
            label_indices: List[int] = []

            for seq in seq_batch:
                texts_flat.extend([str(t) for t in seq["texts"]])
                speakers_flat.extend([str(s) for s in seq["speakers"]])
                label_indices.append(SIGNAL_TO_INDEX[seq["signal"]])

            encoded = tokenizer(
                texts_flat,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(encoder_device)
            attention_mask = encoded["attention_mask"].to(encoder_device)

            outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu()

            n_seq = len(seq_batch)
            x_batch = torch.zeros(n_seq, 3, 771, dtype=torch.float32)
            for seq_idx in range(n_seq):
                for seg_idx in range(3):
                    emb_idx = seq_idx * 3 + seg_idx
                    speaker_vec = _speaker_vector_3d(speakers_flat[emb_idx])
                    x_batch[seq_idx, seg_idx] = torch.cat([cls_embeddings[emb_idx], speaker_vec], dim=-1)

            y_batch = torch.tensor(label_indices, dtype=torch.long)
            out_file = embeddings_dir / f"batch_{file_count:06d}.pt"
            torch.save({"X": x_batch, "y": y_batch}, out_file)
            file_count += 1

            if file_count % 50 == 0 or end == total:
                print(f"  Embedded {end}/{total} sequences ({file_count} files)")

            del outputs, cls_embeddings, input_ids, attention_mask, encoded, x_batch, y_batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return file_count


def train_conversation_from_embedding_files(
    embeddings_dir: Path,
    output_path: Path | str,
    epochs: int = 8,
    learning_rate: float = 1e-4,
):
    classifier_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = classifier_device.type == "cuda"

    dataset = EmbeddingDataset(embeddings_dir)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=2,
        pin_memory=pin_memory,
    )

    model = ConversationAttentionModel(input_size=771).to(classifier_device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    print(f"Training conversation model on {len(dataset)} embedding files")
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for step, (x_batch, y_batch) in enumerate(loader, start=1):
            x_batch = x_batch.squeeze(0).to(classifier_device, non_blocking=pin_memory)
            y_batch = y_batch.squeeze(0).to(classifier_device, non_blocking=pin_memory)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

            if step % 50 == 0:
                print(f"  Epoch {epoch + 1}/{epochs} Step {step}/{len(loader)} Loss {loss.item():.4f}")

        avg_loss = total_loss / max(len(loader), 1)
        print(f"Epoch {epoch + 1}/{epochs} Loss {avg_loss:.4f}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output)
    print(f"Saved conversation attention model to: {output}")


def main() -> None:
    torch.set_num_threads(os.cpu_count() or 1)

    print("=== V2 Training Pipeline ===")
    print(f"Dataset path: {DATA_PATH}")
    print(f"FinBERT intent output: {FINBERT_INTENT_PATH}")
    print(f"Conversation attention output: {CONVERSATION_ATTENTION_PATH}")
    print(f"Embedding cache dir: {EMBEDDINGS_DIR.resolve()}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH).sample(10000, random_state=42)
    print(f"Loaded pragmatic dataset rows: {len(df)}")

    print("\n[1/4] Training FinBERT intent model...")
    finbert_wrapper = train_finbert_intent_model(
        dataset_path=DATA_PATH,
        output_path=FINBERT_INTENT_PATH,
        max_length=64,
        batch_size=32,
        epochs=4,
        learning_rate=2e-5,
    )
    print(f"Saved FinBERT intent model: {FINBERT_INTENT_PATH}")

    print("\n[2/4] Building conversation sequences (CEO -> CFO -> ANALYST)...")
    sequences = build_conversation_sequences(df)
    print(f"Conversation sequence dataset size: {len(sequences)}")
    if sequences:
        print("Sample sequence:")
        print(sequences[0])
    else:
        print("No valid sequences found in dataset.")

    print("\n[3/4] Streaming speaker-aware embeddings to disk...")
    file_count = stream_embeddings_to_disk(
        sequences=sequences,
        finbert_wrapper=finbert_wrapper,
        embeddings_dir=EMBEDDINGS_DIR,
        batch_size=32,
        max_length=64,
    )

    if file_count == 0:
        print(
            "[WARN] No valid embedding sequences available. "
            "Creating a minimal neutral bootstrap sample to avoid crash."
        )
        neutral_idx = SIGNAL_TO_INDEX["neutral"]
        torch.save(
            {
                "X": torch.zeros(1, 3, 771, dtype=torch.float32),
                "y": torch.tensor([neutral_idx], dtype=torch.long),
            },
            EMBEDDINGS_DIR / "batch_000000.pt",
        )
        file_count = 1

    print(f"Embedding files created: {file_count}")

    print("\n[4/4] Training conversation attention model...")
    train_conversation_from_embedding_files(
        embeddings_dir=EMBEDDINGS_DIR,
        output_path=CONVERSATION_ATTENTION_PATH,
        epochs=8,
        learning_rate=1e-4,
    )
    print(f"Saved conversation attention model: {CONVERSATION_ATTENTION_PATH}")

    print("\nV2 pipeline complete.")


if __name__ == "__main__":
    main()
