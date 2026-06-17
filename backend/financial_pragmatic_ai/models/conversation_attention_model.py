from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset


SIGNAL_LABELS = ["neutral", "risk", "growth"]
SIGNAL_TO_INDEX = {label: index for index, label in enumerate(SIGNAL_LABELS)}
INDEX_TO_SIGNAL = {index: label for label, index in SIGNAL_TO_INDEX.items()}

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[0] / "conversation_attention.pt"
DEFAULT_DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "conversation_signal_dataset.csv"


class ConversationAttentionModel(nn.Module):

    def __init__(self, input_size: int = 771, hidden_size: int = 256, num_heads: int = 4):
        super().__init__()
        self.projection = nn.Linear(input_size, hidden_size)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
        )
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Linear(hidden_size, len(SIGNAL_LABELS))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        x = self.projection(embeddings)
        attn_output, _ = self.attention(x, x, x)
        x = self.feedforward(attn_output)
        logits = self.classifier(x.mean(dim=1))
        return logits


class ConversationSequenceDataset(Dataset):

    def __init__(self, sequences: list[torch.Tensor], labels: list[int]):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.sequences[index], torch.tensor(self.labels[index], dtype=torch.long)


def _collate(batch):
    sequences, labels = zip(*batch)
    max_len = max(seq.shape[0] for seq in sequences)
    dim = sequences[0].shape[-1]

    padded = []
    for seq in sequences:
        if seq.shape[0] < max_len:
            pad = torch.zeros(max_len - seq.shape[0], dim, dtype=seq.dtype)
            seq = torch.cat([seq, pad], dim=0)
        padded.append(seq)

    return torch.stack(padded), torch.stack(labels)


def train_conversation_attention_model(
    embedding_sequences: list[torch.Tensor],
    signals: list[str],
    output_path: Path | str = DEFAULT_MODEL_PATH,
    batch_size: int = 8,
    epochs: int = 8,
    learning_rate: float = 1e-3,
):
    if len(embedding_sequences) == 0:
        raise ValueError("No embedding sequences were provided")

    labels = [SIGNAL_TO_INDEX[signal] for signal in signals]
    dataset = ConversationSequenceDataset(embedding_sequences, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=_collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConversationAttentionModel(input_size=embedding_sequences[0].shape[-1]).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

        avg_loss = total_loss / max(len(loader), 1)
        print(f"Epoch {epoch + 1}/{epochs} Loss {avg_loss:.4f}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output)
    print(f"Saved conversation attention model to: {output}")
    return model


def load_conversation_attention_model(
    model_path: Path | str = DEFAULT_MODEL_PATH,
    input_size: int = 771,
    device: torch.device | None = None,
):
    resolved_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConversationAttentionModel(input_size=input_size).to(resolved_device)
    path = Path(model_path)
    if not path.exists():
        return None
    state_dict = torch.load(path, map_location=resolved_device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_conversation_dataset(path: Path | str = DEFAULT_DATASET_PATH):
    resolved_path = Path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Dataset not found: {resolved_path}")
    return pd.read_csv(resolved_path)
