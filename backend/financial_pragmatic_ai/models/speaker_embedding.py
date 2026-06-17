"""Speaker-role embeddings for pragmatic financial modeling."""

import torch
from torch import nn


SPEAKER_TO_INDEX = {
    "CEO": 0,
    "CFO": 1,
    "ANALYST": 2,
    "EXECUTIVE": 3,
}

EMBEDDING_DIM = 32
NUM_SPEAKERS = len(SPEAKER_TO_INDEX)

speaker_embedding_layer = nn.Embedding(NUM_SPEAKERS, EMBEDDING_DIM)


def get_speaker_embedding(speaker: str) -> torch.Tensor:
    """
    Convert a speaker label to an embedding tensor.

    Unknown speakers default to EXECUTIVE.
    """
    speaker_key = speaker.strip().upper()
    speaker_idx = SPEAKER_TO_INDEX.get(speaker_key, SPEAKER_TO_INDEX["EXECUTIVE"])
    speaker_tensor = torch.tensor([speaker_idx], dtype=torch.long)
    return speaker_embedding_layer(speaker_tensor)


if __name__ == "__main__":
    speaker = "CEO"
    embedding = get_speaker_embedding(speaker)
    print(f"Speaker: {speaker}")
    print(f"Embedding shape: {tuple(embedding.shape)}")
