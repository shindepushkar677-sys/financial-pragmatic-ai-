"""Input fusion layer for FinBERT and speaker embeddings."""

import torch
import torch.nn as nn


class PragmaticInputLayer(nn.Module):
    """Combine FinBERT and speaker embeddings into a pragmatic representation."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(800, 512)
        self.activation = nn.ReLU()

    def forward(
        self, finbert_embedding: torch.Tensor, speaker_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            finbert_embedding: Tensor with last dim 768.
            speaker_embedding: Tensor with last dim 32.

        Returns:
            Tensor with last dim 512.
        """
        combined = torch.cat([finbert_embedding, speaker_embedding], dim=-1)
        projected = self.projection(combined)
        return self.activation(projected)


if __name__ == "__main__":
    layer = PragmaticInputLayer()
    dummy_finbert = torch.randn(2, 768)
    dummy_speaker = torch.randn(2, 32)
    output = layer(dummy_finbert, dummy_speaker)
    print(f"Output shape: {tuple(output.shape)}")
