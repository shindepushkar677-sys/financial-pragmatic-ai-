"""Pragmatic attention layer for identifying important financial signals."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PragmaticAttention(nn.Module):
    """Attention over sequence embeddings with input dimension 512."""

    def __init__(self) -> None:
        super().__init__()
        self.attention = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, 512)

        Returns:
            output: Tensor of shape (batch_size, 512)
            attention_weights: Tensor of shape (batch_size, seq_len, 1)
        """
        scores = self.attention(x)
        weights = F.softmax(scores, dim=1)
        output = torch.sum(weights * x, dim=1)
        return output, weights


if __name__ == "__main__":
    layer = PragmaticAttention()
    sample = torch.randn(2, 10, 512)
    output, attention_weights = layer(sample)
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Attention weight shape: {tuple(attention_weights.shape)}")
