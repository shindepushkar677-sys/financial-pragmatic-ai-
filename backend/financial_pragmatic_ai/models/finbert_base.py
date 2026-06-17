"""Load the FinBERT base model and tokenizer."""

from typing import Tuple

from transformers import AutoModel, AutoTokenizer


def load_finbert(device: str = "cpu") -> Tuple[AutoTokenizer, AutoModel]:
    """
    Load ProsusAI/finbert tokenizer and model on the requested device.

    Args:
        device: Target device ("cpu" or "mps").

    Returns:
        (tokenizer, model)
    """
    if device not in {"cpu", "mps"}:
        raise ValueError("device must be 'cpu' or 'mps'")

    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModel.from_pretrained("ProsusAI/finbert")
    model.to(device)
    model.eval()
    return tokenizer, model
