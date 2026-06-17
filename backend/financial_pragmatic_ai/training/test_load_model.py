"""Quick FinBERT load and inference smoke test."""

import torch

from financial_pragmatic_ai.models.finbert_base import load_finbert
from financial_pragmatic_ai.utils.device import get_torch_device


def main() -> None:
    device = get_torch_device()
    tokenizer, model = load_finbert(device=device.type)

    text = "The company reported strong quarterly earnings and improved guidance."
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    print(f"Using device: {device}")
    print(f"Embedding shape: {tuple(outputs.last_hidden_state.shape)}")


if __name__ == "__main__":
    main()
