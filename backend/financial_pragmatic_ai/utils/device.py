"""Device utilities for Apple Silicon and CPU fallback."""

import torch


def get_torch_device() -> torch.device:
    """Return MPS when available, otherwise CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
