"""Diagnostic utility to verify Apple MPS support in PyTorch."""

import platform

import torch


def main() -> None:
    print("Torch:", torch.__version__)
    print("Platform:", platform.platform())
    print("MPS built:", torch.backends.mps.is_built())
    print("MPS available:", torch.backends.mps.is_available())

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    x = torch.ones(1, device=device)
    print("Tensor device:", x.device)


if __name__ == "__main__":
    main()
