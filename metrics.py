"""Sequence metrics for ARC training."""

import torch


def accuracy(y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    """Calculate elementwise accuracy across the full batch."""
    return (y == y_hat).float().mean()


def exact_match_accuracy(y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    """Calculate sequence-level exact-match accuracy."""
    matches = (y == y_hat).all(dim=1)
    return matches.float().mean()
