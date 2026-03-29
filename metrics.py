"""Sequence metrics shared by binary and multiclass ARC training."""

import torch


def accuracy(y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    """Calculate elementwise accuracy across the full batch."""
    return (y == y_hat).float().mean()


def exact_match_accuracy(y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    """Calculate sequence-level exact-match accuracy."""
    matches = (y == y_hat).all(dim=1)
    return matches.float().mean()


def precision(y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    """Calculate binary precision for positive class 1."""
    tp = torch.sum((y == 1) & (y_hat == 1))
    fp = torch.sum((y == 0) & (y_hat == 1))
    return tp / (tp + fp + 1e-10)


def recall(y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    """Calculate binary recall for positive class 1."""
    tp = torch.sum((y == 1) & (y_hat == 1))
    fn = torch.sum((y == 1) & (y_hat == 0))
    return tp / (tp + fn + 1e-10)


def micro_f1(y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    """Calculate binary micro-F1 for positive class 1."""
    tp = torch.sum((y == 1) & (y_hat == 1))
    fp = torch.sum((y == 0) & (y_hat == 1))
    fn = torch.sum((y == 1) & (y_hat == 0))
    prec = tp / (tp + fp + 1e-10)
    rec = tp / (tp + fn + 1e-10)
    return 2 * (prec * rec) / (prec + rec + 1e-10)


def macro_f1(y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    """Calculate binary macro-F1 across tensor columns."""
    tp = torch.sum((y == 1) & (y_hat == 1), dim=0)
    fp = torch.sum((y == 0) & (y_hat == 1), dim=0)
    fn = torch.sum((y == 1) & (y_hat == 0), dim=0)

    prec = tp / (tp + fp + 1e-10)
    rec = tp / (tp + fn + 1e-10)
    f1 = 2 * (prec * rec) / (prec + rec + 1e-10)
    f1[(prec + rec) == 0] = 0
    return f1.mean()
