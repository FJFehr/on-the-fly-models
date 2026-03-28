# metrics.py
# ---------------------------------------------------------------------------
# Evaluation metrics for binary classification tasks.
#
# All functions operate on binary tensors of shape (N, C) where:
#   N = number of samples
#   C = number of classes (or sequence positions)
#
# Inputs:
#   y     — ground-truth binary labels  (0 or 1)
#   y_hat — predicted binary labels      (0 or 1)
#
# A small epsilon (1e-10) is added to every denominator to prevent
# division-by-zero when a class has no positive predictions or no
# positive ground-truth labels.
# ---------------------------------------------------------------------------

import torch


def accuracy(y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    """Calculate overall accuracy across all elements.

    Accuracy = (TP + TN) / (TP + TN + FP + FN)

    Args:
        y: Ground-truth binary tensor of shape (N, C).
        y_hat: Predicted binary tensor of shape (N, C).

    Returns:
        Scalar tensor with the accuracy value.
    """
    # Count true positives: model predicted 1 and ground truth is 1
    tp = torch.sum((y == 1) & (y_hat == 1))
    # Count true negatives: model predicted 0 and ground truth is 0
    tn = torch.sum((y == 0) & (y_hat == 0))
    # Count false positives: model predicted 1 but ground truth is 0
    fp = torch.sum((y == 0) & (y_hat == 1))
    # Count false negatives: model predicted 0 but ground truth is 1
    fn = torch.sum((y == 1) & (y_hat == 0))

    # Accuracy is the ratio of correct predictions to total predictions
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-10)
    return acc


def exact_match_accuracy(y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    """Calculate sequence-level exact-match accuracy.

    A sample is counted as correct only when every output position matches.
    """
    matches = (y == y_hat).all(dim=1)
    return matches.float().mean()


def precision(y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    """Calculate precision (positive predictive value).

    Precision = TP / (TP + FP)

    Args:
        y: Ground-truth binary tensor of shape (N, C).
        y_hat: Predicted binary tensor of shape (N, C).

    Returns:
        Scalar tensor with the precision value.
    """
    # True positives: correctly predicted positive
    tp = torch.sum((y == 1) & (y_hat == 1))
    # False positives: incorrectly predicted positive
    fp = torch.sum((y == 0) & (y_hat == 1))

    # Precision measures how many of the positive predictions are correct
    return tp / (tp + fp + 1e-10)


def recall(y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    """Calculate recall (sensitivity / true positive rate).

    Recall = TP / (TP + FN)

    Args:
        y: Ground-truth binary tensor of shape (N, C).
        y_hat: Predicted binary tensor of shape (N, C).

    Returns:
        Scalar tensor with the recall value.
    """
    # True positives: correctly predicted positive
    tp = torch.sum((y == 1) & (y_hat == 1))
    # False negatives: missed positive predictions
    fn = torch.sum((y == 1) & (y_hat == 0))

    # Recall measures how many of the actual positives were found
    return tp / (tp + fn + 1e-10)


def micro_f1(y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    """Calculate micro-averaged F1 score.

    Micro F1 aggregates TP, FP, FN across all classes before computing
    precision and recall, giving equal weight to every sample.

    Args:
        y: Ground-truth binary tensor of shape (N, C).
        y_hat: Predicted binary tensor of shape (N, C).

    Returns:
        Scalar tensor with the micro F1 score.
    """
    # Aggregate counts across all classes (micro-averaging)
    tp = torch.sum((y == 1) & (y_hat == 1))
    fp = torch.sum((y == 0) & (y_hat == 1))
    fn = torch.sum((y == 1) & (y_hat == 0))

    # Compute precision and recall from aggregated counts
    prec = tp / (tp + fp + 1e-10)
    rec = tp / (tp + fn + 1e-10)

    # F1 is the harmonic mean of precision and recall
    return 2 * (prec * rec) / (prec + rec + 1e-10)


def macro_f1(y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    """Calculate macro-averaged F1 score.

    Macro F1 computes F1 independently for each class (column) and then
    takes the unweighted mean, giving equal weight to every class
    regardless of its frequency.

    Args:
        y: Ground-truth binary tensor of shape (N, C).
        y_hat: Predicted binary tensor of shape (N, C).

    Returns:
        Scalar tensor with the macro F1 score.
    """
    # Compute per-class counts along dimension 0 (each column = one class)
    tp = torch.sum((y == 1) & (y_hat == 1), dim=0)
    fp = torch.sum((y == 0) & (y_hat == 1), dim=0)
    fn = torch.sum((y == 1) & (y_hat == 0), dim=0)

    # Per-class precision and recall
    prec = tp / (tp + fp + 1e-10)
    rec = tp / (tp + fn + 1e-10)

    # Per-class F1 (harmonic mean)
    f1 = 2 * (prec * rec) / (prec + rec + 1e-10)

    # Zero out F1 for classes where both precision and recall are zero,
    # since the harmonic mean formula produces a small non-zero artefact
    # due to the epsilon term
    f1[(prec + rec) == 0] = 0

    # Average across all classes (macro mean)
    return f1.mean()
