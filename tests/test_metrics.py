"""Focused tests for the sequence-level metric contract used in training.

The training loop currently relies on exact-match accuracy as the key sequence
metric, so this file keeps a tiny but explicit contract suite around that
behaviour instead of broad, weak coverage across every helper metric.
"""

import torch

from models.metrics import exact_match_accuracy


def test_exact_match_accuracy_is_one_for_perfect_predictions():
    """A fully correct batch should score the maximum exact-match accuracy."""

    targets = torch.tensor([[1, 0, 1], [0, 1, 0]])
    preds = torch.tensor([[1, 0, 1], [0, 1, 0]])

    assert exact_match_accuracy(targets, preds).item() == 1.0


def test_exact_match_accuracy_is_zero_when_one_cell_is_wrong_in_single_sample():
    """One wrong position must invalidate the whole sample under exact match."""

    targets = torch.tensor([[1, 0, 1]])
    preds = torch.tensor([[1, 1, 1]])

    assert exact_match_accuracy(targets, preds).item() == 0.0


def test_exact_match_accuracy_handles_mixed_batches():
    """Mixed batches should average sequence-level correctness across samples."""

    targets = torch.tensor([[1, 0, 1], [0, 1, 0]])
    preds = torch.tensor([[1, 0, 1], [0, 0, 0]])

    assert exact_match_accuracy(targets, preds).item() == 0.5


def test_exact_match_accuracy_treats_multiclass_sequences_the_same_way():
    """One wrong multiclass token should still fail sequence-level exact match."""

    targets = torch.tensor([[7, 2, 1], [3, 4, 5]])
    preds = torch.tensor([[7, 2, 1], [3, 0, 5]])

    assert exact_match_accuracy(targets, preds).item() == 0.5
