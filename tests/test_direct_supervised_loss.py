"""Focused loss tests for direct supervised ARC models."""

import torch
import torch.nn.functional as F

from models.direct_supervised_lightning import DirectSupervisedLightning


def build_direct_multiclass_model() -> DirectSupervisedLightning:
    return DirectSupervisedLightning(
        backbone_model={
            "name": "rnn",
            "params": {
                "hidden_dim": 8,
                "num_layers": 1,
                "bidirectional": True,
            },
        },
        task_encoding={"embedding_dim": 8, "value_vocab_size": 11},
        input_dim=4,
        prediction_task="multiclass",
        num_classes=10,
        padding_idx=10,
        non_background_loss_weight=2.0,
    )


def test_multiclass_loss_doubles_non_background_pixels_and_ignores_padding():
    """Non-background targets should be weighted while padding stays excluded."""
    model = build_direct_multiclass_model()
    logits = torch.tensor(
        [
            [
                [1.2, 0.1, -0.5, 0.3, 0.0, -0.2, 0.4, -0.1, 0.2, -0.4],
                [0.0, -0.4, 1.4, 0.2, -0.3, 0.1, -0.1, 0.5, -0.2, 0.3],
                [0.7, 0.3, -0.2, 0.0, 0.5, -0.3, 0.2, -0.5, 0.1, -0.1],
                [-0.1, 0.2, 0.4, 1.1, -0.4, 0.0, 0.3, -0.2, 0.5, -0.5],
            ]
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([[0, 2, 10, 3]], dtype=torch.long)

    losses = F.cross_entropy(
        logits.permute(0, 2, 1),
        targets,
        ignore_index=10,
        reduction="none",
    )
    expected = (losses[0, 0] + (2.0 * losses[0, 1]) + (2.0 * losses[0, 3])) / 3

    assert torch.isclose(model.compute_loss(logits, targets), expected)
