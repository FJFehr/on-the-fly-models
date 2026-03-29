"""CNN-specific target-model tests.

Shared validation-example logging behaviour is already covered in
``test_target_model.py`` through the base-class contract. This file stays small
and protects only the CNN-specific shape behaviour.
"""

import torch

from models.target_models.cnn import TargetCNNModelLightning


def test_target_cnn_forward_preserves_sequence_shape():
    """Verify the CNN keeps sequence length unchanged through its channel reshaping.

    This is the core architecture-specific invariant for the 1D CNN baseline:
    inputs are expanded to Conv1d format internally, but callers must still see
    logits with the original batch-by-sequence shape.
    """

    model = TargetCNNModelLightning(
        input_dim=33,
        output_dim=33,
        hidden_channels=52,
    )
    inputs = torch.randn(4, 33)

    logits = model(inputs)

    assert logits.shape == (4, 33)


def test_target_cnn_forward_preserves_sequence_shape_with_multiple_layers():
    model = TargetCNNModelLightning(
        input_dim=33,
        output_dim=33,
        hidden_channels=16,
        kernel_size=3,
        num_layers=3,
    )
    inputs = torch.randn(4, 33)

    logits = model(inputs)

    assert logits.shape == (4, 33)


def test_target_cnn_forward_preserves_sequence_shape_with_skip_connections():
    model = TargetCNNModelLightning(
        input_dim=33,
        output_dim=33,
        hidden_channels=16,
        kernel_size=3,
        num_layers=4,
        use_skip_connections=True,
    )
    inputs = torch.randn(4, 33)

    logits = model(inputs)

    assert logits.shape == (4, 33)


def test_target_cnn_forward_returns_multiclass_logits():
    model = TargetCNNModelLightning(
        input_dim=33,
        output_dim=33,
        hidden_channels=16,
        prediction_task="multiclass",
        num_classes=10,
    )
    inputs = torch.randint(0, 10, (4, 33)).float()

    logits = model(inputs)

    assert logits.shape == (4, 33, 10)
