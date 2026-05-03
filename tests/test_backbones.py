"""Focused tests for standalone CNN, RNN, and MLP backbones."""

import torch

from models.cnn import CNN
from models.mlp import MLP
from models.rnn import RNN


def test_cnn_forward_with_residual_and_dropout_keeps_expected_shape():
    model = CNN(
        input_dim=5,
        hidden_dim=12,
        num_layers=2,
        kernel_size=3,
        output_dim=7,
        activation="silu",
        dropout=0.1,
        use_residual=True,
    )
    inputs = torch.randn(3, 11, 5)

    outputs = model(inputs)

    assert outputs.shape == (3, 11, 7)


def test_rnn_forward_with_residual_and_dropout_keeps_expected_shape():
    model = RNN(
        input_dim=5,
        hidden_dim=6,
        num_layers=2,
        output_dim=7,
        bidirectional=True,
        activation="silu",
        dropout=0.1,
        use_residual=True,
    )
    inputs = torch.randn(3, 11, 5)

    outputs = model(inputs)

    assert outputs.shape == (3, 11, 7)


def test_rnn_forward_with_lengths_restores_original_sequence_length():
    model = RNN(
        input_dim=5,
        hidden_dim=6,
        num_layers=2,
        output_dim=7,
        bidirectional=True,
        activation="silu",
        dropout=0.1,
        use_residual=True,
    )
    inputs = torch.randn(2, 8, 5)
    lengths = torch.tensor([8, 5], dtype=torch.long)

    outputs = model(inputs, lengths=lengths)

    assert outputs.shape == (2, 8, 7)
    assert torch.allclose(outputs[1, 5:], torch.zeros_like(outputs[1, 5:]), atol=1e-6)


def test_mlp_forward_with_residual_and_dropout_keeps_expected_shape():
    model = MLP(
        input_dim=5,
        hidden_dim=16,
        num_layers=3,
        output_dim=7,
        seq_len=9,
        activation="silu",
        dropout=0.1,
        use_residual=True,
    )
    inputs = torch.randn(4, 9, 5)

    outputs = model(inputs)

    assert outputs.shape == (4, 9, 7)
