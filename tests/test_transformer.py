"""Focused tests for the standalone simplified transformer module."""

import pytest
import torch

from models.transformer import Transformer


def test_transformer_forward_keeps_expected_output_shape():
    model = Transformer(
        input_dim=5,
        hidden_dim=16,
        num_layers=2,
        num_heads=2,
        output_dim=7,
    )
    inputs = torch.randn(3, 11, 5)

    outputs = model(inputs)

    assert outputs.shape == (3, 11, 7)


def test_transformer_return_attentions_emits_one_matrix_per_layer():
    model = Transformer(
        input_dim=5,
        hidden_dim=16,
        num_layers=3,
        num_heads=4,
        output_dim=7,
    )
    inputs = torch.randn(2, 9, 5)

    outputs, attentions = model(inputs, return_attentions=True)

    assert outputs.shape == (2, 9, 7)
    assert len(attentions) == 3
    assert all(attention.shape == (2, 4, 9, 9) for attention in attentions)


def test_transformer_rejects_hidden_dim_not_divisible_by_num_heads():
    with pytest.raises(ValueError, match="hidden_dim must be divisible by num_heads"):
        Transformer(
            input_dim=5,
            hidden_dim=10,
            num_layers=2,
            num_heads=3,
            output_dim=7,
        )
