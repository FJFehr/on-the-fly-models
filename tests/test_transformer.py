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


def test_transformer_forward_without_output_head_returns_hidden_states():
    model = Transformer(
        input_dim=5,
        hidden_dim=16,
        num_layers=3,
        num_heads=4,
        output_dim=7,
        use_output_head=False,
    )
    inputs = torch.randn(2, 9, 5)

    outputs = model(inputs)

    assert outputs.shape == (2, 9, 16)


def test_transformer_rejects_hidden_dim_not_divisible_by_num_heads():
    with pytest.raises(ValueError, match="hidden_dim must be divisible by num_heads"):
        Transformer(
            input_dim=5,
            hidden_dim=10,
            num_layers=2,
            num_heads=3,
            output_dim=7,
        )
