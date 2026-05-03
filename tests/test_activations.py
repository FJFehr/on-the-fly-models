"""Tests for shared activation helpers."""

import pytest
import torch
import torch.nn.functional as F

from models.activations import SwiGLU, build_activation, resolve_activation_fn, swiglu


def test_swiglu_function_matches_expected_formula():
    gate = torch.randn(2, 3, 5)
    value = torch.randn(2, 3, 5)

    outputs = swiglu(gate, value)

    assert outputs.shape == gate.shape
    assert torch.allclose(outputs, F.silu(gate) * value)


def test_swiglu_module_matches_function():
    module = SwiGLU()
    gate = torch.randn(4, 7)
    value = torch.randn(4, 7)

    outputs = module(gate, value)

    assert torch.allclose(outputs, swiglu(gate, value))


def test_swiglu_rejects_mismatched_shapes():
    gate = torch.randn(2, 3, 5)
    value = torch.randn(2, 3, 4)

    with pytest.raises(ValueError, match="matching shapes"):
        swiglu(gate, value)


def test_build_activation_and_resolve_activation_support_silu():
    inputs = torch.randn(2, 3, 5)

    module_outputs = build_activation("silu")(inputs)
    function_outputs = resolve_activation_fn("silu")(inputs)

    assert torch.allclose(module_outputs, F.silu(inputs))
    assert torch.allclose(function_outputs, F.silu(inputs))


def test_build_activation_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unsupported activation"):
        build_activation("bogus")
