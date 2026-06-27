"""Tests for the standalone CanonLayer module."""

import pytest
import torch
import torch.nn as nn

from models.canon_layer import CanonLayer


# All tests use use_fast_conv1d=False so they run without causal-conv1d installed.
_FAST = False


# ---------------------------------------------------------------------------
# Shape and construction
# ---------------------------------------------------------------------------


def test_output_shape_matches_input():
    layer = CanonLayer(hidden_size=16, kernel_size=4, use_fast_conv1d=_FAST)
    x = torch.randn(2, 10, 16)
    out, state = layer(x)
    assert out.shape == x.shape
    assert state is None


def test_state_not_returned_by_default():
    layer = CanonLayer(hidden_size=8, kernel_size=4, use_fast_conv1d=_FAST)
    _, state = layer(torch.randn(1, 5, 8))
    assert state is None


def test_output_final_state_returns_correct_shape():
    K = 4
    D = 8
    B = 2
    layer = CanonLayer(hidden_size=D, kernel_size=K, use_fast_conv1d=_FAST)
    _, state = layer(torch.randn(B, 10, D), output_final_state=True)
    assert state is not None
    assert state.shape == (B, D, K)


def test_state_size_property():
    layer = CanonLayer(hidden_size=32, kernel_size=4, use_fast_conv1d=_FAST)
    assert layer.state_size == 32 * 4


def test_parameter_count_is_depthwise():
    """Depthwise conv has hidden_size × kernel_size weight params (no bias)."""
    D, K = 16, 4
    layer = CanonLayer(hidden_size=D, kernel_size=K, bias=False, use_fast_conv1d=_FAST)
    n_params = sum(p.numel() for p in layer.parameters())
    assert n_params == D * K


def test_bias_parameter_count():
    D, K = 16, 4
    layer = CanonLayer(hidden_size=D, kernel_size=K, bias=True, use_fast_conv1d=_FAST)
    n_params = sum(p.numel() for p in layer.parameters())
    assert n_params == D * K + D  # weights + one bias per channel


def test_invalid_activation_raises():
    with pytest.raises(ValueError, match="Unsupported activation"):
        CanonLayer(hidden_size=8, kernel_size=4, activation="relu", use_fast_conv1d=_FAST)


def test_repr_contains_key_info():
    layer = CanonLayer(hidden_size=16, kernel_size=4, activation="silu", residual=True, use_fast_conv1d=_FAST)
    r = repr(layer)
    assert "16" in r
    assert "4" in r
    assert "silu" in r


# ---------------------------------------------------------------------------
# Causality: output at t must not depend on positions > t
# ---------------------------------------------------------------------------


def test_causality_future_tokens_do_not_affect_past_output():
    torch.manual_seed(0)
    D, T = 12, 20
    layer = CanonLayer(
        hidden_size=D, kernel_size=4, activation=None, residual=False, use_fast_conv1d=_FAST
    )
    layer.eval()

    x = torch.randn(1, T, D)
    out, _ = layer(x)

    # Overwrite all tokens strictly after position t and re-run.
    t = T // 2
    x2 = x.clone()
    x2[:, t + 1 :, :] = torch.randn(1, T - t - 1, D)
    out2, _ = layer(x2)

    # Outputs up to (and including) t must be identical.
    assert torch.allclose(out[:, : t + 1, :], out2[:, : t + 1, :], atol=1e-6), (
        "Output at positions ≤ t changed when only positions > t were modified — "
        "the layer is not causal."
    )

    # Outputs after t may differ (sanity check that the test is non-trivial).
    assert not torch.allclose(out[:, t + 1 :, :], out2[:, t + 1 :, :], atol=1e-6), (
        "Outputs after t are identical despite different future inputs — "
        "the perturbation had no effect."
    )


def test_causality_kernel_size_1_is_pointwise():
    """With kernel_size=1, each output depends only on the same-position input."""
    torch.manual_seed(1)
    D = 8
    layer = CanonLayer(
        hidden_size=D, kernel_size=1, activation=None, residual=False, use_fast_conv1d=_FAST
    )
    layer.eval()

    x = torch.randn(1, 10, D)
    out, _ = layer(x)

    # Modify token at position 0 only; rest must be unchanged.
    x2 = x.clone()
    x2[:, 0, :] = torch.randn(D)
    out2, _ = layer(x2)

    assert torch.allclose(out[:, 1:, :], out2[:, 1:, :], atol=1e-6)


# ---------------------------------------------------------------------------
# Residual connection
# ---------------------------------------------------------------------------


def test_residual_true_adds_input_to_conv_output():
    torch.manual_seed(2)
    D = 16
    x = torch.randn(2, 8, D)

    layer_res = CanonLayer(hidden_size=D, kernel_size=4, residual=True, use_fast_conv1d=_FAST)
    layer_no = CanonLayer(hidden_size=D, kernel_size=4, residual=False, use_fast_conv1d=_FAST)
    layer_no.weight.data = layer_res.weight.data.clone()

    out_res, _ = layer_res(x)
    out_no, _ = layer_no(x)

    assert torch.allclose(out_res, x + out_no, atol=1e-6)


def test_residual_false_zero_weights_gives_zero_output():
    layer = CanonLayer(
        hidden_size=8, kernel_size=4, residual=False, activation=None, use_fast_conv1d=_FAST
    )
    nn.init.zeros_(layer.weight)
    x = torch.randn(2, 10, 8)
    out, _ = layer(x)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_residual_true_zero_weights_is_identity():
    layer = CanonLayer(
        hidden_size=8, kernel_size=4, residual=True, activation=None, use_fast_conv1d=_FAST
    )
    nn.init.zeros_(layer.weight)
    x = torch.randn(2, 10, 8)
    out, _ = layer(x)
    assert torch.allclose(out, x, atol=1e-6)


# ---------------------------------------------------------------------------
# Activation
# ---------------------------------------------------------------------------


def test_silu_activation_changes_output():
    torch.manual_seed(3)
    D = 8
    x = torch.randn(2, 10, D)

    layer_silu = CanonLayer(hidden_size=D, kernel_size=4, activation="silu", residual=False, use_fast_conv1d=_FAST)
    layer_none = CanonLayer(hidden_size=D, kernel_size=4, activation=None, residual=False, use_fast_conv1d=_FAST)
    layer_none.weight.data = layer_silu.weight.data.clone()

    out_silu, _ = layer_silu(x)
    out_none, _ = layer_none(x)

    assert not torch.allclose(out_silu, out_none, atol=1e-4)


def test_swish_alias_is_accepted():
    layer = CanonLayer(hidden_size=8, kernel_size=4, activation="swish", use_fast_conv1d=_FAST)
    out, _ = layer(torch.randn(1, 5, 8))
    assert out.shape == (1, 5, 8)


# ---------------------------------------------------------------------------
# Arithmetic correctness (no activation, all-ones weights)
# ---------------------------------------------------------------------------


def test_ones_weights_produce_sliding_sum():
    """
    With kernel_size=2, activation=None, residual=False, all weights=1:
      out[0] = x[0]           (only x[0] visible; left pad is 0)
      out[1] = x[0] + x[1]
      out[t] = x[t-1] + x[t]
    """
    D = 4
    layer = CanonLayer(
        hidden_size=D, kernel_size=2, activation=None, residual=False, use_fast_conv1d=_FAST
    )
    nn.init.ones_(layer.weight)

    x = torch.randn(1, 6, D)
    out, _ = layer(x)

    assert torch.allclose(out[:, 0, :], x[:, 0, :], atol=1e-6), "t=0 should equal x[0]"
    assert torch.allclose(out[:, 1, :], x[:, 0, :] + x[:, 1, :], atol=1e-6), "t=1 should equal x[0]+x[1]"
    for t in range(2, 6):
        assert torch.allclose(out[:, t, :], x[:, t - 1, :] + x[:, t, :], atol=1e-6), (
            f"t={t}: expected x[{t-1}]+x[{t}]"
        )


# ---------------------------------------------------------------------------
# Step-by-step decoding matches full-sequence forward pass
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel_size", [1, 2, 4])
@pytest.mark.parametrize("activation", [None, "silu"])
def test_step_mode_matches_full_forward(kernel_size, activation):
    """Token-by-token autoregressive decoding must match the full forward pass."""
    torch.manual_seed(42)
    B, T, D = 2, 12, 8
    layer = CanonLayer(
        hidden_size=D,
        kernel_size=kernel_size,
        activation=activation,
        residual=False,
        use_fast_conv1d=_FAST,
    )
    layer.eval()

    x = torch.randn(B, T, D)

    # Full forward in one shot.
    out_full, _ = layer(x)

    # Token-by-token with a rolling cache.
    cache = torch.zeros(B, D, kernel_size)
    steps = []
    for t in range(T):
        out_t, cache = layer(x[:, t : t + 1, :], cache=cache)
        steps.append(out_t)
    out_step = torch.cat(steps, dim=1)

    assert torch.allclose(out_full, out_step, atol=1e-5), (
        f"Step-by-step output differs from full forward (kernel_size={kernel_size}, "
        f"activation={activation})."
    )


# ---------------------------------------------------------------------------
# Mask
# ---------------------------------------------------------------------------


def test_mask_zeros_out_padded_positions():
    """Positions where mask=0 should be zeroed before the conv."""
    torch.manual_seed(5)
    D = 8
    layer = CanonLayer(
        hidden_size=D, kernel_size=2, activation=None, residual=False, use_fast_conv1d=_FAST
    )
    layer.eval()

    x = torch.randn(1, 6, D)
    mask = torch.ones(1, 6)
    mask[:, 3:] = 0  # pad positions 3..5

    out_masked, _ = layer(x, mask=mask)

    # Manually zero out those positions and forward.
    x_zeroed = x.clone()
    x_zeroed[:, 3:, :] = 0
    out_zeroed, _ = layer(x_zeroed)

    assert torch.allclose(out_masked, out_zeroed, atol=1e-6)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_gradients_flow_through_layer():
    layer = CanonLayer(hidden_size=8, kernel_size=4, use_fast_conv1d=_FAST)
    x = torch.randn(2, 10, 8, requires_grad=True)
    out, _ = layer(x)
    out.sum().backward()
    assert x.grad is not None
    assert layer.weight.grad is not None


# ---------------------------------------------------------------------------
# Additional tests: step mode with residual and bias, short sequences, dtype
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel_size", [1, 2, 4])
@pytest.mark.parametrize("activation", [None, "silu"])
def test_step_mode_matches_full_forward_with_residual(kernel_size, activation):
    """Step-by-step decoding must match full forward when residual=True."""
    torch.manual_seed(99)
    B, T, D = 2, 10, 8
    layer = CanonLayer(
        hidden_size=D,
        kernel_size=kernel_size,
        activation=activation,
        residual=True,
        use_fast_conv1d=_FAST,
    )
    layer.eval()

    x = torch.randn(B, T, D)
    out_full, _ = layer(x)

    cache = torch.zeros(B, D, kernel_size)
    steps = []
    for t in range(T):
        out_t, cache = layer(x[:, t : t + 1, :], cache=cache)
        steps.append(out_t)
    out_step = torch.cat(steps, dim=1)

    assert torch.allclose(out_full, out_step, atol=1e-5), (
        f"Step-by-step (residual=True) differs from full forward "
        f"(kernel_size={kernel_size}, activation={activation})."
    )


def test_step_mode_with_bias():
    """Bias term in _step must match full forward."""
    torch.manual_seed(7)
    B, T, D = 2, 8, 8
    layer = CanonLayer(
        hidden_size=D, kernel_size=2, bias=True, activation=None, residual=False,
        use_fast_conv1d=_FAST,
    )
    layer.eval()

    x = torch.randn(B, T, D)
    out_full, _ = layer(x)

    cache = torch.zeros(B, D, 2)
    steps = [layer(x[:, t : t + 1, :], cache=cache)[0] for t in range(T)]
    out_step = torch.cat(steps, dim=1)

    assert torch.allclose(out_full, out_step, atol=1e-5)


def test_sequence_shorter_than_kernel_size():
    """T < kernel_size should work fine; left-padding fills the missing history."""
    D, T, K = 8, 2, 4
    layer = CanonLayer(hidden_size=D, kernel_size=K, use_fast_conv1d=_FAST)
    x = torch.randn(3, T, D)
    out, _ = layer(x)
    assert out.shape == (3, T, D)


def test_float16_dtype_preserved():
    """Forward pass must not silently upcast to float32."""
    layer = CanonLayer(hidden_size=8, kernel_size=4, activation=None, use_fast_conv1d=_FAST)
    layer = layer.to(torch.float16)
    x = torch.randn(2, 6, 8, dtype=torch.float16)
    out, _ = layer(x)
    assert out.dtype == torch.float16


def test_mask_preserved_in_residual_path():
    """Bug fix: residual must use the masked input, not the original unmasked one."""
    torch.manual_seed(11)
    D = 8
    layer = CanonLayer(
        hidden_size=D, kernel_size=2, activation=None, residual=True, use_fast_conv1d=_FAST
    )
    layer.eval()

    x = torch.randn(1, 6, D)
    mask = torch.ones(1, 6)
    mask[:, 3:] = 0  # zero out positions 3–5

    out_masked, _ = layer(x, mask=mask)

    # Equivalent: manually zero the input and run without a mask.
    x_zeroed = x.clone()
    x_zeroed[:, 3:, :] = 0.0
    out_zeroed, _ = layer(x_zeroed)

    assert torch.allclose(out_masked, out_zeroed, atol=1e-6), (
        "Residual path uses unmasked input — the mask+residual bug is not fixed."
    )
