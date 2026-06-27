"""Tests for the CanonTransformer and CanonRecursiveTransformer modules."""

import pytest
import torch

from models.canon_transformer import CanonRecursiveTransformer, CanonTransformer

# Small but valid dims for fast tests.
_D, _H, _L, _N = 16, 2, 2, 5  # hidden_dim, num_heads, num_layers, output_dim


def _make(**overrides) -> CanonTransformer:
    defaults = dict(
        input_dim=5,
        hidden_dim=_D,
        num_layers=_L,
        num_heads=_H,
        output_dim=_N,
    )
    return CanonTransformer(**{**defaults, **overrides})


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------


def test_output_shape_ac():
    model = _make(canon_set="AC")
    out = model(torch.randn(3, 11, 5))
    assert out.shape == (3, 11, _N)


def test_output_shape_abcd():
    model = _make(canon_set="ABCD")
    out = model(torch.randn(2, 9, 5))
    assert out.shape == (2, 9, _N)


def test_output_shape_no_canon():
    """Empty canon_set degrades to a plain transformer."""
    model = _make(canon_set="")
    out = model(torch.randn(2, 7, 5))
    assert out.shape == (2, 7, _N)


def test_output_shape_single_position_b():
    model = _make(canon_set="B")
    out = model(torch.randn(2, 6, 5))
    assert out.shape == (2, 6, _N)


def test_output_shape_single_position_d():
    model = _make(canon_set="D")
    out = model(torch.randn(2, 6, 5))
    assert out.shape == (2, 6, _N)


# ---------------------------------------------------------------------------
# Output head control
# ---------------------------------------------------------------------------


def test_without_output_head_returns_hidden_states():
    model = _make(canon_set="AC", use_output_head=False)
    out = model(torch.randn(2, 9, 5))
    assert out.shape == (2, 9, _D)


# ---------------------------------------------------------------------------
# Construction guards
# ---------------------------------------------------------------------------


def test_hidden_dim_not_divisible_by_heads_raises():
    with pytest.raises(ValueError, match="hidden_dim must be divisible by num_heads"):
        _make(hidden_dim=10, num_heads=3)


def test_invalid_canon_set_raises():
    with pytest.raises(ValueError, match="invalid positions"):
        _make(canon_set="XY")


def test_invalid_canon_set_with_mixed_valid_invalid_raises():
    with pytest.raises(ValueError, match="invalid positions"):
        _make(canon_set="AX")


# ---------------------------------------------------------------------------
# Parameter count
# ---------------------------------------------------------------------------


def _count_params(model: CanonTransformer) -> int:
    return sum(p.numel() for p in model.parameters())


def test_canon_set_ac_has_fewer_params_than_abcd():
    ac = _make(canon_set="AC")
    abcd = _make(canon_set="ABCD")
    assert _count_params(ac) < _count_params(abcd)


def test_empty_canon_set_has_fewer_params_than_ac():
    no_canon = _make(canon_set="")
    ac = _make(canon_set="AC")
    assert _count_params(no_canon) < _count_params(ac)


def test_canon_param_counts_are_additive():
    """Each new active position adds exactly hidden_size * kernel_size params."""
    K = 4
    a = _make(canon_set="A", canon_kernel=K)
    ac = _make(canon_set="AC", canon_kernel=K)
    abcd = _make(canon_set="ABCD", canon_kernel=K)

    # A and C each add hidden_dim * K params (× num_layers).
    # B adds 3*hidden_dim * K params (× num_layers).
    # D adds 4*hidden_dim * K params (× num_layers).
    diff_a_to_ac = _count_params(ac) - _count_params(a)
    expected_c = _L * _D * K
    assert diff_a_to_ac == expected_c, f"AC − A should be {expected_c}, got {diff_a_to_ac}"

    diff_ac_to_abcd = _count_params(abcd) - _count_params(ac)
    expected_bd = _L * (3 * _D * K + 4 * _D * K)
    assert diff_ac_to_abcd == expected_bd, f"ABCD − AC should be {expected_bd}, got {diff_ac_to_abcd}"


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_gradients_flow_with_all_positions():
    model = _make(canon_set="ABCD")
    x = torch.randn(2, 8, 5, requires_grad=True)
    model(x).sum().backward()
    assert x.grad is not None
    # All canon layers should have received gradients.
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def test_causal_flag_accepted():
    model = _make(canon_set="AC", causal=True, block_size=64)
    out = model(torch.randn(2, 10, 5))
    assert out.shape == (2, 10, _N)


def test_kernel_size_1_runs():
    """kernel_size=1 makes canon layers pointwise — still valid."""
    model = _make(canon_set="ABCD", canon_kernel=1)
    out = model(torch.randn(2, 8, 5))
    assert out.shape == (2, 8, _N)


def test_canon_residual_false_runs():
    model = _make(canon_set="AC", canon_residual=False)
    out = model(torch.randn(2, 8, 5))
    assert out.shape == (2, 8, _N)


def test_canon_activation_false_runs():
    model = _make(canon_set="AC", canon_activation=False)
    out = model(torch.randn(2, 8, 5))
    assert out.shape == (2, 8, _N)


def test_repr_contains_canon_info():
    model = _make(canon_set="ABCD", canon_kernel=4)
    r = repr(model)
    assert "ABCD" in r
    assert "4" in r


# ---------------------------------------------------------------------------
# CanonRecursiveTransformer tests
# ---------------------------------------------------------------------------


def _make_recursive(**overrides) -> CanonRecursiveTransformer:
    defaults = dict(
        input_dim=5,
        hidden_dim=_D,
        num_layers=1,
        num_heads=_H,
        output_dim=_N,
        n_loops=2,
    )
    return CanonRecursiveTransformer(**{**defaults, **overrides})


def test_canon_recursive_output_shape():
    model = _make_recursive(canon_set="AC")
    out = model(torch.randn(3, 11, 5))
    assert out.shape == (3, 11, _D)  # no output head; returns hidden states


def test_canon_recursive_output_shape_abcd():
    model = _make_recursive(canon_set="ABCD")
    out = model(torch.randn(2, 9, 5))
    assert out.shape == (2, 9, _D)


def test_canon_recursive_no_canon_runs():
    """canon_set='' gives a plain recursive transformer with no canon overhead."""
    model = _make_recursive(canon_set="")
    out = model(torch.randn(2, 7, 5))
    assert out.shape == (2, 7, _D)


def test_canon_recursive_invalid_canon_set_raises():
    with pytest.raises(ValueError, match="invalid positions"):
        _make_recursive(canon_set="XZ")


def test_canon_recursive_param_count_equals_single_block():
    """Weight sharing: total params should equal one CanonBlock (+ input_proj + final_norm)."""
    model_1loop = _make_recursive(n_loops=1)
    model_4loop = _make_recursive(n_loops=4)
    assert _count_params(model_1loop) == _count_params(model_4loop)


def test_canon_recursive_loops_add_no_params():
    """Increasing n_loops must not increase the parameter count."""
    params_2 = _count_params(_make_recursive(n_loops=2))
    params_8 = _count_params(_make_recursive(n_loops=8))
    assert params_2 == params_8


def test_canon_recursive_gradients_flow():
    model = _make_recursive(canon_set="ABCD")
    x = torch.randn(2, 8, 5, requires_grad=True)
    model(x).sum().backward()
    assert x.grad is not None
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_canon_recursive_src_key_padding_mask_accepted():
    """forward must accept src_key_padding_mask without error (ignored internally)."""
    model = _make_recursive(canon_set="AC")
    x = torch.randn(2, 8, 5)
    mask = torch.zeros(2, 8, dtype=torch.bool)
    out = model(x, src_key_padding_mask=mask)
    assert out.shape == (2, 8, _D)


def test_canon_recursive_repr_contains_key_info():
    model = _make_recursive(canon_set="ABCD", n_loops=4)
    r = repr(model)
    assert "ABCD" in r
    assert "4" in r


# ---------------------------------------------------------------------------
# Non-causal canon tests
# ---------------------------------------------------------------------------


def test_non_causal_canon_transformer_output_shape():
    model = _make(canon_set="ABCD", canon_causal=False)
    out = model(torch.randn(2, 9, 5))
    assert out.shape == (2, 9, _N)


def test_non_causal_canon_recursive_output_shape():
    model = _make_recursive(canon_set="ABCD", canon_causal=False)
    out = model(torch.randn(2, 9, 5))
    assert out.shape == (2, 9, _D)


def test_non_causal_has_same_param_count_as_causal():
    """Switching causal=False doesn't add or remove parameters."""
    causal_params = _count_params(_make(canon_set="ABCD", canon_causal=True))
    non_causal_params = _count_params(_make(canon_set="ABCD", canon_causal=False))
    assert causal_params == non_causal_params
