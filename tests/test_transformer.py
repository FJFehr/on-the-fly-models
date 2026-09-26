"""Tests for the Canon layer and the Transformer."""

import pytest
import torch

from models.canon import CanonConv
from models.transformer import Transformer


def test_canon_conv_is_centred_over_kernel_width():
    """A change at one position reaches exactly kernel_size // 2 neighbours on each side."""
    torch.manual_seed(0)
    conv = CanonConv(dim=3, kernel_size=5)
    x = torch.zeros(1, 11, 3)
    bumped = x.clone()
    bumped[0, 5] = 1.0

    changed = (conv(bumped) - conv(x)).abs().sum(dim=-1)[0] > 0
    assert changed.nonzero().flatten().tolist() == [3, 4, 5, 6, 7]


def test_transformer_output_shapes():
    x = torch.randn(2, 7, 4)
    with_head = Transformer(input_dim=4, hidden_dim=8, num_layers=3, num_heads=2, output_dim=10)
    without_head = Transformer(input_dim=4, hidden_dim=8, num_layers=3, num_heads=2)
    assert with_head(x).shape == (2, 7, 10)
    assert without_head(x).shape == (2, 7, 8)


def test_empty_canon_set_has_no_canon_layers():
    model = Transformer(input_dim=4, hidden_dim=4, num_layers=2, num_heads=1, canon_set="")
    assert not any("canon" in name for name, _ in model.named_parameters())


def test_invalid_canon_set_is_rejected():
    with pytest.raises(ValueError, match="canon_set"):
        Transformer(input_dim=4, hidden_dim=4, num_layers=1, num_heads=1, canon_set="AX")


def test_fused_and_manual_attention_agree():
    """The hypernetwork switches the target to manual attention for vmap."""
    torch.manual_seed(0)
    model = Transformer(input_dim=4, hidden_dim=4, num_layers=2, num_heads=2, output_dim=3).eval()
    x = torch.randn(2, 6, 4)
    fused = model(x)
    for module in model.modules():
        if hasattr(module, "flash"):
            module.flash = False
    torch.testing.assert_close(model(x), fused)
