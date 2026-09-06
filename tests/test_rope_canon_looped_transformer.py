"""RoPECanonLoopedTransformer is the target/hypernetwork architecture in all 6 paper
experiments (models/rope_looped_transformer.py), yet had zero test coverage, direct or
indirect -- every existing hypermodel test builds its tiny models from simpler
transformer/rnn backbones for speed, never touching the real one. This is the one
high-signal check that it actually works: forward shape is correct and gradients reach
every parameter. Not a sweep over canon_set/n_loops/skip-flag combinations -- those
variations are already covered structurally by test_canon_layer.py/test_canon_transformer.py.
"""

import torch

from models.rope_looped_transformer import RoPECanonLoopedTransformer


def test_forward_and_backward_produce_finite_gradients_on_every_parameter():
    torch.manual_seed(0)
    model = RoPECanonLoopedTransformer(
        input_dim=4,
        hidden_dim=8,
        num_heads=2,
        output_dim=4,
        n_loops=2,
        canon_set="ABCD",
    )
    inputs = torch.randn(2, 5, 4)  # batch=2, seq_len=5, input_dim=4

    output = model(inputs)
    assert output.shape == (2, 5, 4)  # (batch, seq_len, output_dim)

    output.sum().backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} has a non-finite gradient"


def test_parameter_count_is_independent_of_n_loops():
    """The docstring's central design claim: looping the same shared block more times
    must not add parameters -- only n_loops iterations at forward time changes."""

    def build(n_loops: int) -> RoPECanonLoopedTransformer:
        return RoPECanonLoopedTransformer(
            input_dim=4, hidden_dim=8, num_heads=2, output_dim=4, n_loops=n_loops
        )

    params_at_2_loops = sum(p.numel() for p in build(2).parameters())
    params_at_5_loops = sum(p.numel() for p in build(5).parameters())

    assert params_at_2_loops == params_at_5_loops
