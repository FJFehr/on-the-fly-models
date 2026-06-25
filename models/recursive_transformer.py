"""Weight-shared transformer backbone (Universal Transformer style).

A standard transformer block of `num_layers` encoder layers is applied
`n_loops` times in the forward pass with shared weights across loops.
Parameter count is identical to a plain Transformer with the same num_layers —
the difference is computational depth, not parameter count.
"""

import torch
import torch.nn as nn

from models.activations import resolve_activation_fn


class RecursiveTransformer(nn.Module):
    """Transformer encoder whose block is applied n_loops times with shared weights.

    The shared block contains num_layers independent TransformerEncoderLayers.
    Each loop iteration applies every layer in the block in order, so the total
    effective depth is num_layers × n_loops, while the parameter count equals
    a plain num_layers-deep transformer.

    Interface matches models.transformer.Transformer:
        forward(inputs, src_key_padding_mask=None) -> (batch, seq_len, hidden_dim)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        output_dim: int,
        n_loops: int = 4,
        dropout: float = 0.0,
        activation: str = "relu",
        bias: bool = False,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            msg = "hidden_dim must be divisible by num_heads."
            raise ValueError(msg)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.n_loops = n_loops
        self.dropout = dropout
        self.activation = activation

        # Static input stage: lifts the input into the recurrent hidden space (never looped).
        self.input_projection = nn.Linear(input_dim, hidden_dim, bias=bias)

        # Recurrent core: num_layers encoder layers applied n_loops times with shared weights.
        self.block = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=4 * hidden_dim,
                dropout=dropout,
                activation=resolve_activation_fn(activation),
                batch_first=True,
                norm_first=True,
                bias=bias,
            )
            for _ in range(num_layers)
        ])
        # Final norm applied once after all loops — mirrors nn.TransformerEncoder(norm=...).
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.input_projection(inputs)
        for _ in range(self.n_loops):
            for layer in self.block:
                h = layer(h, src_key_padding_mask=src_key_padding_mask)
        return self.final_norm(h)

    def __repr__(self) -> str:
        return (
            f"RecursiveTransformer(input={self.input_dim}, hidden={self.hidden_dim}, "
            f"layers={self.num_layers}, heads={self.num_heads}, "
            f"n_loops={self.n_loops}, dropout={self.dropout})"
        )
