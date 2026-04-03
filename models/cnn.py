"""General-purpose 1D CNN encoder."""

import torch
import torch.nn as nn


class CNN(nn.Module):
    """1D CNN encoder with same-padding convolutions.

    Input:  (batch, seq_len, input_dim)
    Output: (batch, seq_len, output_dim)

    Works as a normal trained model or as a stateless target inside HyperModel —
    no changes to this class are needed for either mode. Sequence length is
    preserved through all layers via symmetric same-padding, which requires
    kernel_size to be odd.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        kernel_size: int = 3,
        output_dim: int = 1,
    ):
        if kernel_size % 2 == 0:
            msg = f"kernel_size must be odd for symmetric same-padding, got {kernel_size}."
            raise ValueError(msg)

        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.output_dim = output_dim

        padding = kernel_size // 2
        layers: list[nn.Module] = []
        in_channels = input_dim
        for _ in range(num_layers):
            layers.append(
                nn.Conv1d(in_channels, hidden_dim, kernel_size=kernel_size, padding=padding)
            )
            layers.append(nn.ReLU())
            in_channels = hidden_dim
        self.conv_stack = nn.Sequential(*layers)
        self.output_projection = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """inputs: (batch, seq_len, input_dim) → (batch, seq_len, output_dim)"""
        # Conv1d expects (batch, channels, seq_len).
        hidden = self.conv_stack(inputs.transpose(1, 2))
        # Return to (batch, seq_len, hidden_dim) before the output projection.
        return self.output_projection(hidden.transpose(1, 2))

    def __repr__(self) -> str:
        return (
            f"CNN(input={self.input_dim}, hidden={self.hidden_dim}, "
            f"layers={self.num_layers}, kernel={self.kernel_size}, output={self.output_dim})"
        )
