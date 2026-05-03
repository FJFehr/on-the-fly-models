"""General-purpose 1D CNN encoder."""

import torch
import torch.nn as nn

from models.activations import build_activation


class ConvSiLUBlock(nn.Module):
    """Same-padding convolution block with optional residual update."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout: float,
        use_residual: bool,
        activation: str,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.activation = build_activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.use_residual = use_residual
        if use_residual and in_channels != out_channels:
            self.residual_projection = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            )
        else:
            self.residual_projection = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        update = self.activation(self.conv(inputs))
        update = self.dropout(update)
        if not self.use_residual:
            return update
        return self.residual_projection(inputs) + update


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
        dropout: float = 0.0,
        use_residual: bool = False,
        activation: str = "silu",
    ):
        if kernel_size % 2 == 0:
            msg = f"kernel_size must be odd for symmetric same-padding, got {kernel_size}."
            raise ValueError(msg)
        if num_layers < 1:
            msg = "num_layers must be >= 1."
            raise ValueError(msg)
        if not 0.0 <= dropout < 1.0:
            msg = f"dropout must be in [0.0, 1.0), got {dropout!r}."
            raise ValueError(msg)

        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.output_dim = output_dim
        self.dropout = dropout
        self.use_residual = use_residual
        self.activation = activation

        layers: list[nn.Module] = []
        in_channels = input_dim
        for _ in range(num_layers):
            layers.append(
                ConvSiLUBlock(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    use_residual=use_residual,
                    activation=activation,
                )
            )
            in_channels = hidden_dim
        self.conv_stack = nn.ModuleList(layers)
        self.output_projection = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """inputs: (batch, seq_len, input_dim) → (batch, seq_len, output_dim)"""
        # Conv1d expects (batch, channels, seq_len).
        hidden = inputs.transpose(1, 2)
        for block in self.conv_stack:
            hidden = block(hidden)
        # Return to (batch, seq_len, hidden_dim) before the output projection.
        return self.output_projection(hidden.transpose(1, 2))

    def __repr__(self) -> str:
        return (
            f"CNN(input={self.input_dim}, hidden={self.hidden_dim}, "
            f"layers={self.num_layers}, kernel={self.kernel_size}, output={self.output_dim}, "
            f"activation={self.activation}, dropout={self.dropout}, residual={self.use_residual})"
        )
