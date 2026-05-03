"""Global MLP backbone for direct supervised experiments."""

import torch.nn as nn

from models.activations import build_activation


class HiddenSiLUBlock(nn.Module):
    """Linear block with SiLU update, dropout, and optional residual."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float,
        use_residual: bool,
        activation: str,
    ):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
        self.activation = build_activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.use_residual = use_residual
        if use_residual and input_dim != output_dim:
            self.residual_projection = nn.Linear(input_dim, output_dim, bias=False)
        else:
            self.residual_projection = nn.Identity()

    def forward(self, inputs):
        update = self.activation(self.projection(inputs))
        update = self.dropout(update)
        if not self.use_residual:
            return update
        return self.residual_projection(inputs) + update


class MLP(nn.Module):
    """Global MLP: flatten full sequence → FC layers → reshape back.

    Treats the entire input sequence as a single flat vector, applies a stack
    of fully-connected layers, then reshapes the output back to sequence form.
    This allows the model to capture arbitrary cross-position interactions but
    does not share weights across positions.

    Args:
        input_dim: Feature dimension of each input token.
        hidden_dim: Width of the hidden layers.
        num_layers: Total number of FC layers (must be >= 1).
        output_dim: Feature dimension of each output token.
        seq_len: Length of the input/output sequence.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        seq_len: int,
        dropout: float = 0.0,
        use_residual: bool = False,
        activation: str = "silu",
    ):
        if num_layers < 1:
            msg = "num_layers must be >= 1."
            raise ValueError(msg)
        if not 0.0 <= dropout < 1.0:
            msg = f"dropout must be in [0.0, 1.0), got {dropout!r}."
            raise ValueError(msg)

        super().__init__()
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_residual = use_residual
        self.activation = activation

        in_features = seq_len * input_dim
        out_features = seq_len * output_dim

        self.input_block = HiddenSiLUBlock(
            input_dim=in_features,
            output_dim=hidden_dim,
            dropout=dropout,
            use_residual=use_residual,
            activation=activation,
        )
        self.hidden_blocks = nn.ModuleList(
            [
                HiddenSiLUBlock(
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    dropout=dropout,
                    use_residual=use_residual,
                    activation=activation,
                )
                for _ in range(num_layers - 1)
            ]
        )
        self.output_projection = nn.Linear(hidden_dim, out_features)

    def forward(self, x):
        """Forward pass.

        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            (batch, seq_len, output_dim)
        """
        batch_size = x.shape[0]
        hidden = self.input_block(x.reshape(batch_size, -1))
        for block in self.hidden_blocks:
            hidden = block(hidden)
        return self.output_projection(hidden).reshape(batch_size, self.seq_len, self.output_dim)

    def __repr__(self) -> str:
        return (
            f"MLP(hidden={self.hidden_dim}, layers={self.num_layers}, seq_len={self.seq_len}, "
            f"output={self.output_dim}, activation={self.activation}, "
            f"dropout={self.dropout}, residual={self.use_residual})"
        )
