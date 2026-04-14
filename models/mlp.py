"""Global MLP backbone for direct supervised experiments."""

import torch.nn as nn


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
    ):
        super().__init__()
        self.seq_len = seq_len
        self.output_dim = output_dim

        in_features = seq_len * input_dim
        out_features = seq_len * output_dim

        layers: list[nn.Module] = [nn.Linear(in_features, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, out_features))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass.

        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            (batch, seq_len, output_dim)
        """
        B = x.shape[0]
        return self.net(x.reshape(B, -1)).reshape(B, self.seq_len, self.output_dim)
