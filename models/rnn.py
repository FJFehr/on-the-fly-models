"""General-purpose RNN encoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.activations import build_activation


class ResidualRNNLayer(nn.Module):
    """One recurrent layer with a SiLU update path and optional residual."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        bidirectional: bool,
        bias: bool,
        dropout: float,
        use_residual: bool,
        activation: str,
    ):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
            nonlinearity="tanh",
            bias=bias,
        )
        self.output_dim = hidden_dim * (2 if bidirectional else 1)
        self.update_projection = nn.Linear(self.output_dim, self.output_dim, bias=bias)
        self.activation = build_activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.use_residual = use_residual
        if use_residual and input_dim != self.output_dim:
            self.residual_projection = nn.Linear(input_dim, self.output_dim, bias=False)
        else:
            self.residual_projection = nn.Identity()

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        # cuDNN RNN does not support bf16; fall back to PyTorch's own matmul-based
        # implementation which handles bf16 fine. Disabled only for this call.
        prev_cudnn = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False
        try:
            if lengths is not None:
                packed = nn.utils.rnn.pack_padded_sequence(
                    inputs, lengths.cpu(), batch_first=True, enforce_sorted=False
                )
                rnn_out_packed, _ = self.rnn(packed)
                rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out_packed, batch_first=True)
                if rnn_out.shape[1] < inputs.shape[1]:
                    rnn_out = F.pad(rnn_out, (0, 0, 0, inputs.shape[1] - rnn_out.shape[1]))
            else:
                rnn_out, _ = self.rnn(inputs)
        finally:
            torch.backends.cudnn.enabled = prev_cudnn

        hidden = self.activation(self.update_projection(rnn_out))
        hidden = self.dropout(hidden)
        if self.use_residual:
            hidden = self.residual_projection(inputs) + hidden

        if lengths is not None:
            valid = torch.arange(inputs.shape[1], device=inputs.device).unsqueeze(
                0
            ) < lengths.unsqueeze(1)
            hidden = hidden.masked_fill(~valid.unsqueeze(-1), 0.0)
        return hidden


class RNN(nn.Module):
    """Bidirectional RNN encoder.

    Input:  (batch, seq_len, input_dim)
    Output: (batch, seq_len, output_dim)

    Works as a normal trained model or as a stateless target inside HyperModel —
    no changes to this class are needed for either mode. When used as a
    hypernetwork target, set output_dim = target_model.total_params and
    HyperModel will take the last-position output as the parameter vector.
    When used as a prediction target, set output_dim = 1 (binary).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        output_dim: int = 1,
        bidirectional: bool = True,
        bias: bool = True,
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
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.bidirectional = bidirectional
        self.bias = bias
        self.dropout = dropout
        self.use_residual = use_residual
        self.activation = activation

        layers: list[nn.Module] = []
        layer_input_dim = input_dim
        for _ in range(num_layers):
            layer = ResidualRNNLayer(
                input_dim=layer_input_dim,
                hidden_dim=hidden_dim,
                bidirectional=bidirectional,
                bias=bias,
                dropout=dropout,
                use_residual=use_residual,
                activation=activation,
            )
            layers.append(layer)
            layer_input_dim = layer.output_dim
        self.layers = nn.ModuleList(layers)

        rnn_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.output_projection = nn.Linear(rnn_out_dim, output_dim, bias=False)

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """inputs: (batch, seq_len, input_dim) → (batch, seq_len, output_dim)

        Pass lengths (1-D int tensor, one entry per batch item) to skip padding
        positions entirely via PackedSequence.  Without lengths the RNN processes
        the full padded sequence as before.
        """
        hidden = inputs
        for layer in self.layers:
            hidden = layer(hidden, lengths=lengths)
        return self.output_projection(hidden)

    def __repr__(self) -> str:
        return (
            f"RNN(input={self.input_dim}, hidden={self.hidden_dim}, "
            f"layers={self.num_layers}, output={self.output_dim}, "
            f"bidir={self.bidirectional}, bias={self.bias}, activation={self.activation}, "
            f"dropout={self.dropout}, residual={self.use_residual})"
        )
