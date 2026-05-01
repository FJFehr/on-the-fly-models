"""General-purpose RNN encoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.bidirectional = bidirectional
        self.bias = bias

        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            nonlinearity="relu",
            bias=bias,
        )
        rnn_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.output_projection = nn.Linear(rnn_out_dim, output_dim, bias=False)

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """inputs: (batch, seq_len, input_dim) → (batch, seq_len, output_dim)

        Pass lengths (1-D int tensor, one entry per batch item) to skip padding
        positions entirely via PackedSequence.  Without lengths the RNN processes
        the full padded sequence as before.
        """
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                inputs, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            rnn_out_packed, _ = self.rnn(packed)
            rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out_packed, batch_first=True)
            # pad_packed_sequence truncates to max real length in the batch; restore original
            if rnn_out.shape[1] < inputs.shape[1]:
                rnn_out = F.pad(rnn_out, (0, 0, 0, inputs.shape[1] - rnn_out.shape[1]))
        else:
            rnn_out, _ = self.rnn(inputs)
        return self.output_projection(rnn_out)

    def __repr__(self) -> str:
        return (
            f"RNN(input={self.input_dim}, hidden={self.hidden_dim}, "
            f"layers={self.num_layers}, output={self.output_dim}, "
            f"bidir={self.bidirectional}, bias={self.bias})"
        )
