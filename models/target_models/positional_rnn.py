import torch

from models.target_models.base import BaseTargetModel


class TargetPositionalRNNModelLightning(BaseTargetModel):
    """Plain bidirectional RNN with explicit normalized x-position features."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        rnn_hidden_dim: int,
        rnn_bidirectional: bool = True,
        rnn_num_layers: int = 1,
        use_skip_connections: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["kwargs"])

        if input_dim != output_dim:
            msg = "The positional RNN baseline expects input_dim and output_dim to match."
            raise ValueError(msg)
        if rnn_hidden_dim < 1:
            msg = "rnn_hidden_dim must be at least 1."
            raise ValueError(msg)
        if rnn_num_layers < 1:
            msg = "rnn_num_layers must be at least 1."
            raise ValueError(msg)

        self.register_buffer(
            "x_positions",
            torch.linspace(0.0, 1.0, steps=input_dim, dtype=torch.float32).view(1, -1, 1),
        )
        self.rnn = torch.nn.RNN(
            input_size=2,
            hidden_size=rnn_hidden_dim,
            num_layers=rnn_num_layers,
            nonlinearity="relu",
            batch_first=True,
            bidirectional=rnn_bidirectional,
        )
        output_features = rnn_hidden_dim * (2 if rnn_bidirectional else 1)
        self.input_skip = torch.nn.Linear(2, output_features) if use_skip_connections else None
        self.output_layer = torch.nn.Linear(
            output_features,
            1 if self.prediction_task == "binary" else self.num_classes,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Concatenate values and positions, then return canonical logits."""
        value_channel = inputs.unsqueeze(-1)
        x_position_channel = self.x_positions.expand(inputs.shape[0], -1, -1)
        features = torch.cat([value_channel, x_position_channel], dim=-1)
        hidden_states, _ = self.rnn(features)
        if self.input_skip is not None:
            hidden_states = hidden_states + self.input_skip(features)
        logits = self.output_layer(hidden_states)
        if self.prediction_task == "binary":
            return logits.squeeze(-1)
        return logits
