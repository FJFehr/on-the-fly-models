import torch

from models.target_models.base import BaseTargetModel
from models.target_models.rnn_core import TargetRNNCore


class TargetRNNModelLightning(BaseTargetModel):
    """RNN target model with positional features enabled by default."""

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
            msg = "The RNN baseline expects input_dim and output_dim to match."
            raise ValueError(msg)
        if rnn_hidden_dim < 1:
            msg = "rnn_hidden_dim must be at least 1."
            raise ValueError(msg)
        if rnn_num_layers < 1:
            msg = "rnn_num_layers must be at least 1."
            raise ValueError(msg)

        if self.prediction_task == "multiclass":
            self.model = TargetRNNCore(
                num_classes=self.num_classes,
                hidden_dim=rnn_hidden_dim,
                num_layers=rnn_num_layers,
                bidirectional=rnn_bidirectional,
                sequence_length=input_dim,
                use_positional_feature=True,
                use_skip_connections=use_skip_connections,
            )
            self.binary_rnn = None
            self.binary_input_skip = None
            self.binary_output_layer = None
        else:
            self.model = None
            self.register_buffer(
                "x_positions",
                torch.linspace(0.0, 1.0, steps=input_dim, dtype=torch.float32).view(1, -1, 1),
            )
            self.binary_rnn = torch.nn.RNN(
                input_size=2,
                hidden_size=rnn_hidden_dim,
                num_layers=rnn_num_layers,
                nonlinearity="relu",
                batch_first=True,
                bidirectional=rnn_bidirectional,
            )
            output_features = rnn_hidden_dim * (2 if rnn_bidirectional else 1)
            self.binary_input_skip = (
                torch.nn.Linear(2, output_features) if use_skip_connections else None
            )
            self.binary_output_layer = torch.nn.Linear(output_features, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the RNN baseline and return canonical logits."""
        if self.model is not None:
            return self.model(inputs)

        value_features = inputs.unsqueeze(-1)
        x_position_channel = self.x_positions.expand(inputs.shape[0], -1, -1)
        features = torch.cat([value_features, x_position_channel], dim=-1)
        hidden_states, _ = self.binary_rnn(features)
        if self.binary_input_skip is not None:
            hidden_states = hidden_states + self.binary_input_skip(features)
        return self.binary_output_layer(hidden_states).squeeze(-1)


__all__ = ["TargetRNNModelLightning"]
