"""Shared pure RNN modules used by baseline and hypernetwork experiments."""

import torch


class TargetRNNCore(torch.nn.Module):
    """Pure RNN sequence model with optional positional features."""

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int,
        num_layers: int = 1,
        bidirectional: bool = True,
        sequence_length: int | None = None,
        use_positional_feature: bool = True,
        use_skip_connections: bool = False,
    ):
        super().__init__()
        if hidden_dim < 1:
            msg = "hidden_dim must be at least 1."
            raise ValueError(msg)
        if num_layers < 1:
            msg = "num_layers must be at least 1."
            raise ValueError(msg)
        if use_positional_feature and sequence_length is None:
            msg = "sequence_length is required when use_positional_feature is enabled."
            raise ValueError(msg)

        self.num_classes = num_classes
        self.use_positional_feature = use_positional_feature
        self.embedding = torch.nn.Embedding(num_classes, num_classes)

        if use_positional_feature:
            self.register_buffer(
                "x_positions",
                torch.linspace(0.0, 1.0, steps=sequence_length, dtype=torch.float32).view(
                    1, -1, 1
                ),
            )
        else:
            self.x_positions = None

        rnn_input_size = num_classes + (1 if use_positional_feature else 0)
        self.rnn = torch.nn.RNN(
            input_size=rnn_input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            nonlinearity="relu",
            batch_first=True,
            bidirectional=bidirectional,
        )
        output_features = hidden_dim * (2 if bidirectional else 1)
        self.input_skip = (
            torch.nn.Linear(rnn_input_size, output_features) if use_skip_connections else None
        )
        self.output_layer = torch.nn.Linear(output_features, num_classes)

    def build_features(self, inputs: torch.Tensor) -> torch.Tensor:
        value_features = self.embedding(inputs.long())
        if not self.use_positional_feature:
            return value_features
        x_position_channel = self.x_positions.expand(inputs.shape[0], -1, -1)
        return torch.cat([value_features, x_position_channel], dim=-1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.build_features(inputs)
        hidden_states, _ = self.rnn(features)
        if self.input_skip is not None:
            hidden_states = hidden_states + self.input_skip(features)
        return self.output_layer(hidden_states)


__all__ = ["TargetRNNCore"]
