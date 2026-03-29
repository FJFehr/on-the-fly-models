"""Shared pure CNN modules used by baseline and hypernetwork experiments."""

import torch


class ResidualConvBlock(torch.nn.Module):
    """Hidden CNN block with a residual connection over one Conv1d layer."""

    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        padding: int,
    ):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv1d(
                hidden_channels,
                hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            torch.nn.ReLU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.block(inputs)


def cnn_1d(
    hidden_channels: int,
    kernel_size: int = 3,
    num_layers: int = 1,
    input_channels: int = 1,
    output_channels: int = 1,
    use_skip_connections: bool = False,
) -> torch.nn.Sequential:
    """Build a 1D CNN with optional residual hidden blocks."""
    if kernel_size % 2 == 0:
        msg = "kernel_size must be odd so the sequence length stays unchanged."
        raise ValueError(msg)
    if num_layers < 1:
        msg = "num_layers must be at least 1."
        raise ValueError(msg)

    padding = kernel_size // 2
    layers: list[torch.nn.Module] = [
        torch.nn.Conv1d(
            input_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        ),
        torch.nn.ReLU(),
    ]
    for _ in range(num_layers):
        if use_skip_connections:
            layers.append(ResidualConvBlock(hidden_channels, kernel_size, padding))
        else:
            layers.extend(
                [
                    torch.nn.Conv1d(
                        hidden_channels,
                        hidden_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                    ),
                    torch.nn.ReLU(),
                ]
            )
    layers.append(torch.nn.Conv1d(hidden_channels, output_channels, kernel_size=1))
    return torch.nn.Sequential(*layers)


class TargetCNNCore(torch.nn.Module):
    """Pure CNN sequence model with optional positional features."""

    def __init__(
        self,
        num_classes: int,
        hidden_channels: int,
        kernel_size: int = 3,
        num_layers: int = 1,
        sequence_length: int | None = None,
        use_skip_connections: bool = False,
        use_positional_feature: bool = True,
    ):
        super().__init__()
        if hidden_channels < 1:
            msg = "hidden_channels must be at least 1."
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
                    1, 1, -1
                ),
            )
        else:
            self.x_positions = None

        value_channels = num_classes
        input_channels = value_channels + (1 if use_positional_feature else 0)
        self.model = cnn_1d(
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            num_layers=num_layers,
            input_channels=input_channels,
            output_channels=num_classes,
            use_skip_connections=use_skip_connections,
        )

    def build_features(self, inputs: torch.Tensor) -> torch.Tensor:
        value_channels = self.embedding(inputs.long()).transpose(1, 2)
        if not self.use_positional_feature:
            return value_channels
        x_position_channel = self.x_positions.expand(inputs.shape[0], -1, -1)
        return torch.cat([value_channels, x_position_channel], dim=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.model(self.build_features(inputs))
        return logits.transpose(1, 2)


__all__ = ["ResidualConvBlock", "TargetCNNCore", "cnn_1d"]
