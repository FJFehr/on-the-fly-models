import torch

from models.target_models.base import BaseTargetModel


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
    """Build a 1D CNN with positional features enabled by default."""
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


class TargetCNNModelLightning(BaseTargetModel):
    """1D CNN target model with positional features enabled by default."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_channels: int,
        kernel_size: int = 3,
        num_layers: int = 1,
        use_skip_connections: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["kwargs"])

        if input_dim != output_dim:
            msg = "The CNN baseline expects input_dim and output_dim to match."
            raise ValueError(msg)

        self.embedding = (
            torch.nn.Embedding(self.num_classes, self.num_classes)
            if self.prediction_task == "multiclass"
            else None
        )
        value_channels = self.num_classes if self.prediction_task == "multiclass" else 1
        self.register_buffer(
            "x_positions",
            torch.linspace(0.0, 1.0, steps=input_dim, dtype=torch.float32).view(1, 1, -1),
        )
        self.model = cnn_1d(
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            num_layers=num_layers,
            input_channels=value_channels + 1,
            output_channels=1 if self.prediction_task == "binary" else self.num_classes,
            use_skip_connections=use_skip_connections,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Append positions and return logits using the canonical task shape."""
        if self.embedding is not None:
            value_channels = self.embedding(inputs.long()).transpose(1, 2)
        else:
            value_channels = inputs.unsqueeze(1)
        x_position_channel = self.x_positions.expand(inputs.shape[0], -1, -1)
        logits = self.model(torch.cat([value_channels, x_position_channel], dim=1))
        if self.prediction_task == "binary":
            return logits.squeeze(1)
        return logits.transpose(1, 2)


__all__ = ["TargetCNNModelLightning", "cnn_1d"]
