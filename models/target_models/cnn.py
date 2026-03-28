# models/target_models/cnn.py
# ---------------------------------------------------------------------------
# Small 1D CNN target model.
#
# A translation-equivariant baseline for checking whether local convolutions
# (which naturally capture spatial shifts) can handle ARC1D move tasks
# without explicit positional features.
#
# Architecture: Conv1d -> ReLU -> Conv1d -> ReLU -> Conv1d(1x1)
#
# All training logic (loss, metrics, optimizer, WandB logging) is inherited
# from BaseTargetModel in models/target_models/base.py. This file only
# defines the CNN architecture and the forward pass with channel dimension
# handling.
# ---------------------------------------------------------------------------

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
    use_skip_connections: bool = False,
) -> torch.nn.Sequential:
    """Build a translation-equivariant 1D CNN with configurable depth.

    Architecture:
      - input projection: Conv1d(1->H, k) -> ReLU
      - hidden stack: num_layers x [Conv1d(H->H, k) -> ReLU]
      - output projection: Conv1d(H->1, 1)

    Uses same-padding (kernel_size // 2) to keep the sequence length unchanged
    through all layers. The final 1x1 convolution projects back to a single
    output channel.

    Args:
        hidden_channels: Number of feature channels in the hidden layers.
        kernel_size: Size of the convolutional kernels. Must be odd so that
            padding is symmetric and the output length matches the input.
        num_layers: Number of hidden H->H convolution layers. Must be at least 1.
            `num_layers=1` preserves the current CNN architecture.
        input_channels: Number of input channels for the first convolution.
        use_skip_connections: Whether to wrap hidden H->H conv blocks in
            residual connections.

    Returns:
        A torch.nn.Sequential module containing the CNN layers.

    Raises:
        ValueError: If kernel_size is even (would cause asymmetric padding).
    """
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
            layers.append(
                ResidualConvBlock(
                    hidden_channels,
                    kernel_size,
                    padding,
                )
            )
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
    layers.append(torch.nn.Conv1d(hidden_channels, 1, kernel_size=1))
    return torch.nn.Sequential(*layers)


class TargetCNNModelLightning(BaseTargetModel):
    """1D CNN target model for ARC1D binary sequence transformations.

    Inherits all training, logging, and optimizer logic from BaseTargetModel.
    Only defines the CNN architecture and a forward pass that handles the
    channel dimension required by Conv1d.

    The CNN expects input_dim == output_dim because the convolutions use
    same-padding to preserve sequence length.

    Args:
        input_dim: Sequence length (e.g. 33 for padded ARC sequences).
        output_dim: Must equal input_dim (same-padding preserves length).
        hidden_channels: Number of feature channels in hidden conv layers.
        kernel_size: Size of the convolutional kernels (must be odd).
        num_layers: Number of hidden H->H convolution layers.
        **kwargs: Passed through to BaseTargetModel (learning_rate, optimizer,
            weight_decay, logging config, and any extra config keys).

    Raises:
        ValueError: If input_dim != output_dim.
    """

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

        self.model = cnn_1d(
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            num_layers=num_layers,
            use_skip_connections=use_skip_connections,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass: add channel dim, run through CNN, remove channel dim.

        Conv1d expects input shape (batch, channels, length), but our data
        has shape (batch, length). We unsqueeze a channel dimension before
        the CNN and squeeze it back out afterward.

        Args:
            inputs: Input tensor of shape (batch_size, sequence_length).

        Returns:
            Logits tensor of shape (batch_size, sequence_length).
        """
        # (batch, length) -> (batch, 1, length) for Conv1d
        logits = self.model(inputs.unsqueeze(1))
        # (batch, 1, length) -> (batch, length)
        return logits.squeeze(1)
