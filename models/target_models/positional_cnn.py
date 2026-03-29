import torch

from models.target_models.base import BaseTargetModel
from models.target_models.cnn import cnn_1d


class TargetPositionalCNNModelLightning(BaseTargetModel):
    """1D CNN target model with explicit normalized x-position input channel.

    The model preserves the plain CNN structure, but concatenates a normalized
    x-position ramp as a second input channel before the first convolution.

    Args:
        input_dim: Sequence length (e.g. 33 for padded ARC sequences).
        output_dim: Must equal input_dim.
        hidden_channels: Number of feature channels in hidden conv layers.
        kernel_size: Size of the convolutional kernels (must be odd).
        num_layers: Number of hidden H->H convolution layers.
        **kwargs: Passed through to BaseTargetModel.
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
            msg = "The positional CNN baseline expects input_dim and output_dim to match."
            raise ValueError(msg)

        self.register_buffer(
            "x_positions",
            torch.linspace(0.0, 1.0, steps=input_dim, dtype=torch.float32).view(1, 1, -1),
        )
        self.model = cnn_1d(
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            num_layers=num_layers,
            input_channels=2,
            output_channels=1 if self.prediction_task == "binary" else self.num_classes,
            use_skip_connections=use_skip_connections,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Append positions and return logits using the canonical task shape."""
        value_channel = inputs.unsqueeze(1)
        x_position_channel = self.x_positions.expand(inputs.shape[0], -1, -1)
        logits = self.model(torch.cat([value_channel, x_position_channel], dim=1))
        if self.prediction_task == "binary":
            return logits.squeeze(1)
        return logits.transpose(1, 2)
