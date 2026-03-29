import torch

from models.target_models.base import BaseTargetModel
from models.target_models.cnn_core import TargetCNNCore, cnn_1d


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
        if self.prediction_task == "multiclass":
            self.model = TargetCNNCore(
                num_classes=self.num_classes,
                hidden_channels=hidden_channels,
                kernel_size=kernel_size,
                num_layers=num_layers,
                sequence_length=input_dim,
                use_skip_connections=use_skip_connections,
                use_positional_feature=True,
            )
            self.binary_model = None
        else:
            self.model = None
            self.register_buffer(
                "x_positions",
                torch.linspace(0.0, 1.0, steps=input_dim, dtype=torch.float32).view(1, 1, -1),
            )
            self.binary_model = cnn_1d(
                hidden_channels=hidden_channels,
                kernel_size=kernel_size,
                num_layers=num_layers,
                input_channels=2,
                output_channels=1,
                use_skip_connections=use_skip_connections,
            )

    def forward(self, inputs):
        """Return multiclass logits with positional features."""
        if self.model is not None:
            return self.model(inputs)

        value_channels = inputs.unsqueeze(1)
        x_position_channel = self.x_positions.expand(inputs.shape[0], -1, -1)
        logits = self.binary_model(torch.cat([value_channels, x_position_channel], dim=1))
        return logits.squeeze(1)


__all__ = ["TargetCNNModelLightning", "cnn_1d"]
