import torch

from models.target_models.base import BaseTargetModel
from models.target_models.mlp import mlp


class TargetPositionalMLPLightning(BaseTargetModel):
    """MLP target model with concatenated normalized x-position features.

    The input sequence is flattened exactly like the plain MLP baseline, but we
    concatenate a fixed normalized position ramp in [0, 1] before the first
    linear layer. This gives the model absolute location information while
    preserving the existing datamodule and batch contract.

    Args:
        input_dim: Size of the input vector (e.g. 33 for padded ARC sequences).
        hidden_dim: Number of hidden units in the MLP hidden layer.
        output_dim: Size of the output vector (same as input for ARC tasks).
        num_layers: Number of hidden Linear layers before the output layer.
        **kwargs: Passed through to BaseTargetModel.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 1,
        use_skip_connections: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["kwargs"])

        if input_dim < 1:
            msg = "input_dim must be at least 1."
            raise ValueError(msg)

        self.output_dim = output_dim
        final_output_dim = (
            output_dim * self.num_classes if self.prediction_task == "multiclass" else output_dim
        )
        self.register_buffer(
            "x_positions",
            torch.linspace(0.0, 1.0, steps=input_dim, dtype=torch.float32),
        )
        self.model = mlp(
            input_dim * 2,
            hidden_dim,
            final_output_dim,
            num_layers=num_layers,
            use_skip_connections=use_skip_connections,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Concatenate positions, run the MLP, and return canonical logits."""
        x_positions = self.x_positions.unsqueeze(0).expand(inputs.shape[0], -1)
        features = torch.cat([inputs, x_positions], dim=1)
        logits = self.model(features)
        if self.prediction_task == "binary":
            return logits
        return logits.view(inputs.shape[0], self.output_dim, self.num_classes)
