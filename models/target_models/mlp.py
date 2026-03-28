# models/target_models/mlp.py
# ---------------------------------------------------------------------------
# Simple MLP target model.
#
# This is the "target model" in the on-the-fly-models framework: a small
# MLP that performs a 1D binary sequence transformation for a single ARC
# task. The hyper-model (not yet implemented) will eventually learn to
# predict this model's weights from support examples.
#
# Architecture: Linear -> ReLU -> Linear (two-layer MLP)
#
# All training logic (loss, metrics, optimizer, WandB logging) is inherited
# from BaseTargetModel in models/target_models/base.py. This file only
# defines the architecture and the forward pass.
# ---------------------------------------------------------------------------

import torch

from models.target_models.base import BaseTargetModel


class ResidualLinearBlock(torch.nn.Module):
    """Hidden MLP block with an optional residual connection."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.block(inputs)


def mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int = 1,
    use_skip_connections: bool = False,
) -> torch.nn.Sequential:
    """Build an MLP with a configurable number of hidden layers.

    Architecture:
      - num_layers=1: input_dim -> hidden_dim (ReLU) -> output_dim
      - num_layers>1: input_dim -> hidden_dim -> ... -> hidden_dim -> output_dim

    This is the simplest possible baseline: a fully-connected network
    that maps each input position to each output position. It has no
    notion of locality or translation equivariance.

    Args:
        input_dim: Size of the input vector (e.g. 33 for padded ARC sequences).
        hidden_dim: Number of hidden units in the middle layer.
        output_dim: Size of the output vector (same as input for ARC tasks).
        num_layers: Number of hidden Linear layers before the output layer.
            Must be at least 1. `num_layers=1` preserves the current MLP.
        use_skip_connections: Whether to wrap repeated hidden layers in
            residual connections. Has no effect when `num_layers=1`.

    Returns:
        A torch.nn.Sequential module containing the MLP layers.
    """
    if num_layers < 1:
        msg = "num_layers must be at least 1."
        raise ValueError(msg)

    layers: list[torch.nn.Module] = [
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.ReLU(),
    ]
    for _ in range(num_layers - 1):
        if use_skip_connections:
            layers.append(ResidualLinearBlock(hidden_dim))
        else:
            layers.extend(
                [
                    torch.nn.Linear(hidden_dim, hidden_dim),
                    torch.nn.ReLU(),
                ]
            )
    layers.append(torch.nn.Linear(hidden_dim, output_dim))
    return torch.nn.Sequential(*layers)


class TargetModelLightning(BaseTargetModel):
    """MLP target model for 1D-ARC binary sequence transformations.

    Inherits all training, logging, and optimizer logic from BaseTargetModel.
    Only defines the MLP architecture and a trivial forward pass.

    Args:
        input_dim: Size of the input vector (e.g. 33 for padded ARC sequences).
        hidden_dim: Number of hidden units in the MLP hidden layer.
        output_dim: Size of the output vector (same as input for ARC tasks).
        **kwargs: Passed through to BaseTargetModel (learning_rate, optimizer,
            weight_decay, logging config, and any extra config keys).
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

        # save_hyperparameters() stores all named arguments (excluding **kwargs)
        # into self.hparams, which Lightning uses for checkpoint saving/loading.
        # This captures only the architecture-specific params (input_dim,
        # hidden_dim, output_dim) — training params are handled by the base class.
        self.save_hyperparameters(ignore=["kwargs"])

        # Build the MLP architecture
        self.model = mlp(
            input_dim,
            hidden_dim,
            output_dim,
            num_layers=num_layers,
            use_skip_connections=use_skip_connections,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass: feed input sequences directly through the MLP.

        The MLP treats the entire sequence as a flat vector, so no reshaping
        is needed — the input tensor goes straight through the network.

        Args:
            inputs: Input tensor of shape (batch_size, sequence_length).

        Returns:
            Logits tensor of shape (batch_size, sequence_length).
        """
        return self.model(inputs)
