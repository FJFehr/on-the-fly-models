import torch
import torch.nn as nn

from models.target_models.base import BaseTargetModel


def _build_encoder(hidden_dim: int, num_layers: int) -> nn.Sequential:
    """Per-element encoder: (value, position) → hidden_dim.

    num_layers controls depth the same way MLP's num_layers does:
      num_layers=1  →  Linear(2, hidden) → ReLU → Linear(hidden, hidden)
      num_layers=2  →  ... → ReLU → Linear(hidden, hidden)
    """
    if num_layers < 1:
        msg = "num_layers must be at least 1."
        raise ValueError(msg)
    layers: list[nn.Module] = [nn.Linear(2, hidden_dim), nn.ReLU()]
    for _ in range(num_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    layers.append(nn.Linear(hidden_dim, hidden_dim))
    return nn.Sequential(*layers)


class TargetDeepSetModelLightning(BaseTargetModel):
    """DeepSet target model for binary sequence prediction.

    Treats each input position as a set element (value, position ramp).
    Applies a shared per-element encoder, mean-pools, then reads out with a
    single linear decoder to per-position logits.

    Parameter count matches TargetMLPModelLightning at the same hidden_dim and
    num_layers, making the two directly comparable.

    Architecture:
        (batch, seq_len) → (batch, seq_len, 2) [value + position ramp]
        → encoder MLP per element → (batch, seq_len, hidden_dim)
        → mean pool → (batch, hidden_dim)
        → Linear → (batch, seq_len) logits
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["kwargs"])

        self.output_dim = output_dim
        self.encoder = _build_encoder(hidden_dim, num_layers)
        self.decoder = nn.Linear(hidden_dim, output_dim)
        self.register_buffer(
            "x_positions",
            torch.linspace(0.0, 1.0, steps=input_dim, dtype=torch.float32),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode each position as a set element, pool, decode to logits.

        Args:
            inputs: (batch, seq_len) float tensor of binary values.

        Returns:
            Logits of shape (batch, seq_len) for BCEWithLogitsLoss.
        """
        batch_size = inputs.shape[0]
        values = inputs.unsqueeze(-1).float()
        positions = self.x_positions.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1)
        # (batch, seq_len, 2)
        elements = torch.cat([values, positions], dim=-1)
        # (batch, seq_len, hidden_dim) — same encoder weights at every position
        encoded = self.encoder(elements)
        # (batch, hidden_dim) — permutation-invariant aggregation
        pooled = encoded.mean(dim=1)
        # (batch, seq_len)
        return self.decoder(pooled)


__all__ = ["TargetDeepSetModelLightning"]
