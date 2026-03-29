import torch

from models.target_models.base import BaseTargetModel


def build_activation(name: str) -> torch.nn.Module:
    """Build the configured activation module."""
    if name == "relu":
        return torch.nn.ReLU()
    if name == "gelu":
        return torch.nn.GELU()
    msg = f"Unknown activation {name!r}."
    raise ValueError(msg)


def apply_init_scheme(model: torch.nn.Module, init_scheme: str) -> None:
    """Apply a non-default initialization scheme to all Linear layers."""
    if init_scheme == "default":
        return
    if init_scheme not in {"xavier_uniform", "kaiming_uniform"}:
        msg = f"Unknown init_scheme {init_scheme!r}."
        raise ValueError(msg)

    for module in model.modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if init_scheme == "xavier_uniform":
            torch.nn.init.xavier_uniform_(module.weight)
        else:
            torch.nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)


class HiddenLinearBlock(torch.nn.Module):
    """Hidden MLP block with optional normalization and configurable activation."""

    def __init__(
        self,
        hidden_dim: int,
        activation: str = "relu",
        use_layernorm: bool = False,
        dropout: float = 0.0,
        ffn_expansion_factor: int = 1,
    ):
        super().__init__()
        if ffn_expansion_factor < 1:
            msg = "ffn_expansion_factor must be at least 1."
            raise ValueError(msg)
        if not 0.0 <= dropout < 1.0:
            msg = "dropout must be in [0.0, 1.0)."
            raise ValueError(msg)

        expanded_dim = hidden_dim * ffn_expansion_factor
        layers: list[torch.nn.Module] = [torch.nn.Linear(hidden_dim, expanded_dim)]
        if ffn_expansion_factor == 1:
            if use_layernorm:
                layers.append(torch.nn.LayerNorm(hidden_dim))
            layers.append(build_activation(activation))
            if dropout > 0.0:
                layers.append(torch.nn.Dropout(dropout))
        else:
            layers.append(build_activation(activation))
            layers.append(torch.nn.Linear(expanded_dim, hidden_dim))
            if use_layernorm:
                layers.append(torch.nn.LayerNorm(hidden_dim))
            if dropout > 0.0:
                layers.append(torch.nn.Dropout(dropout))
        self.block = torch.nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class ResidualLinearBlock(torch.nn.Module):
    """Hidden MLP block with an optional residual connection."""

    def __init__(
        self,
        hidden_dim: int,
        activation: str = "relu",
        use_layernorm: bool = False,
        dropout: float = 0.0,
        ffn_expansion_factor: int = 1,
    ):
        super().__init__()
        self.block = HiddenLinearBlock(
            hidden_dim,
            activation=activation,
            use_layernorm=use_layernorm,
            dropout=dropout,
            ffn_expansion_factor=ffn_expansion_factor,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.block(inputs)


def mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int = 1,
    use_skip_connections: bool = False,
    activation: str = "relu",
    use_layernorm: bool = False,
    init_scheme: str = "default",
    dropout: float = 0.0,
    ffn_expansion_factor: int = 1,
) -> torch.nn.Sequential:
    """Build a positional MLP with a configurable number of hidden layers."""
    if num_layers < 1:
        msg = "num_layers must be at least 1."
        raise ValueError(msg)
    if ffn_expansion_factor < 1:
        msg = "ffn_expansion_factor must be at least 1."
        raise ValueError(msg)
    if not 0.0 <= dropout < 1.0:
        msg = "dropout must be in [0.0, 1.0)."
        raise ValueError(msg)

    layers: list[torch.nn.Module] = [torch.nn.Linear(input_dim, hidden_dim)]
    if use_layernorm:
        layers.append(torch.nn.LayerNorm(hidden_dim))
    layers.append(build_activation(activation))
    if dropout > 0.0:
        layers.append(torch.nn.Dropout(dropout))
    for _ in range(num_layers - 1):
        if use_skip_connections:
            layers.append(
                ResidualLinearBlock(
                    hidden_dim,
                    activation=activation,
                    use_layernorm=use_layernorm,
                    dropout=dropout,
                    ffn_expansion_factor=ffn_expansion_factor,
                )
            )
        else:
            layers.append(
                HiddenLinearBlock(
                    hidden_dim,
                    activation=activation,
                    use_layernorm=use_layernorm,
                    dropout=dropout,
                    ffn_expansion_factor=ffn_expansion_factor,
                )
            )
    layers.append(torch.nn.Linear(hidden_dim, output_dim))
    model = torch.nn.Sequential(*layers)
    apply_init_scheme(model, init_scheme)
    return model


class TargetMLPModelLightning(BaseTargetModel):
    """MLP target model with positional features enabled by default."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 1,
        use_skip_connections: bool = False,
        activation: str = "relu",
        use_layernorm: bool = False,
        init_scheme: str = "default",
        dropout: float = 0.0,
        ffn_expansion_factor: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["kwargs"])

        if input_dim < 1:
            msg = "input_dim must be at least 1."
            raise ValueError(msg)

        self.output_dim = output_dim
        self.embedding = (
            torch.nn.Embedding(self.num_classes, self.num_classes)
            if self.prediction_task == "multiclass"
            else None
        )
        final_output_dim = (
            output_dim * self.num_classes if self.prediction_task == "multiclass" else output_dim
        )
        self.register_buffer(
            "x_positions",
            torch.linspace(0.0, 1.0, steps=input_dim, dtype=torch.float32),
        )
        mlp_input_dim = (
            input_dim * self.num_classes + input_dim
            if self.prediction_task == "multiclass"
            else input_dim * 2
        )
        self.model = mlp(
            mlp_input_dim,
            hidden_dim,
            final_output_dim,
            num_layers=num_layers,
            use_skip_connections=use_skip_connections,
            activation=activation,
            use_layernorm=use_layernorm,
            init_scheme=init_scheme,
            dropout=dropout,
            ffn_expansion_factor=ffn_expansion_factor,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Concatenate values and positions, then return canonical logits."""
        if self.embedding is not None:
            value_features = self.embedding(inputs.long()).view(inputs.shape[0], -1)
        else:
            value_features = inputs
        x_positions = self.x_positions.unsqueeze(0).expand(inputs.shape[0], -1)
        logits = self.model(torch.cat([value_features, x_positions], dim=1))
        if self.prediction_task == "binary":
            return logits
        return logits.view(inputs.shape[0], self.output_dim, self.num_classes)


__all__ = ["TargetMLPModelLightning", "apply_init_scheme", "build_activation", "mlp"]
