import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from models.target_models.base import BaseTargetModel


class LayerNorm(nn.Module):
    """LayerNorm with an optional bias parameter."""

    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(inputs, self.weight.shape, self.weight, self.bias, 1e-5)


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            msg = "transformer_hidden_dim must be divisible by num_heads."
            raise ValueError(msg)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.causal = config.causal
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if self.causal and not self.flash:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels = x.size()
        query, key, value = self.c_attn(x).split(self.n_embd, dim=2)

        head_dim = channels // self.n_head
        key = key.view(batch_size, seq_len, self.n_head, head_dim).transpose(1, 2)
        query = query.view(batch_size, seq_len, self.n_head, head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.n_head, head_dim).transpose(1, 2)

        if self.flash:
            attended = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.causal,
            )
        else:
            scores = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(key.size(-1)))
            if self.causal:
                scores = scores.masked_fill(
                    self.bias[:, :, :seq_len, :seq_len] == 0,
                    float("-inf"),
                )
            scores = F.softmax(scores, dim=-1)
            scores = self.attn_dropout(scores)
            attended = scores @ value

        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        return self.resid_dropout(self.c_proj(attended))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class TransformerConfig:
    block_size: int
    input_feature_dim: int
    logit_dim: int
    n_layer: int = 1
    n_head: int = 1
    n_embd: int = 32
    dropout: float = 0.0
    bias: bool = True
    causal: bool = False


class Transformer(nn.Module):
    """Small transformer over per-position ARC features."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.input_projection = nn.Linear(
            config.input_feature_dim,
            config.n_embd,
            bias=config.bias,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        self.output_head = nn.Linear(config.n_embd, config.logit_dim, bias=False)

        self.apply(self._init_weights)
        for parameter_name, parameter in self.named_parameters():
            if parameter_name.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    parameter,
                    mean=0.0,
                    std=0.02 / math.sqrt(2 * config.n_layer),
                )

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = features.size()
        if seq_len > self.config.block_size:
            msg = (
                f"Cannot forward sequence of length {seq_len}, block size is only "
                f"{self.config.block_size}."
            )
            raise ValueError(msg)

        hidden_states = self.dropout(self.input_projection(features))
        for block in self.blocks:
            hidden_states = block(hidden_states)
        hidden_states = self.ln_f(hidden_states)
        return self.output_head(hidden_states)


class TargetTransformerModelLightning(BaseTargetModel):
    """Transformer target model with positional features enabled by default."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        transformer_hidden_dim: int,
        num_heads: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        bias: bool = True,
        causal_attention: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["kwargs"])

        if input_dim != output_dim:
            msg = "The transformer baseline expects input_dim and output_dim to match."
            raise ValueError(msg)
        if transformer_hidden_dim < 1:
            msg = "transformer_hidden_dim must be at least 1."
            raise ValueError(msg)
        if num_heads < 1:
            msg = "num_heads must be at least 1."
            raise ValueError(msg)
        if num_layers < 1:
            msg = "num_layers must be at least 1."
            raise ValueError(msg)

        self.embedding = (
            torch.nn.Embedding(self.num_classes, self.num_classes)
            if self.prediction_task == "multiclass"
            else None
        )
        self.register_buffer(
            "x_positions",
            torch.linspace(0.0, 1.0, steps=input_dim, dtype=torch.float32).view(1, -1, 1),
        )

        value_feature_dim = self.num_classes if self.prediction_task == "multiclass" else 1
        self.model = Transformer(
            TransformerConfig(
                block_size=input_dim,
                input_feature_dim=value_feature_dim + 1,
                logit_dim=1 if self.prediction_task == "binary" else self.num_classes,
                n_layer=num_layers,
                n_head=num_heads,
                n_embd=transformer_hidden_dim,
                dropout=dropout,
                bias=bias,
                causal=causal_attention,
            )
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Concatenate values and positions, then return canonical logits."""
        if self.embedding is not None:
            value_features = self.embedding(inputs.long())
        else:
            value_features = inputs.unsqueeze(-1)
        x_position_channel = self.x_positions.expand(inputs.shape[0], -1, -1)
        logits = self.model(torch.cat([value_features, x_position_channel], dim=-1))
        if self.prediction_task == "binary":
            return logits.squeeze(-1)
        return logits


__all__ = ["TargetTransformerModelLightning", "Transformer", "TransformerConfig"]
