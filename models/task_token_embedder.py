"""Shared embeddings for task-serialized ARC tokens."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskTokenEmbedder(nn.Module):
    """Embed binary ARC task tokens by summing learned metadata embeddings."""

    def __init__(
        self,
        embedding_dim: int,
        position_vocab_size: int,
        value_vocab_size: int = 2,
        num_examples: int = 4,
        num_roles: int = 2,
        num_query_flags: int = 2,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.position_vocab_size = position_vocab_size
        self.value_vocab_size = value_vocab_size
        self.num_examples = num_examples
        self.num_roles = num_roles
        self.num_query_flags = num_query_flags

        self.value_embedding = nn.Embedding(value_vocab_size, embedding_dim)

        # Fixed sinusoidal position encoding — no parameters to train.
        # Encodes positional distance geometrically, making shift-amount detection
        # directly learnable without needing the model to discover vector arithmetic.
        pe = torch.zeros(position_vocab_size, embedding_dim)
        position = torch.arange(position_vocab_size).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float() * -(math.log(10000.0) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: embedding_dim // 2])
        self.register_buffer("sinusoidal_position_encoding", pe)  # (position_vocab_size, embedding_dim)

        self.example_embedding = nn.Embedding(num_examples, embedding_dim)
        self.role_embedding = nn.Embedding(num_roles, embedding_dim)
        self.is_query_embedding = nn.Embedding(num_query_flags, embedding_dim)

    def _validate_id_range(self, ids: torch.Tensor, upper_bound: int, name: str) -> None:
        if ids.numel() == 0:
            return
        min_id = int(ids.min().item())
        max_id = int(ids.max().item())
        if min_id < 0 or max_id >= upper_bound:
            msg = (
                f"{name} ids must stay in [0, {upper_bound - 1}], got "
                f"min={min_id}, max={max_id}."
            )
            raise ValueError(msg)

    def forward(
        self,
        value_ids: torch.Tensor,
        position_ids: torch.Tensor,
        example_ids: torch.Tensor,
        role_ids: torch.Tensor,
        is_query_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return summed embeddings with shape matching the id tensors plus embedding_dim."""
        tensors = {
            "value": value_ids,
            "position": position_ids,
            "example": example_ids,
            "role": role_ids,
            "is_query": is_query_ids,
        }
        shapes = {tuple(tensor.shape) for tensor in tensors.values()}
        if len(shapes) != 1:
            msg = f"All token id tensors must share one shape, got {shapes}."
            raise ValueError(msg)

        self._validate_id_range(value_ids, self.value_vocab_size, "value")
        self._validate_id_range(position_ids, self.position_vocab_size, "position")
        self._validate_id_range(example_ids, self.num_examples, "example")
        self._validate_id_range(role_ids, self.num_roles, "role")
        self._validate_id_range(is_query_ids, self.num_query_flags, "is_query")

        return (
            self.value_embedding(value_ids)
            + F.embedding(position_ids, self.sinusoidal_position_encoding)
            + self.example_embedding(example_ids)
            + self.role_embedding(role_ids)
            + self.is_query_embedding(is_query_ids)
        )
