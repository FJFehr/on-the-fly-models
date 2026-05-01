"""Shared embeddings for task-serialized ARC tokens."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_sinusoidal_pe(position_vocab_size: int, embedding_dim: int) -> torch.Tensor:
    """Build a fixed sinusoidal encoding table for one position vocabulary."""
    pe = torch.zeros(position_vocab_size, embedding_dim)
    position = torch.arange(position_vocab_size).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, embedding_dim, 2).float() * -(math.log(10000.0) / embedding_dim)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: embedding_dim // 2])
    return pe


class TaskTokenEmbedder(nn.Module):
    """Embed binary ARC task tokens by summing learned metadata embeddings."""

    def __init__(
        self,
        embedding_dim: int,
        position_vocab_size: int = 0,  # deprecated; PE is now computed dynamically
        value_vocab_size: int = 2,
        num_examples: int = 4,
        num_roles: int = 2,
        padding_idx: int | None = None,
    ):
        del position_vocab_size  # deprecated; PE is now computed dynamically in forward()
        super().__init__()
        self.embedding_dim = embedding_dim
        self.value_vocab_size = value_vocab_size
        self.num_examples = num_examples
        self.num_roles = num_roles

        self.value_embedding = nn.Embedding(
            value_vocab_size, embedding_dim, padding_idx=padding_idx
        )
        self.example_embedding = nn.Embedding(num_examples, embedding_dim)
        self.role_embedding = nn.Embedding(num_roles, embedding_dim)

        # Match the transformer's N(0, 0.02) weight init so embedding scale is
        # consistent with the downstream network (default nn.Embedding uses N(0, 1)).
        for emb in (self.value_embedding, self.example_embedding, self.role_embedding):
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

    def _validate_id_range(self, ids: torch.Tensor, upper_bound: int, name: str) -> None:
        if ids.numel() == 0:
            return
        min_id = int(ids.min().item())
        max_id = int(ids.max().item())
        if min_id < 0 or max_id >= upper_bound:
            msg = (
                f"{name} ids must stay in [0, {upper_bound - 1}], got min={min_id}, max={max_id}."
            )
            raise ValueError(msg)

    def forward(
        self,
        value_ids: torch.Tensor,
        position_ids: torch.Tensor,
        example_ids: torch.Tensor | None = None,
        role_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return summed embeddings with shape matching the id tensors plus embedding_dim.

        Pass example_ids and role_ids for the hypernetwork path (support segments only).
        Omit both for the target path (value + position only).
        """
        reference_shape = tuple(value_ids.shape)
        optional = {"example": example_ids, "role": role_ids}
        for name, tensor in optional.items():
            if tensor is not None and tuple(tensor.shape) != reference_shape:
                msg = (
                    f"{name}_ids shape {tuple(tensor.shape)} does not match "
                    f"value_ids shape {reference_shape}."
                )
                raise ValueError(msg)
        if tuple(position_ids.shape) != reference_shape:
            msg = (
                f"position_ids shape {tuple(position_ids.shape)} does not match "
                f"value_ids shape {reference_shape}."
            )
            raise ValueError(msg)

        self._validate_id_range(value_ids, self.value_vocab_size, "value")

        max_pos = int(position_ids.max().item()) + 1
        pe_table = _build_sinusoidal_pe(max_pos, self.embedding_dim).to(value_ids.device)
        result = self.value_embedding(value_ids) + F.embedding(position_ids, pe_table)
        if example_ids is not None:
            self._validate_id_range(example_ids, self.num_examples, "example")
            result = result + self.example_embedding(example_ids)
        if role_ids is not None:
            self._validate_id_range(role_ids, self.num_roles, "role")
            result = result + self.role_embedding(role_ids)
        return result
