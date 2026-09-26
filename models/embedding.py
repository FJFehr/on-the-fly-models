"""Token embeddings for ARC-1D sequences, and the fixed task-category index."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Fixed index of every ARC-1D task category, used for task-identity embeddings and
# one-hot task indicators. The order is part of saved checkpoints: do not reorder.
TASK_CATEGORY_INDEX: dict[str, int] = {
    "1d_move_1p": 0,
    "1d_move_2p": 1,
    "1d_move_3p": 2,
    "1d_move_dp": 3,
    "1d_move_2p_dp": 4,
    "1d_fill": 5,
    "1d_hollow": 6,
    "1d_flip": 7,
    "1d_mirror": 8,
    "1d_denoising_1c": 9,
    "1d_denoising_mc": 10,
    "1d_pcopy_1c": 11,
    "1d_pcopy_mc": 12,
    "1d_recolor_oe": 13,
    "1d_recolor_cnt": 14,
    "1d_recolor_cmp": 15,
    "1d_scale_dp": 16,
    "1d_padded_fill": 17,
}


def sinusoidal_position_table(num_positions: int, dim: int) -> torch.Tensor:
    """Fixed sine/cosine position encodings, shape (num_positions, dim)."""
    table = torch.zeros(num_positions, dim)
    position = torch.arange(num_positions).unsqueeze(1).float()
    frequency = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
    table[:, 0::2] = torch.sin(position * frequency)
    table[:, 1::2] = torch.cos(position * frequency[: dim // 2])
    return table


class TokenEmbedder(nn.Module):
    """Sum of learned embeddings for each token's value, example index and role.

    The target model and the direct model pass values only. The hypernetwork also passes
    which support example (0-2) a token comes from and its role (0 = input, 1 = output).
    Position normally comes from RoPE inside attention; `use_sinusoidal_pe` adds fixed
    sinusoidal position encodings here as well.
    """

    NUM_EXAMPLES = 4
    NUM_ROLES = 2

    def __init__(
        self,
        dim: int,
        vocab_size: int,
        padding_idx: int | None = None,
        use_sinusoidal_pe: bool = False,
    ):
        super().__init__()
        self.use_sinusoidal_pe = use_sinusoidal_pe
        self.value_embedding = nn.Embedding(vocab_size, dim, padding_idx=padding_idx)
        self.example_embedding = nn.Embedding(self.NUM_EXAMPLES, dim)
        self.role_embedding = nn.Embedding(self.NUM_ROLES, dim)
        for embedding in (self.value_embedding, self.example_embedding, self.role_embedding):
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)

    def forward(
        self,
        value_ids: torch.Tensor,
        position_ids: torch.Tensor,
        example_ids: torch.Tensor | None = None,
        role_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embed id tensors of shape (..., seq_len) into (..., seq_len, dim).

        `position_ids` is each token's position within its own sequence. It only matters
        with `use_sinusoidal_pe`, and restarts at 0 for each support segment in the
        hypernetwork's context.
        """
        embedded = self.value_embedding(value_ids)
        if self.use_sinusoidal_pe:
            num_positions = int(position_ids.max().item()) + 1
            table = sinusoidal_position_table(num_positions, embedded.shape[-1])
            embedded = embedded + F.embedding(position_ids, table.to(embedded.device))
        if example_ids is not None:
            embedded = embedded + self.example_embedding(example_ids)
        if role_ids is not None:
            embedded = embedded + self.role_embedding(role_ids)
        return embedded
