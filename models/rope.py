"""Rotary Position Embedding (RoPE).

Applies position-dependent rotation to query and key tensors inside attention,
replacing additive sinusoidal PE in the input embeddings.

Reference: RoFormer (Su et al., 2021) — https://arxiv.org/abs/2104.09864
"""

import torch
import torch.nn as nn


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of head_dim into the first and negate the first."""
    d = x.shape[-1] // 2
    return torch.cat([-x[..., d:], x[..., :d]], dim=-1)


class RoPE(nn.Module):
    """Precomputed rotary position embedding applied to (query, key) pairs.

    Args:
        head_dim: dimension of each attention head (must be even).
        max_seq_len: maximum sequence length to pre-cache (clipped at forward time).
        base: frequency base; 10000 matches the original Transformer.
    """

    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}.")

        # θ_i = base^{-2i/d} for i in 0..d/2-1
        theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, theta)  # [max_seq_len, head_dim/2]

        # Duplicate so the full head_dim is covered: [cos(mθ), cos(mθ)] for each pair
        cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)  # [max_seq_len, head_dim]
        sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)  # [max_seq_len, head_dim]
        self.register_buffer("cos_cache", cos)
        self.register_buffer("sin_cache", sin)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to query and key.

        Args:
            q: [B, num_heads, T, head_dim]
            k: [B, num_heads, T, head_dim]

        Returns:
            (q_rotated, k_rotated) with same shapes.
        """
        T = q.size(2)
        cos = self.cos_cache[:T].unsqueeze(0).unsqueeze(0)  # [1, 1, T, head_dim]
        sin = self.sin_cache[:T].unsqueeze(0).unsqueeze(0)  # [1, 1, T, head_dim]
        return (
            q * cos + _rotate_half(q) * sin,
            k * cos + _rotate_half(k) * sin,
        )
