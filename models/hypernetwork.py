"""Hypernetwork: generates a target model's weights from a task's support examples.

For each task in the batch:

  ENCODER (support set -> task representation)
    1. `encoder`: a Transformer reads the serialised support examples,
    2. `pooler`: attention pooling turns its token outputs into one task representation,
    3. `task_indicator_proj` (optional): a one-hot task identity is projected and added.

  WEIGHT DECODER (task representation -> target weights)
    4. `projection`: an MLP maps the task representation to a flat vector holding every
       weight of the target model.

  TARGET (runs with the generated weights)
    5. `target`: a Transformer runs on the task's examples using those generated weights.

The target model's own parameters are never trained: they only provide the parameter
names and shapes that the generated vector is cut into.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
from torch.nn.attention import SDPBackend, sdpa_kernel


class AttentionPooler(nn.Module):
    """Pool a token sequence into one vector with a single learned attention query."""

    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """(batch, seq_len, dim) -> (batch, dim)."""
        scores = self.query.expand(tokens.shape[0], -1, -1) @ tokens.transpose(1, 2)
        return (torch.softmax(scores, dim=-1) @ tokens).squeeze(1)


class Hypernetwork(nn.Module):
    """Encoder + pooler + projection MLP that writes the weights of `target`.

    Args:
        encoder: maps (batch, context_len, embedding_dim) to (batch, context_len, encoder_dim).
        target: the model whose weights are generated. Used as a shape template only.
        encoder_dim: output width of the encoder, and the size of the task representation.
        bottleneck_dim: hidden width of the projection MLP.
        num_tasks: if set, a one-hot task id (0..num_tasks-1) is projected and added to
            the task representation ("task ID" arms). If None, the hypernetwork sees the
            support examples only ("w/o task ID" arms).
        freeze_task_indicator: keep that projection at its random initialisation.
    """

    def __init__(
        self,
        encoder: nn.Module,
        target: nn.Module,
        encoder_dim: int,
        bottleneck_dim: int,
        num_tasks: int | None = None,
        freeze_task_indicator: bool = False,
    ):
        super().__init__()
        # Layers are created in this order (encoder and target first, as they arrive
        # already built) so a given seed reproduces the paper's initial weights.

        # --- Encoder: Transformer over the support set --------------------------------
        self.encoder = encoder

        # --- Target: the model whose weights are generated -----------------------------
        # Frozen: its own weights are only a template of names and shapes.
        self.target = target
        for parameter in target.parameters():
            parameter.requires_grad_(False)
        # torch.vmap cannot run the backward pass of the fused attention kernel, so the
        # target uses the plain matmul/softmax attention instead.
        for module in target.modules():
            if hasattr(module, "flash"):
                module.flash = False
        self.target_shapes = [(name, p.shape) for name, p in target.named_parameters()]
        self.num_target_weights = sum(p.numel() for p in target.parameters())

        # --- Encoder (continued): pool the tokens into one task representation --------
        self.pooler = AttentionPooler(encoder_dim)
        self.num_tasks = num_tasks
        self.task_indicator_proj = (
            nn.Linear(num_tasks, encoder_dim, bias=False) if num_tasks is not None else None
        )
        if self.task_indicator_proj is not None and freeze_task_indicator:
            self.task_indicator_proj.weight.requires_grad_(False)

        # --- Weight decoder: task representation -> every target weight ---------------
        # encoder_dim -> bottleneck_dim -> num_target_weights
        self.projection = nn.Sequential(
            nn.Linear(encoder_dim, bottleneck_dim, bias=False),
            nn.GELU(),
            nn.Linear(bottleneck_dim, self.num_target_weights, bias=False),
        )

        # The task representation from the most recent forward pass, kept for the
        # embedding cluster plots.
        self.last_task_representation: torch.Tensor | None = None

    def task_representation(
        self, context: torch.Tensor, task_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Encoder: (batch, context_len, embedding_dim) -> (batch, encoder_dim)."""
        representation = self.pooler(self.encoder(context))
        if self.task_indicator_proj is not None and task_ids is not None:
            one_hot = F.one_hot(task_ids, num_classes=self.num_tasks).float()
            representation = representation + self.task_indicator_proj(one_hot)
        self.last_task_representation = representation.detach()
        return representation

    def generate_weights(
        self, context: torch.Tensor, task_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Encoder + weight decoder: (batch, context_len, embedding_dim) ->
        (batch, num_target_weights), in float32."""
        return self.projection(self.task_representation(context, task_ids)).float()

    def split_weights(self, flat_weights: torch.Tensor) -> dict[str, torch.Tensor]:
        """Cut (batch, num_target_weights) into {name: (batch, *shape)} for the target."""
        weights = {}
        offset = 0
        for name, shape in self.target_shapes:
            size = shape.numel()
            weights[name] = flat_weights[:, offset : offset + size].reshape(-1, *shape)
            offset += size
        return weights

    def run_target(self, flat_weights: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """Target: run it on each task's inputs with that task's generated weights.

        flat_weights: (batch, num_target_weights)
        inputs: (batch, num_examples, seq_len, embedding_dim)
        returns: (batch, num_examples, seq_len, target_output_dim)
        """
        # randomness="different" gives every task its own dropout mask. The MATH backend
        # keeps scaled_dot_product_attention compatible with vmap's backward pass.
        run = torch.vmap(functional_call, in_dims=(None, 0, 0), randomness="different")
        with sdpa_kernel(SDPBackend.MATH):
            return run(self.target, self.split_weights(flat_weights), inputs)

    def forward(
        self,
        context: torch.Tensor,
        target_inputs: torch.Tensor,
        task_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """context: (batch, context_len, embedding_dim), the embedded support set.
        target_inputs: (batch, num_examples, seq_len, embedding_dim).
        Returns target logits (batch, num_examples, seq_len, target_output_dim).
        """
        # Encoder: support set -> one task representation per task.
        representation = self.task_representation(context, task_ids)
        # Weight decoder: task representation -> flat vector of all target weights.
        flat_weights = self.projection(representation).float()
        # Target: each task's examples run through the target with that task's weights.
        return self.run_target(flat_weights, target_inputs)
