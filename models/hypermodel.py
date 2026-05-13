"""Core hypermodel: wires any hypernetwork to any stateless target model."""

import torch
import torch.nn as nn
from torch.func import functional_call

from models.transformer import Block


def _indent_repr(value: object, prefix: str = "    ") -> str:
    """Indent multiline repr output so nested modules stay readable."""
    return repr(value).replace("\n", f"\n{prefix}")


def _describe_hyper_projection(model: "HyperModel") -> str:
    """Build a compact one-line description of the hyper projection head."""
    parts = []
    for layer in model.hyper_projection:
        if isinstance(layer, nn.Linear):
            parts.append(f"Linear({layer.in_features} -> {layer.out_features})")
        elif isinstance(layer, nn.GELU):
            parts.append("GELU")
    return " + ".join(parts)


def _describe_hyper_pooling(model: "HyperModel") -> str:
    """Build a compact one-line description of the pooling path."""
    return model.hyper_pooling.describe()


class LearnedQueryAttentionPool(nn.Module):
    """Pool a token sequence to one vector with a learned attention query."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            msg = "Expected pooling inputs with shape (batch, seq_len, hidden_dim)."
            raise ValueError(msg)

        attn_weights = torch.bmm(
            self.pool_query.expand(inputs.shape[0], -1, -1),
            inputs.transpose(1, 2),
        )
        attn_weights = torch.softmax(attn_weights, dim=-1)
        return torch.bmm(attn_weights, inputs).squeeze(1)


class AttentionPooler(nn.Module):
    """Current one-shot learned attention pooling over all hypernetwork tokens."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pool = LearnedQueryAttentionPool(hidden_dim)

    def forward(self, hyper_output: torch.Tensor) -> torch.Tensor:
        return self.pool(hyper_output)

    def describe(self) -> str:
        return f"attention_pool(query over token sequence, hidden={self.hidden_dim})"


class HierarchicalPooler(nn.Module):
    """Hierarchical learned pooling for 6 support segments -> 3 examples -> 1 task vector."""

    num_segments = 6
    num_examples = 3
    segment_pair_size = 2

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        segment_interaction_layers: int,
        example_interaction_layers: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.segment_interaction_layers = segment_interaction_layers
        self.example_interaction_layers = example_interaction_layers
        self.dropout = dropout

        self.segment_pool = LearnedQueryAttentionPool(hidden_dim)
        self.segment_interaction = nn.ModuleList(
            [
                Block(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    bias=False,
                    causal=False,
                    block_size=self.num_segments,
                )
                for _ in range(segment_interaction_layers)
            ]
        )
        self.example_pool = LearnedQueryAttentionPool(hidden_dim)
        self.example_interaction = nn.ModuleList(
            [
                Block(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    bias=False,
                    causal=False,
                    block_size=self.num_examples,
                )
                for _ in range(example_interaction_layers)
            ]
        )
        self.task_pool = LearnedQueryAttentionPool(hidden_dim)

    def forward(self, hyper_output: torch.Tensor) -> torch.Tensor:
        batch_size, total_seq_len, hidden_dim = hyper_output.shape
        if total_seq_len % self.num_segments != 0:
            msg = (
                "Hierarchical pooling expects the serialized support token count "
                f"to be divisible by {self.num_segments}, got {total_seq_len}."
            )
            raise ValueError(msg)
        if hidden_dim != self.hidden_dim:
            msg = (
                "Hierarchical pooling hidden width must match the configured "
                f"hidden_dim={self.hidden_dim}, got {hidden_dim}."
            )
            raise ValueError(msg)

        segment_length = total_seq_len // self.num_segments
        segments = hyper_output.reshape(batch_size, self.num_segments, segment_length, hidden_dim)
        segment_embeddings = self.segment_pool(
            segments.reshape(batch_size * self.num_segments, segment_length, hidden_dim)
        ).reshape(batch_size, self.num_segments, hidden_dim)
        for block in self.segment_interaction:
            segment_embeddings = block(segment_embeddings)

        examples = segment_embeddings.reshape(
            batch_size,
            self.num_examples,
            self.segment_pair_size,
            hidden_dim,
        )
        example_embeddings = self.example_pool(
            examples.reshape(batch_size * self.num_examples, self.segment_pair_size, hidden_dim)
        ).reshape(batch_size, self.num_examples, hidden_dim)
        for block in self.example_interaction:
            example_embeddings = block(example_embeddings)

        return self.task_pool(example_embeddings)

    def describe(self) -> str:
        return (
            "hierarchical_pool("
            "6 segments -> attention_pool -> "
            f"{self.segment_interaction_layers}x interaction -> "
            "3 examples -> attention_pool -> "
            f"{self.example_interaction_layers}x interaction -> "
            "attention_pool -> 1 task vector)"
        )


class HyperModel(nn.Module):
    """Wires a hypernetwork to any target model via functional_call.

    The hypernetwork emits tokenwise hidden features for each task. HyperModel then:
      - attention-pools those tokenwise features into one task representation
      - projects the pooled task representation to the target parameter size
      - reshapes that flat parameter vector into the target model weights
      - applies the generated weights to every example in the task

    The target model's parameters are frozen — they serve as shape templates only.
    The only trained parameters are in the hypernetwork.
    """

    def __init__(
        self,
        hypernetwork: nn.Module,
        target_model: nn.Module,
        hyper_output_dim: int,
        bottleneck_dim: int | None = None,
        projection_dims: list[int] | None = None,
        hyper_pooling: nn.Module | None = None,
        num_tasks: int | None = None,
    ):
        super().__init__()
        self.hypernetwork = hypernetwork
        self.target_model = target_model

        # The target module is only used as a parameter/template container.
        # Training happens through the hypernetwork and the hyper head.
        for p in target_model.parameters():
            p.requires_grad_(False)

        self._target_parameter_specs = [
            (name, tuple(parameter.shape), parameter.numel())
            for name, parameter in self.target_model.named_parameters()
        ]

        self.hyper_output_dim = hyper_output_dim
        self.hyper_pooling = hyper_pooling or AttentionPooler(self.hyper_output_dim)
        self.task_embedding = (
            nn.Embedding(num_tasks, hyper_output_dim) if num_tasks is not None else None
        )

        # Build the projection MLP from hyper_output_dim to total_target_params.
        # projection_dims specifies intermediate hidden sizes; bottleneck_dim is the
        # legacy single-intermediate fallback.
        intermediate = projection_dims if projection_dims is not None else [bottleneck_dim if bottleneck_dim is not None else hyper_output_dim]
        dims = [hyper_output_dim] + list(intermediate) + [self.total_target_params]
        layers: list[nn.Module] = []
        for in_d, out_d in zip(dims[:-2], dims[1:-1]):
            layers += [nn.Linear(in_d, out_d, bias=False), nn.GELU()]
        layers.append(nn.Linear(dims[-2], dims[-1], bias=False))
        self.hyper_projection = nn.Sequential(*layers)

    @property
    def total_target_params(self) -> int:
        return sum(numel for _, _, numel in self._target_parameter_specs)

    def build_param_dict(self, param_vector: torch.Tensor) -> dict[str, torch.Tensor]:
        """Reshape a flat (total_params,) vector into {name: tensor} matching
        target_model.named_parameters()."""
        params = {}
        offset = 0
        for name, shape, numel in self._target_parameter_specs:
            params[name] = param_vector[offset : offset + numel].view(shape)
            offset += numel
        return params

    def build_batched_param_dict(self, parameter_vectors: torch.Tensor) -> dict[str, torch.Tensor]:
        """Reshape flat parameter vectors into a batched parameter mapping."""
        params = {}
        offset = 0
        for name, shape, numel in self._target_parameter_specs:
            params[name] = parameter_vectors[:, offset : offset + numel].view(
                parameter_vectors.shape[0], *shape
            )
            offset += numel
        return params

    def extract_task_representation(self, hyper_output: torch.Tensor) -> torch.Tensor:
        """Pool tokenwise hypernetwork features into one task representation."""
        if hyper_output.ndim != 3:
            msg = "Expected hypernetwork output with shape (batch, seq_len, hidden_dim)."
            raise ValueError(msg)
        if hyper_output.shape[-1] != self.hyper_output_dim:
            msg = (
                "Expected hypernetwork output hidden width to match "
                f"hyper_output_dim={self.hyper_output_dim}, got {hyper_output.shape[-1]}."
            )
            raise ValueError(msg)
        return self.hyper_pooling(hyper_output)

    def extract_parameter_vectors(
        self, hyper_output: torch.Tensor, task_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Pool tokenwise hypernetwork features and project them to target weights."""
        task_representation = self.extract_task_representation(hyper_output)
        if self.task_embedding is not None and task_ids is not None:
            task_representation = task_representation + self.task_embedding(task_ids)
        return self.hyper_projection(task_representation)

    def forward(
        self,
        task_features: torch.Tensor,
        target_inputs: torch.Tensor,
        task_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply hypernetwork to task context, then run target model for each batch item.

        task_features  : (batch, task_seq_len, hyper_input_dim)
        target_inputs  : (batch, n_examples, seq_len, target_input_dim)
        task_ids       : (batch,) canonical task indices for the task embedding (optional)

        Returns logits  : (batch, n_examples, seq_len)

        """
        hyper_output = self.hypernetwork(task_features)
        parameter_vectors = self.extract_parameter_vectors(hyper_output, task_ids)
        outputs = []
        for i in range(parameter_vectors.shape[0]):
            params = self.build_param_dict(parameter_vectors[i])
            out = functional_call(self.target_model, params, target_inputs[i])
            outputs.append(out.squeeze(-1))
        return torch.stack(outputs)

    def __repr__(self) -> str:
        n = self.total_target_params
        return (
            f"HyperModel(\n"
            f"  Hypernetwork: {_indent_repr(self.hypernetwork)}\n"
            f"  Hyper pooling: {_describe_hyper_pooling(self)}\n"
            f"  Hyper projection: {_describe_hyper_projection(self)}\n"
            f"  Target: {_indent_repr(self.target_model)}\n"
            f"  Target params: {n:,}\n"
            f")"
        )
