"""Core hypermodel: wires any hypernetwork to any stateless target model."""

import math

import torch
import torch.nn as nn
from torch.func import functional_call
from torch.nn.attention import SDPBackend, sdpa_kernel

from models.transformer import Block


def _indent_repr(value: object, prefix: str = "    ") -> str:
    """Indent multiline repr output so nested modules stay readable."""
    return repr(value).replace("\n", f"\n{prefix}")


def _describe_hyper_projection(model: "HyperModel") -> str:
    """Build a compact one-line description of the hyper projection head."""
    if model.low_rank_output:
        parts = []
        for layer in model.hyper_proj_shared:
            if isinstance(layer, nn.Linear):
                parts.append(f"Linear({layer.in_features} -> {layer.out_features})")
            elif isinstance(layer, nn.GELU):
                parts.append("GELU")
        m = model._low_rank_m
        r = model.low_rank_rank
        parts.append(
            f"[A: Linear(-> {m}×{r}), B: Linear(-> {m}×{r})] rank-{r} outer-product -> {model.total_target_params}"
        )
        return " + ".join(parts)
    if model.lora_adapter:
        parts = []
        for layer in model.hyper_proj_shared:
            if isinstance(layer, nn.Linear):
                parts.append(f"Linear({layer.in_features} -> {layer.out_features})")
            elif isinstance(layer, nn.GELU):
                parts.append("GELU")
        backbone_state = "trainable" if model.lora_adapter_train_backbone else "frozen"
        backbone_kind = "zero" if model.lora_adapter_zero_backbone else "random"
        other_numel = model.lora_proj_other.out_features
        parts.append(
            f"[{len(model.lora_proj_b)}x per-tensor rank-{model.lora_adapter_rank} B@A "
            f"adapters on {backbone_state} {backbone_kind} backbone] + [other: Linear(-> {other_numel})]"
        )
        return " + ".join(parts)
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
        low_rank_output: bool = False,
        low_rank_rank: int = 1,
        lora_adapter: bool = False,
        lora_adapter_rank: int = 1,
        lora_adapter_train_backbone: bool = False,
        lora_adapter_zero_backbone: bool = False,
        freeze_task_indicator: bool = False,
    ):
        super().__init__()
        if lora_adapter and low_rank_output:
            msg = "hyper_head.lora_adapter and hyper_head.low_rank_output are mutually exclusive."
            raise ValueError(msg)
        if lora_adapter_train_backbone and not lora_adapter:
            msg = "hyper_head.lora_adapter_train_backbone requires hyper_head.lora_adapter=True."
            raise ValueError(msg)

        self.hypernetwork = hypernetwork
        self.target_model = target_model

        # The target module normally only serves as a parameter/template container: with
        # lora_adapter on, its own randomly-initialized values are also used directly as the
        # frozen (or, if lora_adapter_train_backbone, jointly-trained) backbone that the
        # generated low-rank delta is added to. Training otherwise happens entirely through
        # the hypernetwork and the hyper head.
        backbone_requires_grad = lora_adapter and lora_adapter_train_backbone
        for p in target_model.parameters():
            p.requires_grad_(backbone_requires_grad)

        # forward() below calls target_model via functional_call + vmap whenever the
        # target's own architecture supports it -- vmap has no batching rule at all for
        # nn.RNN/LSTM/GRU's fused cuDNN kernel (confirmed directly: "Batching rule not
        # implemented for aten::rnn_tanh.input"), so targets built on those (e.g.
        # target_model.name: rnn) fall back to the original per-example functional_call
        # loop instead, preserving their existing behavior exactly.
        self._use_vmap = not any(
            isinstance(module, (nn.RNN, nn.LSTM, nn.GRU)) for module in target_model.modules()
        )
        if self._use_vmap:
            # scaled_dot_product_attention's fused/flash backward kernel is incompatible
            # with vmap's batching transform (confirmed via direct GPU testing: forward
            # works, backward raises "LSE is not correctly aligned (strideH)") -- the
            # manual matmul+softmax fallback already built into RoPECanonSelfAttention (and
            # mirrored by any other target class exposing a `flash` attribute) has no such
            # issue and is itself fully vectorized by vmap. This only touches target_model's
            # own attention modules -- the hypernetwork encoder and any direct-supervised
            # use of the same classes elsewhere are unaffected, since they're never called
            # through vmap.
            for module in target_model.modules():
                if hasattr(module, "flash"):
                    module.flash = False

        self._target_parameter_specs = [
            (name, tuple(parameter.shape), parameter.numel())
            for name, parameter in self.target_model.named_parameters()
        ]

        self.hyper_output_dim = hyper_output_dim
        self.hyper_pooling = hyper_pooling or AttentionPooler(self.hyper_output_dim)
        self.num_tasks = num_tasks
        self.task_indicator_proj = (
            nn.Linear(num_tasks, hyper_output_dim, bias=False) if num_tasks is not None else None
        )
        # With freeze_task_indicator, the one-hot projection stays at its random init for
        # every category, trained or not -- so a held-out category's column is drawn from
        # the same distribution the rest of the network learned to interpret, rather than
        # being the one column that never received a gradient while its 17 siblings moved.
        self.freeze_task_indicator = freeze_task_indicator
        if self.task_indicator_proj is not None and freeze_task_indicator:
            self.task_indicator_proj.weight.requires_grad_(False)

        # Build the projection MLP from hyper_output_dim to total_target_params.
        # projection_dims specifies intermediate hidden sizes; bottleneck_dim is the
        # legacy single-intermediate fallback.
        intermediate = (
            projection_dims
            if projection_dims is not None
            else [bottleneck_dim if bottleneck_dim is not None else hyper_output_dim]
        )
        dims = [hyper_output_dim] + list(intermediate)

        self.low_rank_output = low_rank_output
        self.low_rank_rank = low_rank_rank
        self.lora_adapter = lora_adapter
        self.lora_adapter_rank = lora_adapter_rank
        self.lora_adapter_train_backbone = lora_adapter_train_backbone
        self.lora_adapter_zero_backbone = lora_adapter_zero_backbone
        if low_rank_output:
            # Shared MLP up to (but not including) the final output layer.
            shared_layers: list[nn.Module] = []
            for in_d, out_d in zip(dims[:-1], dims[1:]):
                shared_layers += [nn.Linear(in_d, out_d, bias=False), nn.GELU()]
            self.hyper_proj_shared = nn.Sequential(*shared_layers)
            # Two factor heads.  Each outputs m*r values; reshaped to (batch, m, r)
            # the rank-r outer product is A @ B^T ∈ R^{m×m}, flattened to m² values.
            self._low_rank_m = math.ceil(math.sqrt(self.total_target_params))
            self.hyper_proj_a = nn.Linear(dims[-1], self._low_rank_m * low_rank_rank, bias=False)
            self.hyper_proj_b = nn.Linear(dims[-1], self._low_rank_m * low_rank_rank, bias=False)
            # Variance-preserving init for rank-r factorisation ΔW = Σ_k u_k v_k^T:
            # Var(ΔW_ij) = r·σ⁴.  Target Xavier variance 1/m → σ = (r·m)^{-1/4}.
            with torch.no_grad():
                target_std = (low_rank_rank * self._low_rank_m) ** (-0.25)
                nn.init.normal_(self.hyper_proj_a.weight, std=target_std)
                nn.init.normal_(self.hyper_proj_b.weight, std=target_std)
            # Unused in this path but kept as empty seq so __repr__ helpers stay simple.
            self.hyper_projection = nn.Sequential()
        elif lora_adapter:
            # Shared MLP up to (but not including) the per-tensor adapter heads.
            shared_layers = []
            for in_d, out_d in zip(dims[:-1], dims[1:]):
                shared_layers += [nn.Linear(in_d, out_d, bias=False), nn.GELU()]
            self.hyper_proj_shared = nn.Sequential(*shared_layers)

            # Every 2D weight matrix gets its own rank-r pair of factor heads, sized to that
            # tensor's own (d_out, d_in) shape -- unlike low_rank_output's single global
            # reshape, this keeps the rank constraint meaningful per matrix. A ModuleList
            # (not ModuleDict) is used because parameter names contain dots.
            self._lora_2d_specs = [
                (name, shape) for name, shape, _ in self._target_parameter_specs if len(shape) == 2
            ]
            other_numel = sum(
                numel for _, shape, numel in self._target_parameter_specs if len(shape) != 2
            )

            lora_proj_b: list[nn.Module] = []
            lora_proj_a: list[nn.Module] = []
            for _, (d_out, d_in) in self._lora_2d_specs:
                b_head = nn.Linear(dims[-1], d_out * lora_adapter_rank, bias=False)
                a_head = nn.Linear(dims[-1], lora_adapter_rank * d_in, bias=False)
                # Zero-init B (real-LoRA convention): delta = B @ A is exactly zero at
                # step 0, so the effective weight starts at the backbone's own random
                # init -- an ordinary-scale network, not the all-zero collapse the old
                # low_rank_output path hit before its variance-matching fix. A keeps its
                # default init so gradients reach B (and, once B moves, A) from step 0.
                #
                # With lora_adapter_zero_backbone, the backbone is 0 too, so zero-init B
                # would make the WHOLE effective weight 0 at step 0 -- the exact all-zero
                # collapse this convention exists to avoid, and a dead-gradient trap (no
                # backbone left to carry a real forward/backward signal). B keeps its
                # ordinary random init in that case instead, confirmed necessary since a
                # zero-init B reproducibly gave gradient=0.0 at the hypernetwork in testing.
                if not lora_adapter_zero_backbone:
                    with torch.no_grad():
                        nn.init.zeros_(b_head.weight)
                lora_proj_b.append(b_head)
                lora_proj_a.append(a_head)
            self.lora_proj_b = nn.ModuleList(lora_proj_b)
            self.lora_proj_a = nn.ModuleList(lora_proj_a)
            # Non-matrix params (norms, Canon kernels) have no meaningful low-rank
            # structure -- generate them fully, same as the dense path.
            self.lora_proj_other = nn.Linear(dims[-1], other_numel, bias=False)

            # Unused in this path but kept as empty seq so __repr__ helpers stay simple.
            self.hyper_projection = nn.Sequential()
        else:
            full_dims = dims + [self.total_target_params]
            layers: list[nn.Module] = []
            for in_d, out_d in zip(full_dims[:-2], full_dims[1:-1]):
                layers += [nn.Linear(in_d, out_d, bias=False), nn.GELU()]
            layers.append(nn.Linear(full_dims[-2], full_dims[-1], bias=False))
            self.hyper_projection = nn.Sequential(*layers)

        # Disentanglement diagnostics: the pooled per-task latent that actually drove weight
        # generation (post task-descriptor add-in when active), stashed for offline cluster-map
        # visualization -- see extract_parameter_vectors.
        self._last_task_representation: torch.Tensor | None = None

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

    def build_batched_param_dict(self, param_vectors: torch.Tensor) -> dict[str, torch.Tensor]:
        """Reshape a batched (batch, total_params) tensor into {name: (batch, *shape)},
        matching torch.vmap's in_dims=0 batched-argument convention."""
        params = {}
        offset = 0
        for name, shape, numel in self._target_parameter_specs:
            params[name] = param_vectors[:, offset : offset + numel].reshape(-1, *shape)
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
        if self.task_indicator_proj is not None and task_ids is not None:
            one_hot = torch.nn.functional.one_hot(task_ids, num_classes=self.num_tasks).float()
            task_representation = task_representation + self.task_indicator_proj(one_hot)
        self._last_task_representation = task_representation.detach()
        if self.low_rank_output:
            shared = self.hyper_proj_shared(task_representation)
            batch, m, r = shared.shape[0], self._low_rank_m, self.low_rank_rank
            a = self.hyper_proj_a(shared).reshape(batch, m, r)  # (batch, m, r)
            b = self.hyper_proj_b(shared).reshape(batch, m, r)  # (batch, m, r)
            # Rank-r outer product: ΔW = A @ B^T ∈ R^{m×m}
            flat = torch.einsum("bir,bjr->bij", a, b).flatten(1)  # (batch, m*m)
            return flat[:, : self.total_target_params]
        if self.lora_adapter:
            return self._generate_lora_adapter_params(task_representation)
        return self.hyper_projection(task_representation)

    def _generate_lora_adapter_params(self, task_representation: torch.Tensor) -> torch.Tensor:
        """Generate target weights as backbone + per-tensor low-rank delta.

        Every 2D weight matrix is generated as W_backbone + B @ A, where W_backbone is the
        target model's own (frozen, or jointly-trained if lora_adapter_train_backbone) random
        init, and B, A are rank-lora_adapter_rank factors sized to that tensor's own shape.
        If lora_adapter_zero_backbone is set, W_backbone is zero instead of the target
        model's real init, so the generated delta alone determines the weight.
        Non-matrix parameters (norms, Canon kernels) have no meaningful low-rank structure and
        are generated fully, same as the dense hyper_projection path.
        """
        batch = task_representation.shape[0]
        shared = self.hyper_proj_shared(task_representation)
        r = self.lora_adapter_rank

        other_flat = self.lora_proj_other(shared)
        other_offset = 0
        lora_index = 0
        pieces: list[torch.Tensor] = []
        for name, shape, numel in self._target_parameter_specs:
            if len(shape) == 2:
                d_out, d_in = shape
                if self.lora_adapter_zero_backbone:
                    template = self.target_model.get_parameter(name)
                    base = torch.zeros(1, d_out, d_in, device=template.device, dtype=template.dtype)
                else:
                    base = self.target_model.get_parameter(name).reshape(1, d_out, d_in)
                b = self.lora_proj_b[lora_index](shared).reshape(batch, d_out, r)
                a = self.lora_proj_a[lora_index](shared).reshape(batch, r, d_in)
                delta = torch.bmm(b, a)
                pieces.append((base + delta).reshape(batch, numel))
                lora_index += 1
            else:
                pieces.append(other_flat[:, other_offset : other_offset + numel])
                other_offset += numel
        return torch.cat(pieces, dim=1)

    def forward(
        self,
        task_features: torch.Tensor,
        target_inputs: torch.Tensor,
        task_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply hypernetwork to task context, then run target model for every batch item
        (each task gets its own generated weights, so this can't be a normal batched
        forward pass through one shared set of weights): one vmapped functional_call when
        target_model's architecture supports it (self._use_vmap), else the original
        per-example functional_call loop.

        task_features  : (batch, task_seq_len, hyper_input_dim)
        target_inputs  : (batch, n_examples, seq_len, target_input_dim)
        task_ids       : (batch,) canonical task indices for the task embedding (optional)

        Returns logits  : (batch, n_examples, seq_len)

        """
        hyper_output = self.hypernetwork(task_features)
        parameter_vectors = self.extract_parameter_vectors(hyper_output, task_ids).float()

        # Loss-spike localization diagnostics: parameter_vectors is the boundary between
        # "everything the hypernetwork/hyper-head computed" (upstream) and "what the
        # target-model loop does with those generated weights" (downstream, inside vmap).
        # It's a plain (non-vmapped) tensor at this point, so both its forward magnitude
        # and its backward gradient are safe/cheap to capture here without touching the
        # vmapped internals. Comparing these against the final whole-model grad norm
        # (logged in training_step) tells us whether an explosion originates inside the
        # target-model loop (boundary grad already huge) or upstream in the hypernetwork
        # encoder/projection (boundary grad normal, but the final grad norm is huge).
        self._last_generated_weight_norm = parameter_vectors.detach().norm(dim=-1)
        if parameter_vectors.requires_grad:

            def _capture_boundary_grad(grad: torch.Tensor) -> None:
                self._last_boundary_grad_norm = grad.detach().norm(dim=-1)

            parameter_vectors.register_hook(_capture_boundary_grad)
        else:
            self._last_boundary_grad_norm = None

        if self._use_vmap:
            params = self.build_batched_param_dict(parameter_vectors)
            # randomness="different": each batch item draws its own independent dropout
            # mask (target_model.dropout is commonly nonzero), matching what the
            # equivalent per-example loop naturally did -- vmap's default ("error")
            # rejects any random op.
            #
            # sdpa_kernel(MATH): belt-and-suspenders alongside the __init__-time
            # module.flash=False forcing above. That handles the hand-rolled Canon/RoPE
            # attention classes (which branch on their own `flash` attribute); this also
            # covers any target built on nn.MultiheadAttention (e.g. target_model.name:
            # transformer), which has no `flash` attribute to toggle but calls
            # scaled_dot_product_attention internally all the same, and so is exposed to
            # the identical fused-backward-kernel-vs-vmap incompatibility.
            with sdpa_kernel(SDPBackend.MATH):
                out = torch.vmap(functional_call, in_dims=(None, 0, 0), randomness="different")(
                    self.target_model, params, target_inputs
                )
            return out.squeeze(-1)

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
