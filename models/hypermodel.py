"""Core hypermodel: wires any hypernetwork to any stateless target model."""

import torch
import torch.nn as nn
from torch.func import functional_call


def _indent_repr(value: object, prefix: str = "    ") -> str:
    """Indent multiline repr output so nested modules stay readable."""
    return repr(value).replace("\n", f"\n{prefix}")


def _describe_hyper_projection(model: "HyperModel") -> str:
    """Build a compact one-line description of the hyper projection head."""
    if model.use_vae:
        parts = [
            f"mu: Linear({model.hyper_output_dim} -> {model.bottleneck_dim}) + GELU",
            f"log_sigma: Linear({model.hyper_output_dim} -> {model.bottleneck_dim})",
            "reparam -> z",
            f"Linear({model.bottleneck_dim} -> {model.total_target_params})",
        ]
    else:
        parts = [
            f"Linear({model.hyper_output_dim} -> {model.bottleneck_dim})",
            "GELU",
            f"Linear({model.bottleneck_dim} -> {model.total_target_params})",
        ]
    return " + ".join(parts)


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
        use_vae: bool = False,
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
        self.bottleneck_dim = bottleneck_dim if bottleneck_dim is not None else hyper_output_dim
        self.use_vae = use_vae

        # Learned query vector for attention pooling over the token sequence.
        self.pool_query = nn.Parameter(torch.randn(1, 1, self.hyper_output_dim))

        if self.use_vae:
            # VAE bottleneck: two parallel heads produce mu and log_sigma.
            # At train time z is sampled via the reparameterisation trick;
            # at eval time z = mu (no noise).
            self.hyper_mu = nn.Sequential(
                nn.Linear(self.hyper_output_dim, self.bottleneck_dim, bias=False),
                nn.GELU(),
            )
            self.hyper_log_sigma = nn.Linear(
                self.hyper_output_dim, self.bottleneck_dim, bias=False
            )
        else:
            # Deterministic bottleneck.
            self.hyper_bottleneck = nn.Sequential(
                nn.Linear(self.hyper_output_dim, self.bottleneck_dim, bias=False),
                nn.GELU(),
            )

        # Projects the bottleneck code to the full target model parameter vector.
        self.hyper_out = nn.Linear(self.bottleneck_dim, self.total_target_params, bias=False)

        # Initialised to zero; updated every forward pass so the lightning module
        # can read the KL term without changing the forward() return signature.
        self._last_kl: torch.Tensor = torch.tensor(0.0)

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

    def extract_parameter_vectors(self, hyper_output: torch.Tensor) -> torch.Tensor:
        """Pool tokenwise hypernetwork features and project them to target weights."""
        if hyper_output.ndim != 3:
            msg = "Expected hypernetwork output with shape (batch, seq_len, hidden_dim)."
            raise ValueError(msg)
        if hyper_output.shape[-1] != self.hyper_output_dim:
            msg = (
                "Expected hypernetwork output hidden width to match "
                f"hyper_output_dim={self.hyper_output_dim}, got {hyper_output.shape[-1]}."
            )
            raise ValueError(msg)

        # Attention pooling: learned query attends over the token sequence,
        # allowing the model to focus on the most informative tokens rather
        # than averaging over all (including many uninformative inactive-bit tokens).
        attn_weights = torch.bmm(
            self.pool_query.expand(hyper_output.shape[0], -1, -1),
            hyper_output.transpose(1, 2),
        )  # (batch, 1, seq_len)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        task_representation = torch.bmm(attn_weights, hyper_output).squeeze(1)  # (batch, hidden)

        if self.use_vae:
            mu = self.hyper_mu(task_representation)
            log_sigma = self.hyper_log_sigma(task_representation)
            if self.training:
                z = mu + torch.exp(0.5 * log_sigma) * torch.randn_like(mu)
            else:
                z = mu  # use posterior mean at eval time
            # KL divergence: KL(N(mu, sigma) || N(0, I)), averaged over the batch.
            kl = -0.5 * (1 + log_sigma - mu.pow(2) - log_sigma.exp()).sum(dim=-1).mean()
        else:
            z = self.hyper_bottleneck(task_representation)
            kl = task_representation.new_tensor(0.0)

        self._last_kl = kl
        return self.hyper_out(z)

    def apply_target(
        self, params: dict[str, torch.Tensor], example_inputs: torch.Tensor
    ) -> torch.Tensor:
        """Run target_model with generated params on a single batch item's examples.

        example_inputs: (n_examples, seq_len, input_dim)
        returns:        (n_examples, seq_len)
        """
        out = functional_call(self.target_model, params, example_inputs)
        return out.squeeze(-1)

    def forward(self, task_features: torch.Tensor, target_inputs: torch.Tensor) -> torch.Tensor:
        """Apply hypernetwork to task context, then run target model for each batch item.

        task_features  : (batch, task_seq_len, hyper_input_dim)
        target_inputs  : (batch, n_examples, seq_len, target_input_dim)

        Returns logits  : (batch, n_examples, seq_len)

        """
        # The hypernetwork produces one hidden feature vector per serialized task token.
        hyper_output = self.hypernetwork(task_features)

        parameter_vectors = self.extract_parameter_vectors(hyper_output)

        outputs = []
        # Each task in the batch gets its own generated parameter mapping.
        for i in range(parameter_vectors.shape[0]):
            params = self.build_param_dict(parameter_vectors[i])
            outputs.append(self.apply_target(params, target_inputs[i]))
        return torch.stack(outputs)

    def __repr__(self) -> str:
        n = self.total_target_params
        return (
            f"HyperModel(\n"
            f"  Hypernetwork: {_indent_repr(self.hypernetwork)}\n"
            f"  Hyper projection: {_describe_hyper_projection(self)}\n"
            f"  Target: {_indent_repr(self.target_model)}\n"
            f"  Target params: {n:,}\n"
            f")"
        )
