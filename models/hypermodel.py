"""Core hypermodel: wires any hypernetwork to any stateless target model."""

import torch
import torch.nn as nn
from torch.func import functional_call


def _indent_repr(value: object, prefix: str = "    ") -> str:
    """Indent multiline repr output so nested modules stay readable."""
    return repr(value).replace("\n", f"\n{prefix}")


class HyperModel(nn.Module):
    """Wires a hypernetwork to any target model via functional_call.

    The hypernetwork emits tokenwise hidden features for each task. HyperModel then:
      - mean-pools those tokenwise features into one task representation
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

        # The hyper head turns one pooled task representation into one flat
        # parameter vector matching the full target model.
        self.hyper_head = nn.Sequential(
            nn.Linear(self.hyper_output_dim, self.hyper_output_dim, bias=False),
            nn.GELU(),
            nn.Linear(self.hyper_output_dim, self.total_target_params, bias=False),
        )

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

        task_representation = hyper_output.mean(dim=1)
        return self.hyper_head(task_representation)

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
            f"  hypernetwork = {_indent_repr(self.hypernetwork)}\n"
            f"  target       = {_indent_repr(self.target_model)}\n"
            f"  hyper_head   = {_indent_repr(self.hyper_head)}\n"
            f"  target_params = {n:,}\n"
            f")"
        )
