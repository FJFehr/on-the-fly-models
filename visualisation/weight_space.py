"""Helpers for offline weight-space analysis plots and vector extraction."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from matplotlib import pyplot as plt

TARGET_TASK_CATEGORY = "1d_move_1p"


def validate_v1_weight_space_configs(
    hyper_cfg: Mapping[str, object],
    direct_cfg: Mapping[str, object],
) -> dict[str, int]:
    """Validate the fixed v1 comparison contract and return shared dimensions."""
    if hyper_cfg.get("prediction_task") != "binary":
        msg = "Hyper run must use binary prediction."
        raise ValueError(msg)
    if direct_cfg.get("prediction_task") != "binary":
        msg = "Direct run must use binary prediction."
        raise ValueError(msg)

    hyper_categories = hyper_cfg.get("task_categories")
    if hyper_categories != [TARGET_TASK_CATEGORY]:
        msg = f"Hyper run must be trained only on [{TARGET_TASK_CATEGORY!r}]."
        raise ValueError(msg)

    direct_categories = direct_cfg.get("task_categories")
    if direct_categories != [TARGET_TASK_CATEGORY]:
        msg = f"Direct run must be trained only on [{TARGET_TASK_CATEGORY!r}]."
        raise ValueError(msg)

    direct_model_name = direct_cfg.get("model")
    if direct_model_name != "direct_supervised":
        msg = f"Direct run must use model='direct_supervised', got {direct_model_name!r}."
        raise ValueError(msg)

    hyper_model_name = hyper_cfg.get("model")
    if hyper_model_name not in {"hyper_model", "binary_hyper_model"}:
        msg = (
            "Hyper run must use model='hyper_model' or 'binary_hyper_model', got "
            f"{hyper_model_name!r}."
        )
        raise ValueError(msg)

    hyper_target_cfg = _expect_mapping(
        hyper_cfg.get("target_model"),
        "hyper target_model",
    )
    direct_backbone_cfg = _expect_mapping(
        direct_cfg.get("backbone_model"),
        "direct backbone_model",
    )

    if hyper_target_cfg.get("name") != "rnn":
        msg = "Hyper target_model.name must be 'rnn'."
        raise ValueError(msg)
    if direct_backbone_cfg.get("name") != "rnn":
        msg = "Direct backbone_model.name must be 'rnn'."
        raise ValueError(msg)

    hyper_target_params = _expect_mapping(
        hyper_target_cfg.get("params"),
        "hyper target_model.params",
    )
    direct_backbone_params = _expect_mapping(
        direct_backbone_cfg.get("params"),
        "direct backbone_model.params",
    )

    embedding_dim = _read_positive_int(hyper_cfg, ["task_encoding", "embedding_dim"])
    direct_embedding_dim = _read_positive_int(direct_cfg, ["task_encoding", "embedding_dim"])
    if embedding_dim != direct_embedding_dim:
        msg = (
            "Hyper and direct runs must share the same task_encoding.embedding_dim, got "
            f"{embedding_dim} and {direct_embedding_dim}."
        )
        raise ValueError(msg)

    input_dim = _read_positive_int(hyper_cfg, ["input_dim"])
    direct_input_dim = _read_positive_int(direct_cfg, ["input_dim"])
    if input_dim != direct_input_dim:
        msg = (
            "Hyper and direct runs must share the same input_dim, got "
            f"{input_dim} and {direct_input_dim}."
        )
        raise ValueError(msg)

    hidden_dim = _read_positive_int(hyper_target_params, ["hidden_dim"])
    direct_hidden_dim = _read_positive_int(direct_backbone_params, ["hidden_dim"])
    if hidden_dim != direct_hidden_dim:
        msg = (
            "Hyper target_model hidden_dim must match direct backbone hidden_dim, got "
            f"{hidden_dim} and {direct_hidden_dim}."
        )
        raise ValueError(msg)

    hyper_num_layers = _read_positive_int(hyper_target_params, ["num_layers"])
    direct_num_layers = _read_positive_int(direct_backbone_params, ["num_layers"])
    if hyper_num_layers != 1 or direct_num_layers != 1:
        msg = (
            "V1 supports only 1-layer RNNs, got "
            f"hyper={hyper_num_layers}, direct={direct_num_layers}."
        )
        raise ValueError(msg)

    hyper_bidir = _read_bool(hyper_target_params, ["bidirectional"])
    direct_bidir = _read_bool(direct_backbone_params, ["bidirectional"])
    if not hyper_bidir or not direct_bidir:
        msg = "V1 supports only bidirectional RNNs."
        raise ValueError(msg)

    return {
        "embedding_dim": embedding_dim,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
    }


def canonicalize_direct_rnn_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map a direct supervised RNN checkpoint into the hyper target parameter space."""
    output_projection = state_dict["head.weight"] @ state_dict["backbone.output_projection.weight"]
    return {
        "rnn.weight_ih_l0": state_dict["backbone.rnn.weight_ih_l0"],
        "rnn.weight_hh_l0": state_dict["backbone.rnn.weight_hh_l0"],
        "rnn.bias_ih_l0": state_dict["backbone.rnn.bias_ih_l0"],
        "rnn.bias_hh_l0": state_dict["backbone.rnn.bias_hh_l0"],
        "rnn.weight_ih_l0_reverse": state_dict["backbone.rnn.weight_ih_l0_reverse"],
        "rnn.weight_hh_l0_reverse": state_dict["backbone.rnn.weight_hh_l0_reverse"],
        "rnn.bias_ih_l0_reverse": state_dict["backbone.rnn.bias_ih_l0_reverse"],
        "rnn.bias_hh_l0_reverse": state_dict["backbone.rnn.bias_hh_l0_reverse"],
        "output_projection.weight": output_projection,
    }


def flatten_param_dict(
    param_specs: list[tuple[str, tuple[int, ...], int]],
    params: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Flatten a named parameter mapping in the canonical hyper target order."""
    flat_parts = []
    for name, shape, _ in param_specs:
        tensor = params.get(name)
        if tensor is None:
            msg = f"Missing canonical parameter {name!r}."
            raise ValueError(msg)
        if tuple(tensor.shape) != tuple(shape):
            msg = (
                f"Canonical parameter {name!r} has shape {tuple(tensor.shape)} "
                f"but expected {tuple(shape)}."
            )
            raise ValueError(msg)
        flat_parts.append(tensor.reshape(-1))
    return torch.cat(flat_parts)


def compute_pca_projection(
    hyper_vectors: torch.Tensor,
    direct_vector: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Project hyper vectors, their mean, and one direct reference into 2D PCA."""
    if hyper_vectors.ndim != 2:
        msg = (
            "hyper_vectors must have shape (num_tasks, num_params), got "
            f"{tuple(hyper_vectors.shape)}."
        )
        raise ValueError(msg)
    if direct_vector.ndim != 1:
        msg = f"direct_vector must have shape (num_params,), got {tuple(direct_vector.shape)}."
        raise ValueError(msg)
    if hyper_vectors.shape[1] != direct_vector.shape[0]:
        msg = (
            "hyper_vectors width must match direct_vector length, got "
            f"{hyper_vectors.shape[1]} and {direct_vector.shape[0]}."
        )
        raise ValueError(msg)

    fit_vectors = torch.cat([hyper_vectors, direct_vector.unsqueeze(0)], dim=0).to(torch.float64)
    fit_mean = fit_vectors.mean(dim=0, keepdim=True)
    centered_fit = fit_vectors - fit_mean

    if centered_fit.shape[0] < 2 or torch.allclose(centered_fit, torch.zeros_like(centered_fit)):
        basis = torch.zeros((fit_vectors.shape[1], 2), dtype=torch.float64)
        explained = torch.zeros(2, dtype=torch.float64)
    else:
        _, singular_values, vh = torch.linalg.svd(centered_fit, full_matrices=False)
        basis = vh[:2].transpose(0, 1)
        if basis.shape[1] < 2:
            basis = torch.cat(
                [basis, torch.zeros((basis.shape[0], 2 - basis.shape[1]), dtype=basis.dtype)],
                dim=1,
            )
        explained = torch.zeros(2, dtype=torch.float64)
        variance = singular_values.square()
        if variance.sum() > 0:
            explained[: min(2, variance.shape[0])] = variance[:2] / variance.sum()

    hyper_mean_vector = hyper_vectors.mean(dim=0, keepdim=True).to(torch.float64)
    hyper_coords = (hyper_vectors.to(torch.float64) - fit_mean) @ basis
    hyper_mean_coords = (hyper_mean_vector - fit_mean) @ basis
    direct_coords = (direct_vector.to(torch.float64).unsqueeze(0) - fit_mean) @ basis

    return {
        "hyper_coords": hyper_coords.to(torch.float32),
        "hyper_mean_coords": hyper_mean_coords.squeeze(0).to(torch.float32),
        "direct_coords": direct_coords.squeeze(0).to(torch.float32),
        "explained_variance_ratio": explained.to(torch.float32),
    }


def render_weight_space_pca_figure(
    hyper_coords: torch.Tensor,
    hyper_mean_coords: torch.Tensor,
    direct_coords: torch.Tensor,
    explained_variance_ratio: torch.Tensor,
):
    """Render the shared PCA view for the weight-space comparison."""
    figure, axis = plt.subplots(figsize=(8, 6))
    color = "#1f77b4"

    axis.scatter(
        hyper_coords[:, 0].tolist(),
        hyper_coords[:, 1].tolist(),
        s=30,
        alpha=0.22,
        marker="o",
        color=color,
        label="Hyper tasks",
    )
    axis.scatter(
        [float(hyper_mean_coords[0].item())],
        [float(hyper_mean_coords[1].item())],
        s=120,
        marker="o",
        color=color,
        edgecolors="black",
        linewidths=0.9,
        label="Hyper mean",
        zorder=3,
    )
    axis.scatter(
        [float(direct_coords[0].item())],
        [float(direct_coords[1].item())],
        s=160,
        marker="D",
        color=color,
        edgecolors="black",
        linewidths=1.2,
        label="Direct trained model",
        zorder=4,
    )

    axis.set_title("Weight-Space PCA for 1d_move_1p RNN")
    axis.set_xlabel(f"PC1 ({explained_variance_ratio[0].item() * 100:.1f}% var)")
    axis.set_ylabel(f"PC2 ({explained_variance_ratio[1].item() * 100:.1f}% var)")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    return figure


def format_weight_space_summary(
    *,
    hyper_run: str,
    direct_run: str,
    split: str,
    checkpoint: str,
    hyper_checkpoint: str,
    direct_checkpoint: str,
    num_hyper_task_vectors: int,
    canonical_vector_length: int,
    l2_distance_hyper_mean_to_direct: float,
    mean_l2_distance_hyper_tasks_to_direct: float,
    cosine_similarity_hyper_mean_to_direct: float,
) -> str:
    """Return a concise human-readable summary for `summary.txt`."""
    lines = [
        "Weight-space PCA summary",
        f"task_category: {TARGET_TASK_CATEGORY}",
        f"split: {split}",
        f"checkpoint_selector: {checkpoint}",
        f"hyper_run: {hyper_run}",
        f"direct_run: {direct_run}",
        f"hyper_checkpoint: {hyper_checkpoint}",
        f"direct_checkpoint: {direct_checkpoint}",
        f"num_hyper_task_vectors: {num_hyper_task_vectors}",
        f"canonical_vector_length: {canonical_vector_length}",
        f"l2_distance_hyper_mean_to_direct: {l2_distance_hyper_mean_to_direct:.6f}",
        f"mean_l2_distance_hyper_tasks_to_direct: {mean_l2_distance_hyper_tasks_to_direct:.6f}",
        f"cosine_similarity_hyper_mean_to_direct: {cosine_similarity_hyper_mean_to_direct:.6f}",
    ]
    return "\n".join(lines) + "\n"


def _expect_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        msg = f"{name} must be a mapping."
        raise ValueError(msg)
    return value


def _read_positive_int(mapping: Mapping[str, object], path: list[str]) -> int:
    value = _read_value(mapping, path)
    if not isinstance(value, int) or value < 1:
        dotted = ".".join(path)
        msg = f"{dotted} must be a positive integer, got {value!r}."
        raise ValueError(msg)
    return value


def _read_bool(mapping: Mapping[str, object], path: list[str]) -> bool:
    value = _read_value(mapping, path)
    if not isinstance(value, bool):
        dotted = ".".join(path)
        msg = f"{dotted} must be a bool, got {value!r}."
        raise ValueError(msg)
    return value


def _read_value(mapping: Mapping[str, object], path: list[str]) -> object:
    current: object = mapping
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            dotted = ".".join(path)
            msg = f"Missing required config value {dotted}."
            raise ValueError(msg)
        current = current[key]
    return current
