"""Focused tests for offline weight-space analysis helpers."""

from pathlib import Path

import pytest
import torch

from models.direct_supervised_lightning import DirectSupervisedLightning
from models.hypermodel_lightning import HyperModelLightning
from visualisation.weight_space import (
    canonicalize_direct_rnn_state_dict,
    compute_pca_projection,
    flatten_param_dict,
    render_weight_space_pca_figure,
    validate_v1_weight_space_configs,
)


def build_hyper_model() -> HyperModelLightning:
    return HyperModelLightning(
        hyper_model={
            "name": "transformer",
            "params": {
                "hidden_dim": 16,
                "num_layers": 1,
                "num_heads": 1,
                "output_dim": 8,
            },
        },
        target_model={
            "name": "rnn",
            "params": {
                "hidden_dim": 8,
                "num_layers": 1,
                "bidirectional": True,
            },
        },
        task_encoding={"embedding_dim": 8},
        input_dim=4,
        prediction_task="binary",
        task_categories=["1d_move_1p"],
    )


def build_direct_model() -> DirectSupervisedLightning:
    return DirectSupervisedLightning(
        backbone_model={
            "name": "rnn",
            "params": {
                "hidden_dim": 8,
                "num_layers": 1,
                "bidirectional": True,
            },
        },
        task_encoding={"embedding_dim": 8},
        input_dim=4,
        prediction_task="binary",
        task_categories=["1d_move_1p"],
    )


def test_validate_v1_weight_space_configs_rejects_wrong_task_category():
    """V1 should only accept the fixed binary `1d_move_1p` slice."""
    hyper_cfg = {
        "model": "binary_hyper_model",
        "prediction_task": "binary",
        "task_categories": ["1d_fill"],
        "task_encoding": {"embedding_dim": 8},
        "input_dim": 4,
        "target_model": {
            "name": "rnn",
            "params": {"hidden_dim": 8, "num_layers": 1, "bidirectional": True},
        },
    }
    direct_cfg = {
        "model": "direct_supervised",
        "prediction_task": "binary",
        "task_categories": ["1d_move_1p"],
        "task_encoding": {"embedding_dim": 8},
        "input_dim": 4,
        "backbone_model": {
            "name": "rnn",
            "params": {"hidden_dim": 8, "num_layers": 1, "bidirectional": True},
        },
    }

    with pytest.raises(ValueError, match="1d_move_1p"):
        validate_v1_weight_space_configs(hyper_cfg, direct_cfg)


def test_direct_canonical_vector_matches_hyper_target_parameter_count():
    """Direct RNN weights must flatten into the exact hyper target parameter space."""
    hyper_model = build_hyper_model()
    direct_model = build_direct_model()

    direct_params = canonicalize_direct_rnn_state_dict(direct_model.state_dict())
    direct_vector = flatten_param_dict(
        hyper_model.hypermodel._target_parameter_specs,
        direct_params,
    )

    assert direct_vector.shape == (hyper_model.hypermodel.total_target_params,)


def test_pca_projection_and_figure_render_for_valid_vectors(tmp_path: Path):
    """PCA helpers should return 2D coordinates and save a figure cleanly."""
    hyper_vectors = torch.randn(5, 12)
    direct_vector = torch.randn(12)

    projection = compute_pca_projection(hyper_vectors, direct_vector)
    figure = render_weight_space_pca_figure(
        hyper_coords=projection["hyper_coords"],
        hyper_mean_coords=projection["hyper_mean_coords"],
        direct_coords=projection["direct_coords"],
        explained_variance_ratio=projection["explained_variance_ratio"],
    )
    output_path = tmp_path / "pca.png"
    figure.savefig(output_path, dpi=100, bbox_inches="tight")

    assert projection["hyper_coords"].shape == (5, 2)
    assert projection["hyper_mean_coords"].shape == (2,)
    assert projection["direct_coords"].shape == (2,)
    assert output_path.exists()
