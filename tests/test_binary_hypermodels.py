"""Tests for the stateless binary hypernetwork path."""

import os
import subprocess
import sys
from pathlib import Path

import lightning as pl
import torch
from datasets import Dataset, DatasetDict
from omegaconf import OmegaConf

from data_modules.arc1d_meta_simple import Arc1dMetaSimpleDataModule
from models import MODEL_REGISTRY
from models.hypermodel import HyperModel
from models.hypermodel_lightning import HyperModelLightning
from models.hypermodels.binary_stateless_rnn import BinaryHyperRNNMetaModelLightning
from models.rnn import RNN
from models.transformer import Transformer
from train import build_runtime_config_dict


def make_task(task_id: int, base_value: int, category: str = "1d_move_1p") -> dict:
    return {
        "task_category": category,
        "task_id": task_id,
        "sequence_length": 4,
        "support_inputs": [[(base_value + offset) % 2] * 4 for offset in range(3)],
        "support_outputs": [[(base_value + 1 + offset) % 2] * 4 for offset in range(3)],
        "query_input": [(base_value + 1) % 2] * 4,
        "query_output": [base_value % 2] * 4,
    }


def make_batch(batch_size: int = 2, sequence_length: int = 4) -> dict:
    support_inputs = (
        torch.arange(batch_size * 3 * sequence_length).view(batch_size, 3, sequence_length) % 2
    ).float()
    support_outputs = 1.0 - support_inputs
    query_input = (
        torch.arange(batch_size * sequence_length).view(batch_size, sequence_length) % 2
    ).float()
    query_output = 1.0 - query_input
    return {
        "support_inputs": support_inputs,
        "support_outputs": support_outputs,
        "query_input": query_input,
        "query_output": query_output,
        "task_category": ["1d_move_1p"] * batch_size,
        "task_id": torch.arange(batch_size),
    }


def make_model(**overrides) -> BinaryHyperRNNMetaModelLightning:
    return BinaryHyperRNNMetaModelLightning(
        input_dim=4,
        output_dim=4,
        task_encoder_hidden_dim=8,
        task_encoder_num_heads=2,
        task_encoder_num_layers=2,
        target_rnn_hidden_dim=8,
        target_rnn_bidirectional=True,
        target_rnn_num_layers=2,
        **overrides,
    )


def test_model_registry_exposes_binary_hyper_rnn():
    assert MODEL_REGISTRY["binary_hyper_rnn"] is BinaryHyperRNNMetaModelLightning


def test_model_registry_exposes_simplified_binary_hypermodel():
    assert MODEL_REGISTRY["binary_hyper_model"] is HyperModelLightning


def test_binary_hyper_model_emits_one_parameter_vector_per_task():
    model = make_model()
    batch = make_batch(batch_size=3)

    logits, parameter_vectors = model(batch)

    assert logits.shape == (3, 4, 4)
    assert parameter_vectors.shape == (3, model.target_parameter_count)
    assert model.parameter_vector_to_mapping(parameter_vectors[0])[
        "weight_hh_l0_forward"
    ].shape == (8, 8)
    assert model.parameter_vector_to_mapping(parameter_vectors[0])[
        "weight_ih_l1_backward"
    ].shape == (8, 16)


def test_binary_hyper_model_reuses_one_parameter_mapping_across_task_examples():
    model = make_model()
    batch = make_batch(batch_size=1)

    logits, parameter_vectors = model(batch)
    parameter_mapping = model.parameter_vector_to_mapping(parameter_vectors[0])
    task_inputs = torch.cat([batch["support_inputs"], batch["query_input"].unsqueeze(1)], dim=1)[0]

    manual_logits = model.apply_generated_target_model(task_inputs, parameter_mapping)

    assert torch.allclose(logits[0], manual_logits)


def test_binary_hyper_model_only_has_hypernetwork_trainable_parameters():
    model = make_model()

    parameter_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }

    assert parameter_names
    assert all(name.startswith(("task_encoder.", "hyper_head.")) for name in parameter_names)
    assert "target_model.position_ramp" in {name for name, _ in model.named_buffers()}
    assert not any(
        name.startswith("target_model.")
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )


def test_binary_hyper_model_exposes_task_encoder_attentions_with_labels():
    model = make_model()
    batch = make_batch(batch_size=1)

    attention_data = model.get_task_encoder_attention_data(
        batch["support_inputs"],
        batch["support_outputs"],
        batch["query_input"],
    )

    assert len(attention_data["attentions"]) == 2
    assert attention_data["attentions"][0].shape == (1, 2, 28, 28)
    assert attention_data["token_labels"][0][0] == "s1_in|p00|v0"
    assert attention_data["token_labels"][0][4] == "s1_out|p00|v1"
    assert attention_data["token_labels"][0][-1] == "q_in|p03|v1"


def test_binary_hyper_model_uses_bce_and_binary_metrics():
    model = make_model()
    targets = torch.tensor(
        [
            [[1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]],
            [[1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]],
        ],
        dtype=torch.float32,
    )
    predictions = torch.tensor(
        [
            [[1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]],
            [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
        ],
        dtype=torch.long,
    )
    logits = torch.where(predictions == 1, torch.tensor(20.0), torch.tensor(-20.0)).float()

    loss = model.compute_loss(logits, targets)
    metrics = model.compute_metrics(logits, targets)

    assert loss.item() >= 0.0
    assert torch.isclose(metrics["support_exact_match_accuracy"], torch.tensor(5 / 6))
    assert torch.isclose(metrics["query_exact_match_accuracy"], torch.tensor(0.5))
    assert torch.isclose(metrics["all_examples_exact_match_accuracy"], torch.tensor(0.5))


def test_binary_hyper_model_multilayer_skip_connection_changes_parameter_spec():
    base_model = make_model(target_rnn_use_skip_connections=False)
    skip_model = make_model(target_rnn_use_skip_connections=True)

    assert "input_skip_weight" not in {spec["name"] for spec in base_model.target_parameter_specs}
    assert "input_skip_weight" in {spec["name"] for spec in skip_model.target_parameter_specs}
    assert skip_model.target_parameter_count > base_model.target_parameter_count


def test_binary_hyper_model_smoke_run_completes_with_lightning(tmp_path: Path):
    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_list([make_task(1, 0), make_task(2, 1)]),
            "dev": Dataset.from_list([make_task(3, 0)]),
            "test": Dataset.from_list([make_task(4, 1)]),
        }
    )
    dataset_path = tmp_path / "arc1d_meta_simple"
    dataset_dict.save_to_disk(str(dataset_path))

    dm = Arc1dMetaSimpleDataModule(
        data_dir=str(dataset_path),
        batch_size=1,
        task_categories=["1d_move_1p"],
    )
    model = make_model()
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        fast_dev_run=True,
    )

    trainer.fit(model=model, datamodule=dm)
    trainer.test(model=model, datamodule=dm)


def test_binary_hyper_model_mixed_task_batch_runs_end_to_end(tmp_path: Path):
    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_list(
                [make_task(1, 0, "1d_move_1p"), make_task(2, 1, "1d_denoising_1c")]
            ),
            "dev": Dataset.from_list([make_task(3, 0, "1d_denoising_1c")]),
            "test": Dataset.from_list([make_task(4, 1, "1d_move_1p")]),
        }
    )
    dataset_path = tmp_path / "arc1d_meta_simple"
    dataset_dict.save_to_disk(str(dataset_path))

    dm = Arc1dMetaSimpleDataModule(
        data_dir=str(dataset_path),
        batch_size=2,
        task_categories=["1d_move_1p", "1d_denoising_1c"],
    )
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    model = make_model()

    logits, parameter_vectors = model(batch)

    assert logits.shape == (2, 4, 4)
    assert parameter_vectors.shape[0] == 2


# ── New HyperModel architecture tests ─────────────────────────────────────────


def make_hypermodel(seq_len: int = 4, rnn_hidden: int = 8) -> HyperModelLightning:
    """Build a minimal HyperModelLightning for testing."""
    return HyperModelLightning(
        hyper_model={
            "name": "transformer",
            "params": {
                "hidden_dim": 16,
                "num_heads": 2,
                "num_layers": 2,
                "output_dim": 16,
            },
        },
        target_model={
            "name": "rnn",
            "params": {
                "hidden_dim": rnn_hidden,
                "bidirectional": True,
                "num_layers": 1,
            },
        },
        learning_rate=1e-3,
    )


def test_hypermodel_target_params_are_frozen():
    """Target model parameters must be frozen — only hypernetwork parameters are trained."""
    model = make_hypermodel()

    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}

    assert all(
        n.startswith(("hypermodel.hypernetwork.", "hypermodel.hyper_head."))
        for n in trainable
    ), trainable
    assert any(n.startswith("hypermodel.target_model.") for n in frozen), frozen


def test_hypermodel_forward_produces_correct_shape():
    """HyperModel should produce (batch, n_examples, seq_len) logits."""
    model = make_hypermodel(seq_len=4)
    batch = make_batch(batch_size=3)

    logits, targets = model(batch)

    assert logits.shape == (3, 4, 4)
    assert targets.shape == (3, 4, 4)


def test_hypermodel_prepare_inputs_matches_legacy_task_features_and_targets():
    legacy_model = make_model()
    simplified_model = HyperModelLightning(
        hyper_model={
            "name": "transformer",
            "params": {
                "hidden_dim": 8,
                "num_heads": 2,
                "num_layers": 2,
                "output_dim": 8,
            },
        },
        target_model={"name": "rnn", "params": {"hidden_dim": 8}},
    )
    batch = make_batch(batch_size=2)

    task_features, example_inputs, example_targets = simplified_model.prepare_inputs(batch)

    assert torch.allclose(
        task_features,
        legacy_model.build_task_features(
            batch["support_inputs"],
            batch["support_outputs"],
            batch["query_input"],
        ),
    )
    assert torch.allclose(
        example_targets,
        legacy_model.build_targets(batch),
    )
    assert example_inputs.shape == (2, 4, 4, 2)
    assert task_features.shape == (2, 28, 5)


def test_hypermodel_target_inputs_are_value_plus_position_only():
    model = HyperModelLightning(
        hyper_model={
            "name": "transformer",
            "params": {
                "hidden_dim": 8,
                "num_heads": 2,
                "num_layers": 2,
                "output_dim": 8,
            },
        },
        target_model={"name": "rnn", "params": {"hidden_dim": 8}},
    )
    batch = make_batch(batch_size=1, sequence_length=4)
    _, example_inputs, _ = model.prepare_inputs(batch)

    expected_positions = torch.tensor([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    assert torch.allclose(example_inputs[0, 0, :, 0], batch["support_inputs"][0, 0])
    assert torch.allclose(example_inputs[0, 3, :, 0], batch["query_input"][0])
    assert torch.allclose(example_inputs[0, 0, :, 1], expected_positions)
    assert torch.allclose(example_inputs[0, 3, :, 1], expected_positions)


def test_hypermodel_uses_full_task_bce_loss_and_keeps_split_metrics():
    simplified_model = HyperModelLightning(
        hyper_model={
            "name": "transformer",
            "params": {
                "hidden_dim": 8,
                "num_heads": 2,
                "num_layers": 2,
                "output_dim": 8,
            },
        },
        target_model={"name": "rnn", "params": {"hidden_dim": 8}},
    )
    targets = torch.tensor(
        [
            [[1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]],
            [[1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]],
        ],
        dtype=torch.float32,
    )
    predictions = torch.tensor(
        [
            [[1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]],
            [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
        ],
        dtype=torch.long,
    )
    logits = torch.where(predictions == 1, torch.tensor(20.0), torch.tensor(-20.0)).float()

    simplified_loss = simplified_model.compute_loss(logits, targets)
    simplified_metrics = simplified_model.compute_metrics(logits, targets)
    expected_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)

    assert torch.isclose(expected_loss, simplified_loss)
    assert torch.isclose(simplified_metrics["support_exact_match_accuracy"], torch.tensor(5 / 6))
    assert torch.isclose(simplified_metrics["query_exact_match_accuracy"], torch.tensor(0.5))
    assert torch.isclose(
        simplified_metrics["all_examples_exact_match_accuracy"], torch.tensor(0.5)
    )


def test_hypermodel_reuses_one_generated_parameter_vector_per_task():
    model = HyperModelLightning(
        hyper_model={
            "name": "transformer",
            "params": {
                "hidden_dim": 8,
                "num_heads": 2,
                "num_layers": 2,
                "output_dim": 8,
            },
        },
        target_model={"name": "rnn", "params": {"hidden_dim": 8}},
    )
    batch = make_batch(batch_size=1)

    task_features, example_inputs, _ = model.prepare_inputs(batch)
    logits = model.hypermodel(task_features, example_inputs)
    parameter_vectors = model.hypermodel.extract_parameter_vectors(
        model.hypermodel.hypernetwork(task_features)
    )
    parameter_mapping = model.hypermodel.build_param_dict(parameter_vectors[0])
    manual_logits = model.hypermodel.apply_target(parameter_mapping, example_inputs[0])

    assert parameter_vectors.shape[0] == 1
    assert torch.allclose(logits[0], manual_logits)


def test_hypermodel_mean_pools_hidden_features_before_projection():
    target = RNN(
        input_dim=2,
        hidden_dim=8,
        num_layers=1,
        output_dim=1,
        bidirectional=True,
    )
    hypernetwork = Transformer(
        input_dim=5,
        hidden_dim=16,
        num_layers=2,
        num_heads=2,
        output_dim=16,
    )
    hypermodel = HyperModel(
        hypernetwork=hypernetwork,
        target_model=target,
        hyper_output_dim=16,
    )
    task_features = torch.arange(2 * 7 * 5, dtype=torch.float32).view(2, 7, 5)

    raw = hypernetwork(task_features)
    parameter_vectors = hypermodel.extract_parameter_vectors(raw)
    expected = hypermodel.hyper_head(raw.mean(dim=1))

    assert torch.allclose(parameter_vectors, expected)
    assert parameter_vectors.shape == (2, hypermodel.total_target_params)


def test_hypermodel_projects_batch_hidden_features_to_target_params():
    target = RNN(
        input_dim=2,
        hidden_dim=8,
        num_layers=1,
        output_dim=1,
        bidirectional=True,
    )
    hypernetwork = torch.nn.Identity()
    hypermodel = HyperModel(
        hypernetwork=hypernetwork,
        target_model=target,
        hyper_output_dim=16,
    )
    hidden_features = torch.randn(3, 16)

    try:
        hypermodel.extract_parameter_vectors(hidden_features)
    except ValueError as error:
        assert "shape (batch, seq_len, hidden_dim)" in str(error)
    else:
        raise AssertionError("Expected 2D hypernetwork output to raise ValueError.")

def test_hypermodel_rejects_invalid_hidden_width():
    target = RNN(
        input_dim=2,
        hidden_dim=8,
        num_layers=1,
        output_dim=1,
        bidirectional=True,
    )
    hypernetwork = torch.nn.Identity()
    hypermodel = HyperModel(
        hypernetwork=hypernetwork,
        target_model=target,
        hyper_output_dim=16,
    )

    try:
        hypermodel.extract_parameter_vectors(torch.randn(3, 7, 15))
    except ValueError as error:
        assert "hidden width" in str(error)
    else:
        raise AssertionError("Expected hidden-width mismatch to raise ValueError.")


def test_hypermodel_lightning_constructor_runs_from_nested_config():
    model = HyperModelLightning(
        hyper_model={
            "name": "transformer",
            "params": {
                "hidden_dim": 8,
                "num_heads": 2,
                "num_layers": 2,
                "output_dim": 8,
            },
        },
        target_model={"name": "rnn", "params": {"hidden_dim": 8}},
    )
    batch = make_batch(batch_size=2)

    logits, targets = model(batch)

    assert logits.shape == (2, 4, 4)
    assert targets.shape == (2, 4, 4)


def test_build_runtime_config_dict_keeps_nested_binary_hypermodel_sections():
    cfg = OmegaConf.create(
        {
            "model": "binary_hyper_model",
            "learning_rate": 0.001,
            "hyper_model": {
                "name": "transformer",
                "params": {
                    "hidden_dim": 8,
                    "num_heads": 2,
                    "num_layers": 2,
                    "output_dim": 8,
                },
            },
            "target_model": {
                "name": "rnn",
                "params": {
                    "hidden_dim": 8,
                    "bidirectional": True,
                    "num_layers": 1,
                },
            },
        }
    )

    runtime_cfg = build_runtime_config_dict(cfg)

    assert runtime_cfg["hyper_model"]["name"] == "transformer"
    assert runtime_cfg["hyper_model"]["params"]["hidden_dim"] == 8
    assert runtime_cfg["target_model"]["name"] == "rnn"
    assert runtime_cfg["target_model"]["params"]["hidden_dim"] == 8
    assert "target_rnn_hidden_dim" not in runtime_cfg


def test_hypermodel_lightning_smoke_run(tmp_path: Path):
    """Full Lightning train/test loop should complete without error."""
    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_list([make_task(1, 0), make_task(2, 1)]),
            "dev": Dataset.from_list([make_task(3, 0)]),
            "test": Dataset.from_list([make_task(4, 1)]),
        }
    )
    dataset_path = tmp_path / "arc1d_meta_simple"
    dataset_dict.save_to_disk(str(dataset_path))

    dm = Arc1dMetaSimpleDataModule(
        data_dir=str(dataset_path), batch_size=1, task_categories=["1d_move_1p"]
    )
    model = make_hypermodel(seq_len=4)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        fast_dev_run=True,
    )

    trainer.fit(model=model, datamodule=dm)
    trainer.test(model=model, datamodule=dm)


def test_train_binary_hypermodel_entrypoint_runs(tmp_path: Path):
    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_list([make_task(1, 0), make_task(2, 1)]),
            "dev": Dataset.from_list([make_task(3, 0)]),
            "test": Dataset.from_list([make_task(4, 1)]),
        }
    )
    dataset_path = tmp_path / "arc1d_meta_simple"
    dataset_dict.save_to_disk(str(dataset_path))
    output_path = tmp_path / "outputs"
    config_path = tmp_path / "binary_hypermodel.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project_name: test_binary_hypermodel",
                "output_dir: " + str(output_path),
                "seed: 42",
                "experiment_name: smoke",
                "output_path: " + str(output_path / "smoke"),
                "data: arc_1d_meta_simple",
                "data_dir: " + str(dataset_path),
                "model: binary_hyper_model",
                "primary_metric: val_query_exact_match_accuracy",
                "num_workers: 0",
                "task_categories:",
                "  - 1d_move_1p",
                "train_split: train",
                "val_split: val",
                "test_split: test",
                "overfit_single_batch: false",
                "accelerator: cpu",
                "batch_size: 1",
                "learning_rate: 0.001",
                "optimizer: Adam",
                "weight_decay: 0.0",
                "max_steps: 1",
                "input_dim: 4",
                "output_dim: 4",
                "prediction_task: binary",
                "stop_on_perfect_val_exact_match: false",
                "hyper_model:",
                "  name: transformer",
                "  params:",
                "    hidden_dim: 8",
                "    num_heads: 2",
                "    num_layers: 2",
                "    output_dim: 8",
                "target_model:",
                "  name: rnn",
                "  params:",
                "    hidden_dim: 8",
                "    bidirectional: true",
                "    num_layers: 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"
    env["WANDB_SILENT"] = "true"

    subprocess.run(
        [
            sys.executable,
            "train_binary_hypermodel.py",
            "--config",
            str(config_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert (output_path / "smoke" / "config.yaml").exists()
    assert (output_path / "smoke" / "model.txt").exists()
    assert (output_path / "smoke" / "results.txt").exists()
