"""Tests for the stateless binary hypernetwork path."""

from pathlib import Path

import lightning as pl
import torch
from datasets import Dataset, DatasetDict

from data_modules.arc1d_meta_simple import Arc1dMetaSimpleDataModule
from models import MODEL_REGISTRY
from models.hypermodels.binary_stateless_rnn import BinaryHyperRNNMetaModelLightning


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


def test_binary_hyper_model_emits_one_parameter_vector_per_task():
    model = make_model()
    batch = make_batch(batch_size=3)

    logits, parameter_vectors = model(batch)

    assert logits.shape == (3, 4, 4)
    assert parameter_vectors.shape == (3, model.target_parameter_count)
    assert (
        model.parameter_vector_to_mapping(parameter_vectors[0])["weight_hh_l0_forward"].shape
        == (8, 8)
    )
    assert (
        model.parameter_vector_to_mapping(parameter_vectors[0])["weight_ih_l1_backward"].shape
        == (8, 16)
    )


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
