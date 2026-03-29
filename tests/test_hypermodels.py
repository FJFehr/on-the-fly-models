"""Tests for the task-conditioned hypernetwork path."""

from pathlib import Path

import lightning as pl
import torch
from datasets import Dataset, DatasetDict
from torch.func import functional_call

from data_modules.arc1d_meta_padded_multiclass import Arc1dMetaPaddedMulticlassDataModule
from models import MODEL_REGISTRY
from models.hypermodels import HyperRNNMetaModelLightning


def make_task(task_id: int, base_value: int) -> dict:
    return {
        "task_category": "1d_move_1p",
        "task_id": task_id,
        "sequence_length": 4,
        "support_inputs": [[(base_value + offset) % 10] * 4 for offset in range(3)],
        "support_outputs": [[(base_value + 1 + offset) % 10] * 4 for offset in range(3)],
        "query_input": [(base_value + 4) % 10] * 4,
        "query_output": [(base_value + 5) % 10] * 4,
    }


def make_batch(batch_size: int = 2, sequence_length: int = 4) -> dict:
    support_inputs = (
        torch.arange(batch_size * 3 * sequence_length).view(batch_size, 3, sequence_length) % 10
    )
    support_outputs = (support_inputs + 1) % 10
    query_input = torch.arange(batch_size * sequence_length).view(batch_size, sequence_length) % 10
    query_output = (query_input + 2) % 10
    return {
        "support_inputs": support_inputs.long(),
        "support_outputs": support_outputs.long(),
        "query_input": query_input.long(),
        "query_output": query_output.long(),
        "task_category": ["1d_move_1p"] * batch_size,
        "task_id": torch.arange(batch_size),
    }


def test_model_registry_exposes_hyper_rnn():
    assert MODEL_REGISTRY["hyper_rnn"] is HyperRNNMetaModelLightning


def test_hyper_model_predicts_full_rnn_weights_and_backpropagates():
    """The hypernetwork should emit one full target-RNN parameter vector per task."""

    model = HyperRNNMetaModelLightning(
        input_dim=4,
        output_dim=4,
        num_classes=10,
        task_encoder_hidden_dim=8,
        task_encoder_num_heads=2,
        task_encoder_num_layers=2,
        target_rnn_hidden_dim=8,
    )
    batch = make_batch()

    logits, parameter_vectors = model(batch)
    loss = model.compute_loss(logits, model.build_targets(batch))
    loss.backward()

    assert logits.shape == (2, 4, 4, 10)
    assert parameter_vectors.shape == (2, model.target_parameter_count)
    assert model.value_embedding.weight.grad is not None
    assert model.hyper_head[-1].weight.grad is not None


def test_hyper_model_runs_generated_rnn_independently_per_example():
    """Each task example should be forwarded independently through the generated RNN."""

    model = HyperRNNMetaModelLightning(
        input_dim=4,
        output_dim=4,
        num_classes=10,
        task_encoder_hidden_dim=8,
        task_encoder_num_heads=2,
        task_encoder_num_layers=1,
        target_rnn_hidden_dim=8,
    )
    batch = make_batch(batch_size=1)

    logits, parameter_vectors = model(batch)
    parameter_mapping = model.parameter_vector_to_mapping(parameter_vectors[0])
    task_inputs = torch.cat([batch["support_inputs"], batch["query_input"].unsqueeze(1)], dim=1)[0]

    manual_logits = []
    for example_input in task_inputs:
        manual_logits.append(
            functional_call(
                model.target_model_template,
                parameter_mapping,
                (example_input.unsqueeze(0),),
            ).squeeze(0)
        )
    manual_logits = torch.stack(manual_logits, dim=0)

    assert logits.shape == (1, 4, 4, 10)
    assert parameter_vectors.shape[0] == 1
    assert torch.allclose(logits[0], manual_logits)

def test_hyper_model_metrics_track_query_support_and_all_examples():
    model = HyperRNNMetaModelLightning(
        input_dim=4,
        output_dim=4,
        num_classes=10,
        task_encoder_hidden_dim=8,
        task_encoder_num_heads=2,
        task_encoder_num_layers=1,
        target_rnn_hidden_dim=8,
    )
    targets = torch.tensor(
        [
            [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
            [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
        ]
    )
    predictions = torch.tensor(
        [
            [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
            [[1, 1, 1, 1], [9, 9, 9, 9], [3, 3, 3, 3], [9, 9, 9, 9]],
        ]
    )
    logits = torch.full((2, 4, 4, 10), -20.0)
    logits.scatter_(-1, predictions.unsqueeze(-1), 20.0)

    metrics = model.compute_metrics(logits, targets)

    assert torch.isclose(metrics["support_exact_match_accuracy"], torch.tensor(5 / 6))
    assert torch.isclose(metrics["query_exact_match_accuracy"], torch.tensor(0.5))
    assert torch.isclose(metrics["all_examples_exact_match_accuracy"], torch.tensor(0.5))


def test_hyper_model_builds_task_visualization_records():
    model = HyperRNNMetaModelLightning(
        input_dim=4,
        output_dim=4,
        num_classes=10,
        task_encoder_hidden_dim=8,
        task_encoder_num_heads=2,
        task_encoder_num_layers=1,
        target_rnn_hidden_dim=8,
    )
    batch = make_batch(batch_size=1)
    _, predictions, targets = model.predict_batch(batch)

    records = model.build_task_records(batch, predictions, targets)

    assert len(records) == 1
    record = records[0]
    assert record["task_category"] == "1d_move_1p"
    assert record["task_id"] == 0
    assert len(record["support_inputs"]) == 3
    assert len(record["support_predictions"]) == 3
    assert len(record["query_input"]) == 4
    assert len(record["query_prediction"]) == 4
    assert "query_exact_match" in record


def test_hyper_model_selects_one_representative_task_per_category(tmp_path: Path):
    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_list(
                [
                    make_task(1, 1),
                    make_task(2, 2),
                    {**make_task(3, 3), "task_category": "1d_flip"},
                ]
            ),
            "dev": Dataset.from_list([{**make_task(4, 4), "task_category": "1d_flip"}]),
            "test": Dataset.from_list([make_task(5, 5)]),
        }
    )
    dataset_path = tmp_path / "arc1d_meta_padded_multiclass"
    dataset_dict.save_to_disk(str(dataset_path))

    dm = Arc1dMetaPaddedMulticlassDataModule(
        data_dir=str(dataset_path),
        batch_size=2,
    )
    dm.setup()
    model = HyperRNNMetaModelLightning(
        input_dim=4,
        output_dim=4,
        num_classes=10,
        task_encoder_hidden_dim=8,
        task_encoder_num_heads=2,
        task_encoder_num_layers=1,
        target_rnn_hidden_dim=8,
    )

    records = model.select_representative_task_records(dm.train_dataloader())

    assert len(records) == 2
    assert {record["task_category"] for record in records} == {"1d_move_1p", "1d_flip"}


def test_hyper_model_reuses_selected_representative_task_ids(tmp_path: Path):
    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_list([make_task(1, 1), make_task(2, 2)]),
            "dev": Dataset.from_list([make_task(3, 3)]),
            "test": Dataset.from_list([make_task(4, 4)]),
        }
    )
    dataset_path = tmp_path / "arc1d_meta_padded_multiclass"
    dataset_dict.save_to_disk(str(dataset_path))

    dm = Arc1dMetaPaddedMulticlassDataModule(
        data_dir=str(dataset_path),
        batch_size=1,
    )
    dm.setup()
    model = HyperRNNMetaModelLightning(
        input_dim=4,
        output_dim=4,
        num_classes=10,
        task_encoder_hidden_dim=8,
        task_encoder_num_heads=2,
        task_encoder_num_layers=1,
        target_rnn_hidden_dim=8,
    )

    task_ids = model.select_representative_task_records_from_dataset(
        dm.train_dataset,
        split_name="train",
        limit=1,
    )
    records = model.collect_task_records_from_dataset_by_task_ids(dm.train_dataset, task_ids)

    assert task_ids == [1]
    assert model.selected_representative_task_ids["train"] == [1]
    assert len(records) == 1
    assert records[0]["task_id"] == 1


def test_hyper_model_hard_tasks_are_ranked_by_query_accuracy():
    records = [
        {"task_id": 1, "query_exact_match": False, "query_accuracy": 0.75},
        {"task_id": 2, "query_exact_match": False, "query_accuracy": 0.25},
        {"task_id": 3, "query_exact_match": True, "query_accuracy": 1.0},
    ]

    hard_records = [record for record in records if not record["query_exact_match"]]
    hard_records.sort(key=lambda record: record["query_accuracy"])

    assert [record["task_id"] for record in hard_records] == [2, 1]


def test_hyper_model_smoke_run_completes_with_lightning(tmp_path: Path):
    """A tiny task-level training run should execute end to end on train/dev/test."""

    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_list([make_task(1, 1), make_task(2, 2)]),
            "dev": Dataset.from_list([make_task(3, 3)]),
            "test": Dataset.from_list([make_task(4, 4)]),
        }
    )
    dataset_path = tmp_path / "arc1d_meta_padded_multiclass"
    dataset_dict.save_to_disk(str(dataset_path))

    dm = Arc1dMetaPaddedMulticlassDataModule(
        data_dir=str(dataset_path),
        batch_size=1,
        task_categories=["1d_move_1p"],
    )
    model = HyperRNNMetaModelLightning(
        input_dim=4,
        output_dim=4,
        num_classes=10,
        task_encoder_hidden_dim=8,
        task_encoder_num_heads=2,
        task_encoder_num_layers=1,
        target_rnn_hidden_dim=8,
    )
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
