"""Tests for shared target-model logging behaviour.

This file intentionally carries the shared validation-example assertions using
the MLP model as the lightest concrete subclass. The CNN suite is then free to
focus on CNN-specific architecture behaviour instead of repeating the same base
class checks.
"""

import torch

import wandb
from models.target_models.mlp import TargetModelLightning
from models.target_models.positional_rnn import TargetPositionalRNNModelLightning
from visualisation import figure_to_wandb_image, render_val_example_figure


def test_target_mlp_forward_preserves_sequence_shape_with_multiple_layers():
    model = TargetModelLightning(
        input_dim=33,
        hidden_dim=16,
        output_dim=33,
        num_layers=3,
    )
    inputs = torch.randn(4, 33)

    logits = model(inputs)

    assert logits.shape == (4, 33)


def test_target_positional_rnn_forward_preserves_sequence_shape():
    model = TargetPositionalRNNModelLightning(
        input_dim=33,
        output_dim=33,
        rnn_hidden_dim=8,
    )
    inputs = torch.randn(4, 33)

    logits = model(inputs)

    assert logits.shape == (4, 33)


def test_target_positional_rnn_forward_preserves_sequence_shape_with_skips_and_depth():
    model = TargetPositionalRNNModelLightning(
        input_dim=33,
        output_dim=33,
        rnn_hidden_dim=32,
        rnn_num_layers=2,
        use_skip_connections=True,
    )
    inputs = torch.randn(4, 33)

    logits = model(inputs)

    assert logits.shape == (4, 33)


def test_build_val_example_records_contains_metadata_and_combined_image():
    """Verify validation examples retain metadata and materialize one stacked image.

    This protects the shared record-building contract in BaseTargetModel: the
    logged record must keep the task identifiers, raw sequences, exact-match
    flag, and one rendered image that W&B can log independently.
    """

    model = TargetModelLightning(
        input_dim=4,
        hidden_dim=8,
        output_dim=4,
        log_val_examples=True,
        max_logged_val_examples=2,
        max_logged_val_examples_per_category=2,
        log_val_examples_every_n_epochs=5,
    )
    batch = {
        "inputs": torch.tensor([[0, 1, 1, 0]], dtype=torch.float32),
        "targets": torch.tensor([[0, 0, 1, 0]], dtype=torch.float32),
        "task_category": ["1d_move_1p"],
        "task_id": torch.tensor([8]),
        "example_index": torch.tensor([0]),
        "source": ["query"],
    }
    preds = torch.tensor([[0, 0, 1, 0]])
    targets = torch.tensor([[0, 0, 1, 0]])
    metadata = {
        "task_category": batch["task_category"],
        "task_id": batch["task_id"],
        "example_index": batch["example_index"],
        "source": batch["source"],
    }

    records = model.build_val_example_records(batch, preds, targets, metadata)

    assert len(records) == 1
    record = records[0]
    assert record["task_category"] == "1d_move_1p"
    assert record["task_id"] == 8
    assert record["input"] == [0, 1, 1, 0]
    assert record["prediction"] == [0, 0, 1, 0]
    assert record["target"] == [0, 0, 1, 0]
    assert record["exact_match"] is True
    assert record["log_key"] == "val_example_1d_move_1p_8"
    assert isinstance(record["combined_image"], wandb.Image)


def test_build_val_example_log_payload_uses_per_example_media_keys():
    """Verify validation logging emits one W&B media item per example.

    The payload shape matters more than the image internals here because the
    logger integration expects category/task-labelled media entries.
    """

    model = TargetModelLightning(
        input_dim=4,
        hidden_dim=8,
        output_dim=4,
        log_val_examples=True,
    )
    records = [
        {
            "log_key": "val_example_cat_a_3",
            "combined_image": figure_to_wandb_image(
                render_val_example_figure(
                    input_sequence=[0, 0, 0, 0],
                    target_sequence=[1, 1, 1, 1],
                    prediction_sequence=[1, 1, 1, 1],
                )
            ),
        }
    ]

    payload = model.build_val_example_log_payload(records)

    assert set(payload) == {"val_example_cat_a_3"}
    assert "val_examples_raw" not in payload
    assert isinstance(payload["val_example_cat_a_3"], wandb.Image)


def test_build_val_example_records_limits_examples_per_category():
    """Verify validation collection keeps up to the configured per-category budget."""

    model = TargetModelLightning(
        input_dim=4,
        hidden_dim=8,
        output_dim=4,
        log_val_examples=True,
        max_logged_val_examples=10,
        max_logged_val_examples_per_category=3,
    )
    batch = {
        "inputs": torch.tensor(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [1, 0, 0, 0],
            ],
            dtype=torch.float32,
        ),
        "targets": torch.tensor(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [1, 0, 0, 0],
            ],
            dtype=torch.float32,
        ),
        "task_category": ["cat_a", "cat_a", "cat_a", "cat_a", "cat_b"],
        "task_id": torch.tensor([1, 2, 3, 4, 5]),
        "example_index": torch.tensor([0, 0, 0, 0, 0]),
        "source": ["query", "query", "query", "query", "query"],
    }
    preds = batch["targets"].long()
    metadata = {
        "task_category": batch["task_category"],
        "task_id": batch["task_id"],
        "example_index": batch["example_index"],
        "source": batch["source"],
    }

    records = model.build_val_example_records(batch, preds, batch["targets"].long(), metadata)

    assert [record["task_category"] for record in records] == ["cat_a", "cat_a", "cat_a", "cat_b"]
    assert [record["task_id"] for record in records] == [1, 2, 3, 5]
    assert model.val_examples_per_category == {"cat_a": 3, "cat_b": 1}


def test_should_log_examples_for_epoch_logs_first_and_every_fifth_epoch():
    """Verify the logging cadence keeps early visibility without flooding runs."""

    model = TargetModelLightning(
        input_dim=4,
        hidden_dim=8,
        output_dim=4,
        log_val_examples=True,
        log_val_examples_every_n_epochs=5,
    )

    decisions = []
    for epoch_index in range(1, 7):
        model.validation_epoch_count = epoch_index
        decisions.append(model.should_log_examples_for_epoch())

    assert decisions == [True, False, False, False, True, False]
