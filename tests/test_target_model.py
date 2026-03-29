"""Tests for shared target-model logging behaviour."""

import pytest
import torch
import wandb
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

from models import MODEL_REGISTRY
from models.target_models.mlp import TargetMLPModelLightning
from models.target_models.rnn import TargetRNNModelLightning
from models.target_models.transformer import TargetTransformerModelLightning
from visualisation import figure_to_wandb_image, render_val_example_figure


def test_model_registry_exposes_simplified_model_keys():
    assert MODEL_REGISTRY["mlp"] is TargetMLPModelLightning
    assert "cnn" in MODEL_REGISTRY
    assert MODEL_REGISTRY["rnn"] is TargetRNNModelLightning
    assert MODEL_REGISTRY["transformer"] is TargetTransformerModelLightning


def test_target_mlp_forward_preserves_sequence_shape_with_multiple_layers():
    model = TargetMLPModelLightning(
        input_dim=33,
        hidden_dim=16,
        output_dim=33,
        num_layers=3,
    )
    inputs = torch.randn(4, 33)

    logits = model(inputs)

    assert logits.shape == (4, 33)


def test_target_mlp_forward_returns_multiclass_logits():
    model = TargetMLPModelLightning(
        input_dim=33,
        hidden_dim=16,
        output_dim=33,
        prediction_task="multiclass",
        num_classes=10,
    )
    inputs = torch.randint(0, 10, (4, 33)).float()

    logits = model(inputs)

    assert logits.shape == (4, 33, 10)


def test_target_rnn_forward_preserves_sequence_shape():
    model = TargetRNNModelLightning(
        input_dim=33,
        output_dim=33,
        rnn_hidden_dim=8,
    )
    inputs = torch.randn(4, 33)

    logits = model(inputs)

    assert logits.shape == (4, 33)


def test_target_rnn_forward_preserves_sequence_shape_with_skips_and_depth():
    model = TargetRNNModelLightning(
        input_dim=33,
        output_dim=33,
        rnn_hidden_dim=32,
        rnn_num_layers=2,
        use_skip_connections=True,
    )
    inputs = torch.randn(4, 33)

    logits = model(inputs)

    assert logits.shape == (4, 33)


def test_target_rnn_forward_returns_multiclass_logits():
    model = TargetRNNModelLightning(
        input_dim=33,
        output_dim=33,
        rnn_hidden_dim=8,
        prediction_task="multiclass",
        num_classes=10,
    )
    inputs = torch.randint(0, 10, (4, 33)).float()

    logits = model(inputs)

    assert logits.shape == (4, 33, 10)


def test_target_transformer_forward_returns_multiclass_logits():
    model = TargetTransformerModelLightning(
        input_dim=33,
        output_dim=33,
        transformer_hidden_dim=40,
        num_heads=2,
        num_layers=1,
        prediction_task="multiclass",
        num_classes=10,
    )
    inputs = torch.randint(0, 10, (4, 33)).float()

    logits = model(inputs)

    assert logits.shape == (4, 33, 10)


def test_target_transformer_supports_causal_attention_toggle():
    inputs = torch.randint(0, 10, (4, 33)).float()

    encoder_style = TargetTransformerModelLightning(
        input_dim=33,
        output_dim=33,
        transformer_hidden_dim=40,
        num_heads=2,
        num_layers=1,
        prediction_task="multiclass",
        num_classes=10,
        causal_attention=False,
    )
    decoder_style = TargetTransformerModelLightning(
        input_dim=33,
        output_dim=33,
        transformer_hidden_dim=40,
        num_heads=2,
        num_layers=1,
        prediction_task="multiclass",
        num_classes=10,
        causal_attention=True,
    )

    assert encoder_style(inputs).shape == (4, 33, 10)
    assert decoder_style(inputs).shape == (4, 33, 10)


def test_common_step_uses_cross_entropy_and_argmax_for_multiclass():
    """Verify the shared base logic switches loss and decoding for multiclass tasks."""

    model = TargetRNNModelLightning(
        input_dim=4,
        output_dim=4,
        rnn_hidden_dim=8,
        prediction_task="multiclass",
        num_classes=10,
    )
    batch = {
        "inputs": torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=torch.float32),
        "targets": torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long),
        "task_category": ["cat_a", "cat_b"],
        "task_id": torch.tensor([1, 2]),
        "example_index": torch.tensor([0, 0]),
        "source": ["query", "query"],
    }

    loss, preds, targets, _ = model.common_step(batch, "val")

    assert isinstance(model.loss_fn, CrossEntropyLoss)
    assert torch.isfinite(loss)
    assert preds.shape == (2, 4)
    assert targets.shape == (2, 4)
    assert preds.min().item() >= 0
    assert preds.max().item() <= 9


def test_format_logits_rejects_flattened_multiclass_outputs():
    model = TargetMLPModelLightning(
        input_dim=4,
        hidden_dim=8,
        output_dim=4,
        prediction_task="multiclass",
        num_classes=10,
    )
    targets = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long)
    flattened_logits = torch.randn(2, 40)

    with pytest.raises(ValueError, match="Multiclass models must return logits with shape"):
        model.format_logits(flattened_logits, targets)


def test_binary_models_keep_bce_loss():
    model = TargetMLPModelLightning(
        input_dim=4,
        hidden_dim=8,
        output_dim=4,
    )

    assert isinstance(model.loss_fn, BCEWithLogitsLoss)


def test_build_val_example_records_contains_metadata_and_combined_image():
    """Verify validation examples retain metadata and materialize one stacked image."""

    model = TargetMLPModelLightning(
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
    """Verify validation logging emits one W&B media item per example."""

    model = TargetMLPModelLightning(
        input_dim=4,
        hidden_dim=8,
        output_dim=4,
        log_val_examples=True,
        max_logged_val_examples=2,
        max_logged_val_examples_per_category=2,
        log_val_examples_every_n_epochs=5,
    )
    records = [
        {
            "log_key": "val_example_1d_move_1p_8",
            "combined_image": wandb.Image(render_val_example_figure([0], [0], [0])),
            "task_category": "1d_move_1p",
            "task_id": 8,
            "input": [0],
            "prediction": [0],
            "target": [0],
            "exact_match": True,
        }
    ]

    payload = model.build_val_example_log_payload(records)

    assert list(payload) == ["val_example_1d_move_1p_8"]
    assert isinstance(payload["val_example_1d_move_1p_8"], wandb.Image)


def test_figure_to_wandb_image_preserves_caption():
    figure = render_val_example_figure(
        input_sequence=[0, 1, 0],
        target_sequence=[1, 1, 0],
        prediction_sequence=[1, 0, 0],
    )

    image = figure_to_wandb_image(figure, caption="caption")

    assert isinstance(image, wandb.Image)
    assert image._caption == "caption"


def test_should_log_val_examples_respects_schedule():
    model = TargetMLPModelLightning(
        input_dim=4,
        hidden_dim=8,
        output_dim=4,
        log_val_examples=True,
        log_val_examples_every_n_epochs=5,
    )
    model.validation_epoch_count = 1
    assert model.should_log_examples_for_epoch() is True
    model.validation_epoch_count = 3
    assert model.should_log_examples_for_epoch() is False


def test_should_log_val_examples_can_be_disabled():
    model = TargetMLPModelLightning(
        input_dim=4,
        hidden_dim=8,
        output_dim=4,
        log_val_examples=False,
    )

    assert model.should_log_examples_for_epoch() is False
