"""Focused tests for shared-embedding HyperModelLightning inputs."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.hypermodel import AttentionPooler, HierarchicalPooler, HyperModel
from models.hypermodel_lightning import HyperModelLightning
from models.rnn import RNN
from models.transformer import Transformer
from training.trainer import StopOnMetricThreshold, build_callbacks


def build_multiclass_model() -> HyperModelLightning:
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
        task_encoding={"embedding_dim": 8, "value_vocab_size": 11},
        prediction_task="multiclass",
        num_classes=10,
        padding_idx=10,
        input_dim=4,
    )


def logits_from_predictions(predictions: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
    return F.one_hot(predictions, num_classes=num_classes).float()


def test_task_context_token_ids_match_expected_metadata(build_hypermodel, make_hypermodel_batch):
    """Hypernetwork sees 6 support segments with example and role ids; no query, no is_query."""
    model = build_hypermodel()
    batch = make_hypermodel_batch()

    flat_values, position_ids, example_ids, role_ids = model.build_task_context_token_ids(
        batch["support_inputs"],
        batch["support_outputs"],
    )

    # 6 segments × 4 positions = 24 tokens
    assert example_ids.shape == (1, 24)
    assert example_ids[0].tolist() == [0] * 8 + [1] * 8 + [2] * 8
    assert role_ids[0].tolist() == [0] * 4 + [1] * 4 + [0] * 4 + [1] * 4 + [0] * 4 + [1] * 4


def test_hierarchical_pooling_reduces_serialized_support_tokens_to_one_task_vector():
    """Hierarchical pooling must collapse 6 serialized support segments to one task vector."""
    target_model = RNN(
        input_dim=8,
        hidden_dim=4,
        num_layers=1,
        bidirectional=True,
        output_dim=1,
    )
    hypermodel = HyperModel(
        hypernetwork=nn.Identity(),
        target_model=target_model,
        hyper_output_dim=8,
        bottleneck_dim=4,
        hyper_pooling=HierarchicalPooler(
            hidden_dim=8,
            num_heads=1,
            segment_interaction_layers=1,
            example_interaction_layers=1,
        ),
    )
    hyper_output = torch.randn(2, 24, 8)

    task_representation = hypermodel.extract_task_representation(hyper_output)
    parameter_vectors = hypermodel.extract_parameter_vectors(hyper_output)

    assert task_representation.shape == (2, 8)
    assert parameter_vectors.shape == (2, hypermodel.total_target_params)


def test_target_token_ids_are_value_and_position_only(build_hypermodel, make_hypermodel_batch):
    """Target model receives only value and position ids — no metadata."""
    model = build_hypermodel()
    batch = make_hypermodel_batch()

    value_ids, position_ids = model.build_target_token_ids(
        batch["support_inputs"],
        batch["query_input"],
    )

    # 4 examples (3 support + query) × 4 positions
    assert value_ids.shape == (1, 4, 4)
    assert position_ids.shape == (1, 4, 4)
    assert position_ids[0, 0].tolist() == [0, 1, 2, 3]
    assert position_ids[0, 1].tolist() == [0, 1, 2, 3]


def test_forward_runs_with_expected_shapes(build_hypermodel, make_hypermodel_batch):
    """Smoke-test the shared embedding path through the generic hypermodel."""
    model = build_hypermodel()
    batch = make_hypermodel_batch()

    task_features, example_inputs, example_targets = model.prepare_inputs(batch)
    logits, targets = model(batch)

    # Hypernetwork: 6 segments × 4 positions = 24 tokens
    assert task_features.shape == (1, 24, 8)
    # Target: 4 examples × 4 positions
    assert example_inputs.shape == (1, 4, 4, 8)
    assert example_targets.shape == (1, 4, 4)
    assert logits.shape == (1, 4, 4)
    assert torch.equal(targets, example_targets)


def test_transformer_hypernetwork_builds_and_runs_with_canonical_wrapper(
    build_hypermodel, make_hypermodel_batch
):
    """The config-facing transformer hypernetwork should remain constructible end to end."""
    model = build_hypermodel()
    batch = make_hypermodel_batch()

    assert isinstance(model.hypermodel.hypernetwork, Transformer)

    logits, targets = model(batch)

    assert logits.shape == (1, 4, 4)
    assert targets.shape == (1, 4, 4)


def test_forward_runs_with_hierarchical_pooling(build_hypermodel, make_hypermodel_batch):
    """Hierarchical pooling should preserve the end-to-end HyperModelLightning shapes."""
    model = build_hypermodel(
        hyper_head={
            "pooling": "hierarchical",
            "interaction_num_heads": 1,
            "segment_interaction_layers": 1,
            "example_interaction_layers": 1,
        },
    )
    batch = make_hypermodel_batch()

    task_features, example_inputs, example_targets = model.prepare_inputs(batch)
    logits, targets = model(batch)

    assert task_features.shape == (1, 24, 8)
    assert example_inputs.shape == (1, 4, 4, 8)
    assert example_targets.shape == (1, 4, 4)
    assert logits.shape == (1, 4, 4)
    assert torch.equal(targets, example_targets)


def test_attention_pooling_remains_the_default_hyper_head_behavior(build_hypermodel):
    """Omitting pooling config should preserve the legacy single-stage attention pooler."""
    default_model = build_hypermodel()
    explicit_model = build_hypermodel(hyper_head={"pooling": "attention"})

    assert isinstance(default_model.hypermodel.hyper_pooling, AttentionPooler)
    assert isinstance(explicit_model.hypermodel.hyper_pooling, AttentionPooler)


def test_compute_metrics_ignore_padding_positions_for_accuracy_and_exact_match():
    """Padding targets must be excluded from both accuracy and exact-match scoring."""
    model = build_multiclass_model()
    predictions = torch.tensor(
        [
            [
                [1, 2, 4, 4],
                [3, 0, 5, 0],
                [6, 7, 8, 0],
                [9, 1, 4, 3],
            ]
        ],
        dtype=torch.long,
    )
    targets = torch.tensor(
        [
            [
                [1, 2, 10, 10],
                [3, 4, 5, 10],
                [6, 7, 8, 10],
                [9, 1, 10, 10],
            ]
        ],
        dtype=torch.long,
    )

    metrics = model._metrics_from_sums(
        model._metric_sums(logits_from_predictions(predictions), targets), prefix="val"
    )

    assert torch.isclose(metrics["val_support_accuracy"], torch.tensor(7.0 / 8.0))
    assert torch.isclose(metrics["val_support_exact_match"], torch.tensor(2.0 / 3.0))
    assert torch.isclose(metrics["val_query_accuracy"], torch.tensor(1.0))
    assert torch.isclose(metrics["val_query_exact_match"], torch.tensor(1.0))
    assert "val_all_examples_exact_match" not in metrics


def test_build_task_records_trim_padding_from_visualized_sequences():
    """Task-record visualizations should drop padded suffixes before scoring or rendering."""
    model = build_multiclass_model()
    batch = {
        "support_inputs": torch.tensor(
            [[[1, 2, 10, 10], [3, 4, 5, 10], [6, 7, 8, 10]]],
            dtype=torch.long,
        ),
        "task_category": ["1d_hollow"],
        "task_id": torch.tensor([6], dtype=torch.long),
        "query_input": torch.tensor([[8, 9, 10, 10]], dtype=torch.long),
    }
    predictions = torch.tensor(
        [
            [
                [1, 2, 0, 0],
                [9, 4, 5, 0],
                [6, 7, 8, 0],
                [1, 0, 3, 4],
            ]
        ],
        dtype=torch.long,
    )
    targets = torch.tensor(
        [
            [
                [1, 2, 10, 10],
                [3, 4, 5, 10],
                [6, 7, 8, 10],
                [1, 2, 10, 10],
            ]
        ],
        dtype=torch.long,
    )

    records = model.build_task_records(batch, predictions, targets)

    assert len(records) == 1
    record = records[0]
    assert record["support_inputs"] == [[1, 2], [3, 4, 5], [6, 7, 8]]
    assert record["query_input"] == [8, 9]
    assert record["query_output"] == [1, 2]
    assert record["query_prediction"] == [1, 0]
    assert record["query_exact_match"] is False
    assert record["query_accuracy"] == 0.5


def test_forward_stashes_pooled_task_representation(build_hypermodel, make_hypermodel_batch):
    """A forward pass must stash the pooled task latent for embedding-cluster diagnostics."""
    model = build_hypermodel()
    batch = make_hypermodel_batch()

    logits, _ = model(batch)

    stashed = model.hypermodel._last_task_representation
    assert stashed is not None
    assert stashed.shape == (logits.shape[0], model.hypermodel.hyper_output_dim)


def test_forward_stashes_pooled_task_representation_with_task_descriptor(
    build_hypermodel, make_hypermodel_batch
):
    """With a task descriptor configured, the stashed vector reflects the post-add value."""
    model = build_hypermodel(hyper_head={"num_tasks": 3})
    batch = make_hypermodel_batch()
    batch["task_category"] = ["1d_move_1p"]

    logits, _ = model(batch)

    stashed = model.hypermodel._last_task_representation
    assert stashed is not None
    assert stashed.shape == (logits.shape[0], model.hypermodel.hyper_output_dim)


def test_forward_stashes_pooled_task_representation_with_lora_adapter(
    build_hypermodel, make_hypermodel_batch
):
    """The lora_adapter path must also stash the pooled task latent, same as the dense path."""
    model = build_hypermodel(hyper_head={"lora_adapter": True, "lora_adapter_rank": 1})
    batch = make_hypermodel_batch()

    logits, _ = model(batch)

    stashed = model.hypermodel._last_task_representation
    assert stashed is not None
    assert stashed.shape == (logits.shape[0], model.hypermodel.hyper_output_dim)


def test_collect_embedding_records_from_dataloader_returns_expected_records(
    build_hypermodel, make_hypermodel_batch
):
    """The embedding-collection helper should pair one pooled vector per task with its label."""
    model = build_hypermodel()
    batch = make_hypermodel_batch(task_category="1d_move_1p", task_id=2)
    dataloader = [batch, batch]

    records = model.collect_embedding_records_from_dataloader(dataloader)

    assert len(records) == 2
    for record in records:
        assert record["task_category"] == "1d_move_1p"
        assert record["task_id"] == 2
        assert record["pooled_embedding"].shape == (model.hypermodel.hyper_output_dim,)


def test_stop_on_perfect_exact_match_monitors_query_metric():
    """The perfect-exact-match stop callback should follow the query exact-match metric."""

    class DummyCfg:
        primary_metric = "val_query_exact_match"
        output_path = "/tmp"

        def get(self, key, default=None):
            if key == "stop_on_perfect_val_exact_match":
                return True
            return default

    callbacks, _ = build_callbacks(DummyCfg(), build_multiclass_model(), wandb_logger=False)
    stop_callbacks = [cb for cb in callbacks if isinstance(cb, StopOnMetricThreshold)]

    assert len(stop_callbacks) == 1
    assert stop_callbacks[0].monitor == "val_query_exact_match"
