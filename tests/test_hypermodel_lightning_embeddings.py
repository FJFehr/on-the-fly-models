"""Focused tests for shared-embedding HyperModelLightning inputs."""

import torch
import torch.nn as nn

from models.hypermodel import AttentionPooler, HierarchicalPooler, HyperModel
from models.hypermodel_lightning import HyperModelLightning
from models.rnn import RNN
from models.transformer import Transformer


def build_model(
    task_encoding: dict | None = None,
    hyper_head: dict | None = None,
) -> HyperModelLightning:
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
        hyper_head=hyper_head,
        task_encoding=task_encoding,
        input_dim=4,
    )


def make_batch() -> dict:
    return {
        "support_inputs": torch.tensor(
            [[[0, 1, 0, 1], [1, 0, 1, 0], [0, 0, 1, 1]]],
            dtype=torch.float32,
        ),
        "support_outputs": torch.tensor(
            [[[1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 0, 1]]],
            dtype=torch.float32,
        ),
        "query_input": torch.tensor([[1, 0, 0, 1]], dtype=torch.float32),
        "query_output": torch.tensor([[0, 1, 1, 0]], dtype=torch.float32),
    }


def test_task_context_token_ids_match_expected_metadata():
    """Hypernetwork sees 6 support segments with example and role ids; no query, no is_query."""
    model = build_model({"embedding_dim": 8})
    batch = make_batch()

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


def test_target_token_ids_are_value_and_position_only():
    """Target model receives only value and position ids — no metadata."""
    model = build_model({"embedding_dim": 8})
    batch = make_batch()

    value_ids, position_ids = model.build_target_token_ids(
        batch["support_inputs"],
        batch["query_input"],
    )

    # 4 examples (3 support + query) × 4 positions
    assert value_ids.shape == (1, 4, 4)
    assert position_ids.shape == (1, 4, 4)
    assert position_ids[0, 0].tolist() == [0, 1, 2, 3]
    assert position_ids[0, 1].tolist() == [0, 1, 2, 3]


def test_forward_runs_with_expected_shapes():
    """Smoke-test the shared embedding path through the generic hypermodel."""
    model = build_model({"embedding_dim": 8})
    batch = make_batch()

    task_features, example_inputs, example_targets = model.prepare_inputs(batch)
    logits, targets = model(batch)

    # Hypernetwork: 6 segments × 4 positions = 24 tokens
    assert task_features.shape == (1, 24, 8)
    # Target: 4 examples × 4 positions
    assert example_inputs.shape == (1, 4, 4, 8)
    assert example_targets.shape == (1, 4, 4)
    assert logits.shape == (1, 4, 4)
    assert torch.equal(targets, example_targets)


def test_transformer_hypernetwork_builds_and_runs_with_canonical_wrapper():
    """The config-facing transformer hypernetwork should remain constructible end to end."""
    model = build_model({"embedding_dim": 8})
    batch = make_batch()

    assert isinstance(model.hypermodel.hypernetwork, Transformer)

    logits, targets = model(batch)

    assert logits.shape == (1, 4, 4)
    assert targets.shape == (1, 4, 4)


def test_forward_runs_with_hierarchical_pooling():
    """Hierarchical pooling should preserve the end-to-end HyperModelLightning shapes."""
    model = build_model(
        {"embedding_dim": 8},
        {
            "pooling": "hierarchical",
            "interaction_num_heads": 1,
            "segment_interaction_layers": 1,
            "example_interaction_layers": 1,
        },
    )
    batch = make_batch()

    task_features, example_inputs, example_targets = model.prepare_inputs(batch)
    logits, targets = model(batch)

    assert task_features.shape == (1, 24, 8)
    assert example_inputs.shape == (1, 4, 4, 8)
    assert example_targets.shape == (1, 4, 4)
    assert logits.shape == (1, 4, 4)
    assert torch.equal(targets, example_targets)


def test_attention_pooling_remains_the_default_hyper_head_behavior():
    """Omitting pooling config should preserve the legacy single-stage attention pooler."""
    default_model = build_model({"embedding_dim": 8})
    explicit_model = build_model({"embedding_dim": 8}, {"pooling": "attention"})

    assert isinstance(default_model.hypermodel.hyper_pooling, AttentionPooler)
    assert isinstance(explicit_model.hypermodel.hyper_pooling, AttentionPooler)
