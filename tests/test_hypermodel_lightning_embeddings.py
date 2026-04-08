"""Focused tests for shared-embedding HyperModelLightning inputs."""

import torch

from models.hypermodel_lightning import HyperModelLightning


def build_model(task_encoding: dict | None = None) -> HyperModelLightning:
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
