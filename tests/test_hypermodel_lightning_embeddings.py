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


def test_shared_embedding_token_ids_match_expected_task_metadata():
    """Protect the serialized metadata contract used by the shared embedder."""
    model = build_model(
        {
            "name": "shared_embeddings",
            "params": {"embedding_dim": 8, "share_input_embeddings": True},
        }
    )
    batch = make_batch()

    task_ids = model.build_task_context_token_ids(
        batch["support_inputs"],
        batch["support_outputs"],
        batch["query_input"],
    )
    target_ids = model.build_target_token_ids(
        batch["support_inputs"],
        batch["query_input"],
    )

    _, _, task_example_ids, task_role_ids, task_is_query_ids = task_ids
    _, _, target_example_ids, target_role_ids, target_is_query_ids = target_ids

    assert task_example_ids.shape == (1, 28)
    assert task_example_ids[0].tolist() == [0] * 8 + [1] * 8 + [2] * 8 + [3] * 4
    assert task_role_ids[0].tolist() == (
        [0] * 4 + [1] * 4 + [0] * 4 + [1] * 4 + [0] * 4 + [1] * 4 + [0] * 4
    )
    assert task_is_query_ids[0].tolist() == [0] * 24 + [1] * 4

    assert target_example_ids.shape == (1, 4, 4)
    assert target_example_ids[0, :, 0].tolist() == [0, 1, 2, 3]
    assert torch.equal(target_role_ids, torch.zeros_like(target_role_ids))
    assert target_is_query_ids[0, :, 0].tolist() == [0, 0, 0, 1]


def test_shared_embeddings_forward_runs_with_expected_shapes():
    """Smoke-test the shared embedding path through the generic hypermodel."""
    model = build_model(
        {
            "name": "shared_embeddings",
            "params": {"embedding_dim": 8, "share_input_embeddings": True},
        }
    )
    batch = make_batch()

    task_features, example_inputs, example_targets = model.prepare_inputs(batch)
    logits, targets = model(batch)

    assert task_features.shape == (1, 28, 8)
    assert example_inputs.shape == (1, 4, 4, 8)
    assert example_targets.shape == (1, 4, 4)
    assert logits.shape == (1, 4, 4)
    assert torch.equal(targets, example_targets)
