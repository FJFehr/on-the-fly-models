"""Test configuration and shared fixtures.

The project is intentionally lightweight and does not install itself as a
package in every workflow. Keep the repository root on ``sys.path`` so tests
can import project modules directly without repeating path setup in each file.

Fixtures below were promoted out of individual test files where the same
setup code had been copy-pasted across several of them (arc1d datamodule
tests; hypermodel-adjacent tests) -- consolidating here removes the
duplication without changing what any existing test asserts.
"""

import sys
from pathlib import Path

import pytest
import torch
from datasets import Dataset, DatasetDict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# ARC1D task-level dataset builders, shared by the arc1d_direct /
# arc1d_meta_multiclass datamodule tests (and reusable for any test that
# needs a real on-disk DatasetDict without touching the repo's real data/).
# ---------------------------------------------------------------------------


def make_arc1d_task(task_id: int, base_value: int, category: str = "1d_move_1p") -> dict:
    # base_value is often a large, distinguishing value (e.g. a task_id used as its own
    # seed) -- fine for tests that only check row counts/task_ids, but the 6 consecutive
    # values below (base_value .. base_value+5) must stay valid ARC-1D pixel colours
    # (0-9, 10 reserved for padding) for anything that runs a real forward pass. Mod down
    # to [0, 3] so the whole run of +5 stays under 10, independent of how large base_value
    # itself is.
    value = base_value % 4
    return {
        "task_category": category,
        "task_id": task_id,
        "sequence_length": 4,
        "support_inputs": [[value + offset] * 4 for offset in range(3)],
        "support_outputs": [[value + 1 + offset] * 4 for offset in range(3)],
        "query_input": [value + 4] * 4,
        "query_output": [value + 5] * 4,
    }


def make_arc1d_augmented_task(
    base_task_id: int, aug_index: int, category: str = "1d_move_1p"
) -> dict:
    """task_id follows augment_arc_1d.py's convention: orig*10000 + aug_index.

    aug_index == 0 is always the untransformed original.
    """
    task_id = base_task_id * 10000 + aug_index
    return make_arc1d_task(task_id, base_value=task_id, category=category)


@pytest.fixture
def make_arc1d_dataset_dict(tmp_path: Path):
    """Factory: build a tiny synthetic ARC1D DatasetDict, save it under tmp_path, and
    return the directory path -- the same save/load round trip a real datamodule uses,
    without ever touching the repo's real data/."""

    def _make(*, n_base_tasks: int, n_variants_per_base_task: int, categories: list[str]) -> str:
        train_tasks = [
            make_arc1d_augmented_task(
                base_task_id=cat_idx * 100 + base_id, aug_index=aug_index, category=category
            )
            for cat_idx, category in enumerate(categories)
            for base_id in range(n_base_tasks)
            for aug_index in range(n_variants_per_base_task)
        ]
        dev_tasks = [make_arc1d_task(task_id=90000 + i, base_value=i) for i in range(2)]
        test_tasks = [make_arc1d_task(task_id=91000 + i, base_value=i) for i in range(2)]

        dataset_dict = DatasetDict(
            {
                "train": Dataset.from_list(train_tasks),
                "dev": Dataset.from_list(dev_tasks),
                "test": Dataset.from_list(test_tasks),
            }
        )
        dataset_path = tmp_path / "dataset"
        dataset_dict.save_to_disk(str(dataset_path))
        return str(dataset_path)

    return _make


# ---------------------------------------------------------------------------
# Tiny real HyperModelLightning + a matching batch, shared by every
# hypermodel-adjacent test (embeddings, variational, embedding-cluster
# logging) that needs one.
# ---------------------------------------------------------------------------


@pytest.fixture
def build_hypermodel():
    """Factory for a tiny real HyperModelLightning: transformer hyper_model, rnn
    target, input_dim=4, task_encoding defaults to embedding_dim=8 (every existing
    caller across the suite uses this exact value; override via the task_encoding
    kwarg if a future test needs something else). hyper_head and any other
    HyperModelLightning constructor kwarg pass straight through."""

    def _build(hyper_head: dict | None = None, task_encoding: dict | None = None, **kwargs):
        from models.hypermodel_lightning import HyperModelLightning

        return HyperModelLightning(
            hyper_model={
                "name": "transformer",
                "params": {"hidden_dim": 16, "num_layers": 1, "num_heads": 1, "output_dim": 8},
            },
            target_model={
                "name": "rnn",
                "params": {"hidden_dim": 8, "num_layers": 1, "bidirectional": True},
            },
            hyper_head=hyper_head,
            task_encoding=task_encoding if task_encoding is not None else {"embedding_dim": 8},
            input_dim=4,
            **kwargs,
        )

    return _build


@pytest.fixture
def make_hypermodel_batch():
    """Factory for a single-example hypermodel batch: 3 support pairs + 1 query, all
    4-length binary sequences. Pass task_category/task_id to also include those (needed
    by the embedding-collection path); omitted by default since most callers don't need
    them."""

    def _make(task_category: str | None = None, task_id: int | None = None) -> dict:
        batch = {
            "support_inputs": torch.tensor(
                [[[0, 1, 0, 1], [1, 0, 1, 0], [0, 0, 1, 1]]], dtype=torch.float32
            ),
            "support_outputs": torch.tensor(
                [[[1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 0, 1]]], dtype=torch.float32
            ),
            "query_input": torch.tensor([[1, 0, 0, 1]], dtype=torch.float32),
            "query_output": torch.tensor([[0, 1, 1, 0]], dtype=torch.float32),
        }
        if task_category is not None:
            batch["task_category"] = [task_category]
        if task_id is not None:
            batch["task_id"] = torch.tensor([task_id], dtype=torch.long)
        return batch

    return _make
