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
# Tiny real Lightning modules + matching batches, shared by the model tests.
# ---------------------------------------------------------------------------

TINY_TRAINING_CONFIG = {
    "num_classes": 10,
    "padding_idx": 10,
    "optimizer": "AdamW",
    "learning_rate": 1e-3,
    "weight_decay": 0.01,
}


@pytest.fixture
def build_direct():
    """Factory for a tiny DirectLightning. Keyword arguments override the defaults."""

    def _build(task_encoding: dict | None = None, **overrides):
        from lightning_modules.direct import DirectLightning

        config = {
            **TINY_TRAINING_CONFIG,
            "backbone": {"hidden_dim": 4, "num_layers": 3, "num_heads": 1, "dropout": 0.1},
            "task_encoding": task_encoding or {"embedding_dim": 4},
            **overrides,
        }
        return DirectLightning(**config)

    return _build


@pytest.fixture
def build_hypernetwork():
    """Factory for a tiny HypernetworkLightning. Keyword arguments override the defaults."""

    def _build(hyper_head: dict | None = None, **overrides):
        from lightning_modules.hypernetwork import HypernetworkLightning

        config = {
            **TINY_TRAINING_CONFIG,
            "encoder": {"hidden_dim": 8, "num_layers": 1, "num_heads": 1, "output_dim": 8},
            "target": {"hidden_dim": 4, "num_layers": 3, "num_heads": 1, "dropout": 0.1},
            "hyper_head": hyper_head or {"bottleneck_dim": 8},
            "task_encoding": {"embedding_dim": 4},
            **overrides,
        }
        return HypernetworkLightning(**config)

    return _build


@pytest.fixture
def make_hypernetwork_batch():
    """Factory for a batch of `n` tasks: 3 support pairs + 1 query, length-5 sequences whose
    last position is padding (10)."""

    def _make(n: int = 2, task_category: str = "1d_move_1p", task_id: int = 0) -> dict:
        generator = torch.Generator().manual_seed(task_id)

        def sequences(*shape):
            values = torch.randint(0, 10, (*shape, 5), generator=generator)
            values[..., -1] = 10
            return values

        return {
            "support_inputs": sequences(n, 3),
            "support_outputs": sequences(n, 3),
            "query_input": sequences(n),
            "query_output": sequences(n),
            "task_category": [task_category] * n,
            "task_id": torch.arange(task_id, task_id + n),
        }

    return _make
