"""Contract tests for the padded multiclass ARC1D datamodule."""

from pathlib import Path

import torch
from datasets import Dataset, DatasetDict

from data_modules.arc1d_padded_multiclass import Arc1dPaddedMulticlassDataModule


def make_task(task_id: int, base_value: int) -> dict:
    return {
        "task_category": "cat_a",
        "task_id": task_id,
        "sequence_length": 4,
        "support_inputs": [[base_value + offset] * 4 for offset in range(3)],
        "support_outputs": [[base_value + 1 + offset] * 4 for offset in range(3)],
        "query_input": [base_value + 5] * 4,
        "query_output": [base_value + 6] * 4,
    }


def dataset_sequences(dataset, key: str) -> list[tuple[int, ...]]:
    return [tuple(sample[key].tolist()) for sample in dataset]


def test_arc1d_padded_multiclass_uses_support_for_train_and_query_for_eval(tmp_path: Path):
    """Verify the multiclass datamodule preserves the existing pair contract."""

    train_tasks = [make_task(1, 1), make_task(2, 3)]
    dataset_dict = DatasetDict({"train": Dataset.from_list(train_tasks)})
    dataset_dict.save_to_disk(str(tmp_path / "arc1d_padded_multiclass"))

    dm = Arc1dPaddedMulticlassDataModule(
        data_dir=str(tmp_path / "arc1d_padded_multiclass"),
        batch_size=2,
    )
    dm.setup()

    assert len(dm.train_dataset) == 6
    assert len(dm.val_dataset) == 2
    assert len(dm.test_dataset) == 2
    assert dm.train_dataset[0]["targets"].dtype == torch.int64
    assert dataset_sequences(dm.val_dataset, "targets") == [(7, 7, 7, 7), (9, 9, 9, 9)]
