"""Contract tests for the meta-learning padded multiclass ARC1D datamodule."""

from pathlib import Path

from datasets import Dataset, DatasetDict

from data_modules.arc1d_meta_padded_multiclass import Arc1dMetaPaddedMulticlassDataModule


def make_task(task_id: int, base_value: int, category: str = "1d_move_1p") -> dict:
    return {
        "task_category": category,
        "task_id": task_id,
        "sequence_length": 4,
        "support_inputs": [[base_value + offset] * 4 for offset in range(3)],
        "support_outputs": [[base_value + 1 + offset] * 4 for offset in range(3)],
        "query_input": [base_value + 4] * 4,
        "query_output": [base_value + 5] * 4,
    }


def test_meta_datamodule_uses_task_splits_and_preserves_whole_tasks(tmp_path: Path):
    """Ensure the meta datamodule keeps support/query structure across train/val/test."""

    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_list([make_task(1, 1), make_task(2, 2, category="1d_flip")]),
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
    dm.setup()

    assert len(dm.train_dataset) == 1
    assert len(dm.val_dataset) == 1
    assert len(dm.test_dataset) == 1

    sample = dm.train_dataset[0]
    assert sample["support_inputs"].shape == (3, 4)
    assert sample["support_outputs"].shape == (3, 4)
    assert sample["query_input"].shape == (4,)
    assert sample["query_output"].shape == (4,)
    assert sample["task_category"] == "1d_move_1p"
    assert sample["task_id"] == 1


def test_meta_datamodule_accepts_val_alias_for_dev_split(tmp_path: Path):
    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_list([make_task(1, 1)]),
            "dev": Dataset.from_list([make_task(2, 2)]),
            "test": Dataset.from_list([make_task(3, 3)]),
        }
    )
    dataset_path = tmp_path / "arc1d_meta_padded_multiclass"
    dataset_dict.save_to_disk(str(dataset_path))

    dm = Arc1dMetaPaddedMulticlassDataModule(
        data_dir=str(dataset_path),
        batch_size=1,
        train_split="train",
        val_split="val",
        test_split="test",
    )
    dm.setup()

    assert len(dm.train_dataset) == 1
    assert len(dm.val_dataset) == 1
    assert len(dm.test_dataset) == 1
    assert dm.val_dataset[0]["task_id"] == 2
