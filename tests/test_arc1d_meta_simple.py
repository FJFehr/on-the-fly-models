"""Contract tests for the binary arc_1d_simple meta datamodule."""

from pathlib import Path

from datasets import Dataset, DatasetDict

from data_modules.arc1d_meta_simple import Arc1dMetaSimpleDataModule


def make_task(task_id: int, base_value: int, category: str = "1d_move_1p") -> dict:
    return {
        "task_category": category,
        "task_id": task_id,
        "sequence_length": 4,
        "support_inputs": [[(base_value + offset) % 2] * 4 for offset in range(3)],
        "support_outputs": [[(base_value + 1 + offset) % 2] * 4 for offset in range(3)],
        "query_input": [(base_value + 1) % 2] * 4,
        "query_output": [base_value % 2] * 4,
    }


def test_binary_meta_datamodule_preserves_whole_tasks(tmp_path: Path):
    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_list([make_task(1, 0), make_task(2, 1, category="1d_move_2p")]),
            "dev": Dataset.from_list([make_task(3, 0)]),
            "test": Dataset.from_list([make_task(4, 1)]),
        }
    )
    dataset_path = tmp_path / "arc1d_meta_simple"
    dataset_dict.save_to_disk(str(dataset_path))

    dm = Arc1dMetaSimpleDataModule(
        data_dir=str(dataset_path),
        batch_size=1,
        task_categories=["1d_move_1p"],
    )
    dm.setup()

    sample = dm.train_dataset[0]
    assert len(dm.train_dataset) == 1
    assert len(dm.val_dataset) == 1
    assert len(dm.test_dataset) == 1
    assert sample["support_inputs"].shape == (3, 4)
    assert sample["support_outputs"].shape == (3, 4)
    assert sample["query_input"].shape == (4,)
    assert sample["query_output"].shape == (4,)
    assert sample["support_inputs"].dtype == sample["query_input"].dtype
    assert sample["task_category"] == "1d_move_1p"
    assert sample["task_id"] == 1


def test_binary_meta_datamodule_accepts_val_alias_and_mixed_categories(tmp_path: Path):
    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_list(
                [make_task(1, 0, "1d_move_1p"), make_task(2, 1, "1d_denoising_1c")]
            ),
            "dev": Dataset.from_list([make_task(3, 0, "1d_denoising_1c")]),
            "test": Dataset.from_list([make_task(4, 1, "1d_move_1p")]),
        }
    )
    dataset_path = tmp_path / "arc1d_meta_simple"
    dataset_dict.save_to_disk(str(dataset_path))

    dm = Arc1dMetaSimpleDataModule(
        data_dir=str(dataset_path),
        batch_size=2,
        task_categories=["1d_move_1p", "1d_denoising_1c"],
        val_split="val",
    )
    dm.setup()

    assert len(dm.train_dataset) == 2
    assert len(dm.val_dataset) == 1
    assert len(dm.test_dataset) == 1
    assert {dm.train_dataset[index]["task_category"] for index in range(2)} == {
        "1d_move_1p",
        "1d_denoising_1c",
    }
