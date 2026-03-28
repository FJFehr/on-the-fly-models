"""Contract tests for the simplified ARC1D datamodule.

These tests protect the most important dataset-view decisions: which examples
feed training versus evaluation, how task filtering behaves, and when the
module intentionally reuses the same dataset object across splits.
"""

from pathlib import Path

from datasets import Dataset, DatasetDict

from data_modules.arc1d_simple import Arc1dSimpleDataModule


def make_task(task_id: int, base_value: int) -> dict:
    return {
        "task_category": "cat_a",
        "task_id": task_id,
        "sequence_length": 4,
        "support_inputs": [[base_value + offset] * 4 for offset in range(3)],
        "support_outputs": [[base_value + 10 + offset] * 4 for offset in range(3)],
        "query_input": [base_value + 100] * 4,
        "query_output": [base_value + 110] * 4,
    }


def dataset_sequences(dataset, key: str) -> list[tuple[float, ...]]:
    return [tuple(sample[key].tolist()) for sample in dataset]


def test_arc1d_simple_uses_train_support_for_train_and_train_query_for_eval(tmp_path: Path):
    """Verify the core meta-learning split contract.

    Training must see support pairs only, while validation and test must use
    the held-out query pair for each task. This is the most important guard
    against silently training on evaluation examples.
    """

    train_tasks = [make_task(1, 1), make_task(2, 20)]
    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_list(train_tasks),
            "dev": Dataset.from_list([make_task(3, 200)]),
            "test": Dataset.from_list([make_task(4, 400)]),
        }
    )
    dataset_dict.save_to_disk(str(tmp_path / "arc1d_simple"))

    dm = Arc1dSimpleDataModule(
        data_dir=str(tmp_path / "arc1d_simple"),
        batch_size=2,
    )
    dm.setup()

    assert len(dm.train_dataset) == 6
    assert len(dm.val_dataset) == 2
    assert len(dm.test_dataset) == 2

    train_inputs = set(dataset_sequences(dm.train_dataset, "inputs"))
    train_targets = set(dataset_sequences(dm.train_dataset, "targets"))
    val_inputs = dataset_sequences(dm.val_dataset, "inputs")
    val_targets = dataset_sequences(dm.val_dataset, "targets")
    test_inputs = dataset_sequences(dm.test_dataset, "inputs")
    test_targets = dataset_sequences(dm.test_dataset, "targets")

    expected_support_inputs = {
        (1.0, 1.0, 1.0, 1.0),
        (2.0, 2.0, 2.0, 2.0),
        (3.0, 3.0, 3.0, 3.0),
        (20.0, 20.0, 20.0, 20.0),
        (21.0, 21.0, 21.0, 21.0),
        (22.0, 22.0, 22.0, 22.0),
    }
    expected_support_targets = {
        (11.0, 11.0, 11.0, 11.0),
        (12.0, 12.0, 12.0, 12.0),
        (13.0, 13.0, 13.0, 13.0),
        (30.0, 30.0, 30.0, 30.0),
        (31.0, 31.0, 31.0, 31.0),
        (32.0, 32.0, 32.0, 32.0),
    }
    expected_query_inputs = [
        (101.0, 101.0, 101.0, 101.0),
        (120.0, 120.0, 120.0, 120.0),
    ]
    expected_query_targets = [
        (111.0, 111.0, 111.0, 111.0),
        (130.0, 130.0, 130.0, 130.0),
    ]

    assert train_inputs == expected_support_inputs
    assert train_targets == expected_support_targets
    assert val_inputs == expected_query_inputs
    assert val_targets == expected_query_targets
    assert test_inputs == expected_query_inputs
    assert test_targets == expected_query_targets

    assert (101.0, 101.0, 101.0, 101.0) not in train_inputs
    assert (120.0, 120.0, 120.0, 120.0) not in train_inputs


def test_arc1d_simple_filters_by_category_and_task_id(tmp_path: Path):
    """Verify that configured filters narrow the task set before pair expansion.

    This ensures experiment configs can target a precise subset of ARC tasks
    without leaking unrelated categories or task identifiers into training.
    """

    train_tasks = [
        make_task(1, 1),
        {
            **make_task(2, 20),
            "task_category": "cat_b",
        },
    ]
    dataset_dict = DatasetDict({"train": Dataset.from_list(train_tasks)})
    dataset_dict.save_to_disk(str(tmp_path / "arc1d_simple"))

    dm = Arc1dSimpleDataModule(
        data_dir=str(tmp_path / "arc1d_simple"),
        batch_size=2,
        task_categories=["cat_b"],
        task_ids=[2],
    )
    dm.setup()

    assert len(dm.train_dataset) == 3
    assert len(dm.val_dataset) == 1
    assert {sample["task_category"] for sample in dm.train_dataset} == {"cat_b"}
    assert {sample["task_id"] for sample in dm.train_dataset} == {2}


def test_arc1d_simple_same_train_pairs_reuses_training_dataset(tmp_path: Path):
    """Verify that the explicit reuse mode points eval splits at the train dataset.

    The identity checks matter here: this mode is not "equal contents" but a
    deliberate request to reuse the exact same dataset object for overfit-style
    debugging runs.
    """

    dataset_dict = DatasetDict({"train": Dataset.from_list([make_task(1, 1)])})
    dataset_dict.save_to_disk(str(tmp_path / "arc1d_simple"))

    dm = Arc1dSimpleDataModule(
        data_dir=str(tmp_path / "arc1d_simple"),
        batch_size=1,
        train_source="train_query_pairs",
        val_source="same_train_pairs",
        test_source="same_train_pairs",
    )
    dm.setup()

    assert len(dm.train_dataset) == 1
    assert dm.train_dataset is dm.val_dataset
    assert dm.train_dataset is dm.test_dataset
