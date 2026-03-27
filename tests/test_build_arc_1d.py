from collections import Counter
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "build_arc_1d.py"
MODULE_SPEC = spec_from_file_location("build_arc_1d", MODULE_PATH)
assert MODULE_SPEC is not None
assert MODULE_SPEC.loader is not None
build_arc_1d = module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(build_arc_1d)
build_dataset_dict = build_arc_1d.build_dataset_dict
validate_task_examples = build_arc_1d.validate_task_examples


def make_tasks(category: str, num_tasks: int = 50) -> list[dict]:
    tasks = []
    for task_id in range(num_tasks):
        sequence_length = 10 if task_id < num_tasks // 2 else 20
        tasks.append(
            {
                "task_category": category,
                "task_id": task_id,
                "sequence_length": sequence_length,
                "support_inputs": [[0] * sequence_length for _ in range(3)],
                "support_outputs": [[1] * sequence_length for _ in range(3)],
                "query_input": [0] * sequence_length,
                "query_output": [1] * sequence_length,
            }
        )
    return tasks


def get_task_lengths(dataset) -> Counter:
    return Counter(dataset["sequence_length"])


def test_build_dataset_dict_splits_whole_tasks_without_leakage():
    dataset_dict = build_dataset_dict(make_tasks("cat_a"), seed=0)

    assert set(dataset_dict.keys()) == {"train", "dev", "test"}
    assert len(dataset_dict["train"]) == 40
    assert len(dataset_dict["dev"]) == 5
    assert len(dataset_dict["test"]) == 5

    task_membership = {}
    for split_name, dataset in dataset_dict.items():
        task_ids = set(dataset["task_id"])
        assert len(task_ids) == {"train": 40, "dev": 5, "test": 5}[split_name]

        for task_id in task_ids:
            assert task_id not in task_membership
            task_membership[task_id] = split_name

        task_lengths = get_task_lengths(dataset)
        if split_name == "train":
            assert task_lengths == Counter({10: 20, 20: 20})
        else:
            assert task_lengths in (Counter({10: 3, 20: 2}), Counter({10: 2, 20: 3}))


def test_build_dataset_dict_places_holdout_category_in_separate_split():
    tasks = make_tasks("cat_a") + make_tasks("cat_b")
    dataset_dict = build_dataset_dict(tasks, holdout_category="cat_b", seed=0)

    assert set(dataset_dict.keys()) == {"train", "dev", "test", "holdout_test"}
    assert set(dataset_dict["holdout_test"]["task_category"]) == {"cat_b"}
    assert len(set(dataset_dict["holdout_test"]["task_id"])) == 50
    assert len(dataset_dict["holdout_test"]) == 50

    for split_name in ("train", "dev", "test"):
        assert set(dataset_dict[split_name]["task_category"]) == {"cat_a"}


def test_validate_task_examples_rejects_inconsistent_task_lengths():
    train_examples = [
        {"input": [[0] * 10], "output": [[1] * 10]},
        {"input": [[0] * 10], "output": [[1] * 10]},
        {"input": [[0] * 12], "output": [[1] * 12]},
    ]
    test_examples = [{"input": [[0] * 10], "output": [[1] * 10]}]

    with pytest.raises(ValueError, match="inconsistent sequence lengths"):
        validate_task_examples("cat_a", 7, train_examples, test_examples)


def test_build_dataset_dict_preserves_grouped_examples():
    dataset_dict = build_dataset_dict(make_tasks("cat_a"), seed=0)

    for dataset in dataset_dict.values():
        for task in dataset:
            assert len(task["support_inputs"]) == 3
            assert len(task["support_outputs"]) == 3
            assert all(len(seq) == task["sequence_length"] for seq in task["support_inputs"])
            assert all(len(seq) == task["sequence_length"] for seq in task["support_outputs"])
            assert len(task["query_input"]) == task["sequence_length"]
            assert len(task["query_output"]) == task["sequence_length"]
