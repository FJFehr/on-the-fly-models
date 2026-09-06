"""Contract tests for Arc1dDirectDataModule's stratified/nested data-reduction sampling."""

from pathlib import Path

from datasets import Dataset, DatasetDict

from data_modules.arc1d_direct import Arc1dDirectDataModule


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


def make_augmented_task(base_task_id: int, aug_index: int, category: str = "1d_move_1p") -> dict:
    """task_id follows augment_arc_1d.py's convention: orig*10000 + aug_index.

    aug_index == 0 is always the untransformed original.
    """
    task_id = base_task_id * 10000 + aug_index
    return make_task(task_id, base_value=task_id, category=category)


def make_dataset_dict(
    tmp_path: Path,
    *,
    n_base_tasks: int,
    n_variants_per_base_task: int,
    categories: list[str],
) -> str:
    train_tasks = [
        make_augmented_task(base_task_id=cat_idx * 100 + base_id, aug_index=aug_index, category=category)
        for cat_idx, category in enumerate(categories)
        for base_id in range(n_base_tasks)
        for aug_index in range(n_variants_per_base_task)
    ]
    dev_tasks = [make_task(task_id=90000 + i, base_value=i) for i in range(2)]
    test_tasks = [make_task(task_id=91000 + i, base_value=i) for i in range(2)]

    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_list(train_tasks),
            "dev": Dataset.from_list(dev_tasks),
            "test": Dataset.from_list(test_tasks),
        }
    )
    dataset_path = tmp_path / "arc1d_direct"
    dataset_dict.save_to_disk(str(dataset_path))
    return str(dataset_path)


def _make_dm(data_dir: str, **overrides) -> Arc1dDirectDataModule:
    kwargs = {
        "data_dir": data_dir,
        "batch_size": 1,
        "prediction_task": "multiclass",
        "task_categories": ["1d_move_1p"],
    }
    kwargs.update(overrides)
    return Arc1dDirectDataModule(**kwargs)


def test_variants_per_base_task_defaults_to_no_subsampling(tmp_path: Path):
    data_dir = make_dataset_dict(
        tmp_path, n_base_tasks=4, n_variants_per_base_task=20, categories=["1d_move_1p"]
    )
    dm = _make_dm(data_dir)
    dm.setup()
    assert len(dm.train_dataset) == 4 * 20 * 3  # 4 base tasks x 20 variants x 3 support pairs
    assert len(dm.val_dataset) == 2  # query-only, untouched
    assert len(dm.test_dataset) == 2


def test_level_1_selects_exactly_the_original_per_base_task(tmp_path: Path):
    data_dir = make_dataset_dict(
        tmp_path, n_base_tasks=5, n_variants_per_base_task=10, categories=["1d_move_1p"]
    )
    dm = _make_dm(data_dir, variants_per_base_task=1)
    dm.setup()

    assert len(dm.train_dataset) == 5 * 3  # 1 row per base task x 3 support pairs
    for i in range(len(dm.train_dataset)):
        task_id = dm.train_dataset[i]["task_id"]
        assert task_id % 10000 == 0, f"expected aug_index 0 (original), got task_id={task_id}"


def test_variants_per_base_task_caps_train_only(tmp_path: Path):
    data_dir = make_dataset_dict(
        tmp_path, n_base_tasks=4, n_variants_per_base_task=20, categories=["1d_move_1p", "1d_flip"]
    )
    dm = _make_dm(
        data_dir, task_categories=["1d_move_1p", "1d_flip"], variants_per_base_task=5
    )
    dm.setup()

    assert len(dm.train_dataset) == 5 * 4 * 2 * 3  # levels x base tasks x categories x support pairs
    assert len(dm.val_dataset) == 2  # untouched
    assert len(dm.test_dataset) == 2  # untouched


def test_variants_per_base_task_keeps_all_when_fewer_available(tmp_path: Path):
    data_dir = make_dataset_dict(
        tmp_path, n_base_tasks=3, n_variants_per_base_task=4, categories=["1d_move_1p"]
    )
    dm = _make_dm(data_dir, variants_per_base_task=100)
    dm.setup()
    assert len(dm.train_dataset) == 3 * 4 * 3  # 3 base tasks x 4 available variants x support pairs


def test_levels_are_nested_across_increasing_variants_per_base_task(tmp_path: Path):
    """Level K's selected base-task rows must be a strict subset of level K+1's, for the
    same data_seed - this is the core cumulative/nested guarantee. Checked via the set of
    distinct row task_ids present in the flattened train_dataset (each row contributes 3
    flat items sharing the same task_id)."""
    data_dir = make_dataset_dict(
        tmp_path, n_base_tasks=6, n_variants_per_base_task=30, categories=["1d_move_1p"]
    )

    def selected_row_ids(variants_per_base_task: int) -> set[int]:
        dm = _make_dm(data_dir, variants_per_base_task=variants_per_base_task, data_seed=7)
        dm.setup()
        return {dm.train_dataset[i]["task_id"] for i in range(len(dm.train_dataset))}

    levels = [1, 2, 3, 5, 10, 20]
    previous = selected_row_ids(levels[0])
    for level in levels[1:]:
        current = selected_row_ids(level)
        assert previous <= current, f"level {level}'s selection is not a superset of the previous level"
        assert len(current) == level * 6  # 6 base tasks
        previous = current


def test_variants_per_base_task_deterministic_given_same_data_seed(tmp_path: Path):
    data_dir = make_dataset_dict(
        tmp_path, n_base_tasks=5, n_variants_per_base_task=20, categories=["1d_move_1p"]
    )

    def sampled_ids(data_seed: int) -> list[int]:
        dm = _make_dm(data_dir, variants_per_base_task=5, data_seed=data_seed)
        dm.setup()
        return sorted(dm.train_dataset[i]["task_id"] for i in range(len(dm.train_dataset)))

    assert sampled_ids(data_seed=7) == sampled_ids(data_seed=7)


def test_variants_per_base_task_different_data_seed_gives_different_subsample(tmp_path: Path):
    data_dir = make_dataset_dict(
        tmp_path, n_base_tasks=5, n_variants_per_base_task=20, categories=["1d_move_1p"]
    )

    def sampled_ids(data_seed: int) -> list[int]:
        dm = _make_dm(data_dir, variants_per_base_task=5, data_seed=data_seed)
        dm.setup()
        return sorted(dm.train_dataset[i]["task_id"] for i in range(len(dm.train_dataset)))

    assert sampled_ids(data_seed=1) != sampled_ids(data_seed=2)


def test_variants_per_base_task_independent_of_training_seed():
    """The training `seed` (pl.seed_everything) must not affect which rows get
    sampled - only `data_seed` should."""
    import tempfile

    import lightning as pl

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        data_dir = make_dataset_dict(
            tmp_path, n_base_tasks=5, n_variants_per_base_task=20, categories=["1d_move_1p"]
        )

        def sampled_ids(training_seed: int) -> list[int]:
            pl.seed_everything(training_seed)
            dm = _make_dm(data_dir, variants_per_base_task=5, data_seed=42)
            dm.setup()
            return sorted(dm.train_dataset[i]["task_id"] for i in range(len(dm.train_dataset)))

        assert sampled_ids(training_seed=1) == sampled_ids(training_seed=2)
