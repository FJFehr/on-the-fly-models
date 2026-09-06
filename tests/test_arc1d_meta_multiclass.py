"""Contract tests for the meta-learning multiclass ARC1D datamodule."""

from data_modules.arc1d_meta_multiclass import Arc1dMetaMulticlassDataModule


def _make_dm(data_dir: str, **overrides) -> Arc1dMetaMulticlassDataModule:
    kwargs = {
        "data_dir": data_dir,
        "batch_size": 1,
        "task_categories": ["1d_move_1p"],
    }
    kwargs.update(overrides)
    return Arc1dMetaMulticlassDataModule(**kwargs)


def test_variants_per_base_task_defaults_to_no_subsampling(make_arc1d_dataset_dict):
    data_dir = make_arc1d_dataset_dict(
        n_base_tasks=4, n_variants_per_base_task=20, categories=["1d_move_1p"]
    )
    dm = _make_dm(data_dir)
    dm.setup()
    assert len(dm.train_dataset) == 80  # 4 base tasks x 20 variants each
    assert len(dm.val_dataset) == 2
    assert len(dm.test_dataset) == 2


def test_level_1_selects_exactly_the_original_per_base_task(make_arc1d_dataset_dict):
    data_dir = make_arc1d_dataset_dict(
        n_base_tasks=5, n_variants_per_base_task=10, categories=["1d_move_1p"]
    )
    dm = _make_dm(data_dir, variants_per_base_task=1)
    dm.setup()

    assert len(dm.train_dataset) == 5  # 1 per base task
    for i in range(len(dm.train_dataset)):
        task_id = dm.train_dataset[i]["task_id"]
        assert task_id % 10000 == 0, f"expected aug_index 0 (original), got task_id={task_id}"


def test_variants_per_base_task_caps_train_only(make_arc1d_dataset_dict):
    data_dir = make_arc1d_dataset_dict(
        n_base_tasks=4, n_variants_per_base_task=20, categories=["1d_move_1p", "1d_flip"]
    )
    dm = _make_dm(
        data_dir, task_categories=["1d_move_1p", "1d_flip"], variants_per_base_task=5
    )
    dm.setup()

    assert len(dm.train_dataset) == 40  # 5 per base task x 4 base tasks x 2 categories
    assert len(dm.val_dataset) == 2  # untouched
    assert len(dm.test_dataset) == 2  # untouched


def test_variants_per_base_task_keeps_all_when_fewer_available(make_arc1d_dataset_dict):
    data_dir = make_arc1d_dataset_dict(
        n_base_tasks=3, n_variants_per_base_task=4, categories=["1d_move_1p"]
    )
    dm = _make_dm(data_dir, variants_per_base_task=100)
    dm.setup()
    assert len(dm.train_dataset) == 12  # 3 base tasks x 4 available variants each


def test_levels_are_nested_across_increasing_variants_per_base_task(make_arc1d_dataset_dict):
    """Level K's selected task_ids must be a strict subset of level K+1's, for the
    same data_seed - this is the core cumulative/nested guarantee."""
    data_dir = make_arc1d_dataset_dict(
        n_base_tasks=6, n_variants_per_base_task=30, categories=["1d_move_1p"]
    )

    def selected_ids(variants_per_base_task: int) -> set[int]:
        dm = _make_dm(data_dir, variants_per_base_task=variants_per_base_task, data_seed=7)
        dm.setup()
        return {dm.train_dataset[i]["task_id"] for i in range(len(dm.train_dataset))}

    levels = [1, 2, 3, 5, 10, 20]
    previous = selected_ids(levels[0])
    for level in levels[1:]:
        current = selected_ids(level)
        assert previous <= current, f"level {level}'s selection is not a superset of the previous"
        assert len(current) == level * 6  # 6 base tasks
        previous = current


def test_variants_per_base_task_deterministic_given_same_data_seed(make_arc1d_dataset_dict):
    data_dir = make_arc1d_dataset_dict(
        n_base_tasks=5, n_variants_per_base_task=20, categories=["1d_move_1p"]
    )

    def sampled_ids(data_seed: int) -> list[int]:
        dm = _make_dm(data_dir, variants_per_base_task=5, data_seed=data_seed)
        dm.setup()
        return sorted(dm.train_dataset[i]["task_id"] for i in range(len(dm.train_dataset)))

    assert sampled_ids(data_seed=7) == sampled_ids(data_seed=7)


def test_variants_per_base_task_different_data_seed_gives_different_subsample(
    make_arc1d_dataset_dict,
):
    data_dir = make_arc1d_dataset_dict(
        n_base_tasks=5, n_variants_per_base_task=20, categories=["1d_move_1p"]
    )

    def sampled_ids(data_seed: int) -> list[int]:
        dm = _make_dm(data_dir, variants_per_base_task=5, data_seed=data_seed)
        dm.setup()
        return sorted(dm.train_dataset[i]["task_id"] for i in range(len(dm.train_dataset)))

    assert sampled_ids(data_seed=1) != sampled_ids(data_seed=2)


def test_variants_per_base_task_independent_of_training_seed(make_arc1d_dataset_dict):
    """The training `seed` (pl.seed_everything) must not affect which rows get
    sampled - only `data_seed` should. This is what lets multiple training seeds
    at a fixed data level share identical training data."""
    import lightning as pl

    data_dir = make_arc1d_dataset_dict(
        n_base_tasks=5, n_variants_per_base_task=20, categories=["1d_move_1p"]
    )

    def sampled_ids(training_seed: int) -> list[int]:
        pl.seed_everything(training_seed)
        dm = _make_dm(data_dir, variants_per_base_task=5, data_seed=42)
        dm.setup()
        return sorted(dm.train_dataset[i]["task_id"] for i in range(len(dm.train_dataset)))

    assert sampled_ids(training_seed=1) == sampled_ids(training_seed=2)
