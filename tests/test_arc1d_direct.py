"""Contract tests for Arc1dDirectDataModule's stratified/nested data-reduction sampling."""

from data_modules.arc1d_direct import Arc1dDirectDataModule


def _make_dm(data_dir: str, **overrides) -> Arc1dDirectDataModule:
    kwargs = {
        "data_dir": data_dir,
        "batch_size": 1,
        "prediction_task": "multiclass",
        "task_categories": ["1d_move_1p"],
    }
    kwargs.update(overrides)
    return Arc1dDirectDataModule(**kwargs)


def test_variants_per_base_task_defaults_to_no_subsampling(make_arc1d_dataset_dict):
    data_dir = make_arc1d_dataset_dict(
        n_base_tasks=4, n_variants_per_base_task=20, categories=["1d_move_1p"]
    )
    dm = _make_dm(data_dir)
    dm.setup()
    assert len(dm.train_dataset) == 4 * 20 * 3  # 4 base tasks x 20 variants x 3 support pairs
    assert len(dm.val_dataset) == 2  # query-only, untouched
    assert len(dm.test_dataset) == 2


def test_level_1_selects_exactly_the_original_per_base_task(make_arc1d_dataset_dict):
    data_dir = make_arc1d_dataset_dict(
        n_base_tasks=5, n_variants_per_base_task=10, categories=["1d_move_1p"]
    )
    dm = _make_dm(data_dir, variants_per_base_task=1)
    dm.setup()

    assert len(dm.train_dataset) == 5 * 3  # 1 row per base task x 3 support pairs
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

    assert len(dm.train_dataset) == 5 * 4 * 2 * 3  # levels x base tasks x categories x pairs
    assert len(dm.val_dataset) == 2  # untouched
    assert len(dm.test_dataset) == 2  # untouched


def test_variants_per_base_task_keeps_all_when_fewer_available(make_arc1d_dataset_dict):
    data_dir = make_arc1d_dataset_dict(
        n_base_tasks=3, n_variants_per_base_task=4, categories=["1d_move_1p"]
    )
    dm = _make_dm(data_dir, variants_per_base_task=100)
    dm.setup()
    assert len(dm.train_dataset) == 3 * 4 * 3  # 3 base tasks x 4 available variants x pairs


def test_levels_are_nested_across_increasing_variants_per_base_task(make_arc1d_dataset_dict):
    """Level K's selected base-task rows must be a strict subset of level K+1's, for the
    same data_seed - this is the core cumulative/nested guarantee. Checked via the set of
    distinct row task_ids present in the flattened train_dataset (each row contributes 3
    flat items sharing the same task_id)."""
    data_dir = make_arc1d_dataset_dict(
        n_base_tasks=6, n_variants_per_base_task=30, categories=["1d_move_1p"]
    )

    def selected_row_ids(variants_per_base_task: int) -> set[int]:
        dm = _make_dm(data_dir, variants_per_base_task=variants_per_base_task, data_seed=7)
        dm.setup()
        return {dm.train_dataset[i]["task_id"] for i in range(len(dm.train_dataset))}

    levels = [1, 2, 3, 5, 10, 20]
    previous = selected_row_ids(levels[0])
    for level in levels[1:]:
        current = selected_row_ids(level)
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
    sampled - only `data_seed` should."""
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


# ---------------------------------------------------------------------------
# base_tasks_per_category -- one level up from variants_per_base_task: drops
# whole base tasks (not variants within one) to reach data levels below the
# "every base task's original" floor. Same contract shape as the tests above.
# ---------------------------------------------------------------------------


def test_base_tasks_per_category_defaults_to_no_subsampling(make_arc1d_dataset_dict):
    data_dir = make_arc1d_dataset_dict(
        n_base_tasks=4, n_variants_per_base_task=20, categories=["1d_move_1p"]
    )
    dm = _make_dm(data_dir)
    dm.setup()
    assert len(dm.train_dataset) == 4 * 20 * 3
    assert len(dm.val_dataset) == 2
    assert len(dm.test_dataset) == 2


def test_base_tasks_per_category_selects_exactly_n_base_tasks(make_arc1d_dataset_dict):
    data_dir = make_arc1d_dataset_dict(
        n_base_tasks=6, n_variants_per_base_task=5, categories=["1d_move_1p"]
    )
    dm = _make_dm(data_dir, base_tasks_per_category=3)
    dm.setup()

    # No "always keep the original" rule at this axis -- a kept base task keeps
    # ALL of its variants (variants_per_base_task is untouched/None here).
    assert len(dm.train_dataset) == 3 * 5 * 3  # 3 base tasks x 5 variants x 3 pairs
    base_task_ids = {dm.train_dataset[i]["task_id"] // 10000 for i in range(len(dm.train_dataset))}
    assert len(base_task_ids) == 3


def test_base_tasks_per_category_caps_train_only(make_arc1d_dataset_dict):
    data_dir = make_arc1d_dataset_dict(
        n_base_tasks=6, n_variants_per_base_task=5, categories=["1d_move_1p", "1d_flip"]
    )
    dm = _make_dm(
        data_dir, task_categories=["1d_move_1p", "1d_flip"], base_tasks_per_category=2
    )
    dm.setup()

    assert len(dm.train_dataset) == 2 * 5 * 2 * 3  # base tasks x variants x categories x pairs
    assert len(dm.val_dataset) == 2  # untouched
    assert len(dm.test_dataset) == 2  # untouched


def test_base_tasks_per_category_keeps_all_when_fewer_available(make_arc1d_dataset_dict):
    data_dir = make_arc1d_dataset_dict(
        n_base_tasks=3, n_variants_per_base_task=4, categories=["1d_move_1p"]
    )
    dm = _make_dm(data_dir, base_tasks_per_category=100)
    dm.setup()
    assert len(dm.train_dataset) == 3 * 4 * 3


def test_base_tasks_per_category_levels_are_nested(make_arc1d_dataset_dict):
    """Level K's selected base-task-id set must be a strict subset of level K+1's,
    for the same data_seed -- the same nested guarantee as the variants axis, one
    level up. Uses this experiment's actual level set."""
    data_dir = make_arc1d_dataset_dict(
        n_base_tasks=25, n_variants_per_base_task=1, categories=["1d_move_1p"]
    )

    def selected_base_task_ids(base_tasks_per_category: int) -> set[int]:
        dm = _make_dm(data_dir, base_tasks_per_category=base_tasks_per_category, data_seed=7)
        dm.setup()
        return {dm.train_dataset[i]["task_id"] // 10000 for i in range(len(dm.train_dataset))}

    levels = [1, 3, 5, 10, 20]
    previous = selected_base_task_ids(levels[0])
    for level in levels[1:]:
        current = selected_base_task_ids(level)
        assert previous <= current, f"level {level}'s selection is not a superset of the previous"
        assert len(current) == level
        previous = current


def test_base_tasks_per_category_deterministic_given_same_data_seed(make_arc1d_dataset_dict):
    data_dir = make_arc1d_dataset_dict(
        n_base_tasks=10, n_variants_per_base_task=1, categories=["1d_move_1p"]
    )

    def sampled_ids(data_seed: int) -> list[int]:
        dm = _make_dm(data_dir, base_tasks_per_category=5, data_seed=data_seed)
        dm.setup()
        return sorted(dm.train_dataset[i]["task_id"] for i in range(len(dm.train_dataset)))

    assert sampled_ids(data_seed=7) == sampled_ids(data_seed=7)


def test_base_tasks_per_category_different_data_seed_gives_different_subsample(
    make_arc1d_dataset_dict,
):
    data_dir = make_arc1d_dataset_dict(
        n_base_tasks=10, n_variants_per_base_task=1, categories=["1d_move_1p"]
    )

    def sampled_ids(data_seed: int) -> list[int]:
        dm = _make_dm(data_dir, base_tasks_per_category=5, data_seed=data_seed)
        dm.setup()
        return sorted(dm.train_dataset[i]["task_id"] for i in range(len(dm.train_dataset)))

    assert sampled_ids(data_seed=1) != sampled_ids(data_seed=2)


def test_base_tasks_per_category_independent_of_training_seed(make_arc1d_dataset_dict):
    import lightning as pl

    data_dir = make_arc1d_dataset_dict(
        n_base_tasks=10, n_variants_per_base_task=1, categories=["1d_move_1p"]
    )

    def sampled_ids(training_seed: int) -> list[int]:
        pl.seed_everything(training_seed)
        dm = _make_dm(data_dir, base_tasks_per_category=5, data_seed=42)
        dm.setup()
        return sorted(dm.train_dataset[i]["task_id"] for i in range(len(dm.train_dataset)))

    assert sampled_ids(training_seed=1) == sampled_ids(training_seed=2)


def test_base_tasks_per_category_and_variants_per_base_task_are_independently_composable(
    make_arc1d_dataset_dict,
):
    """The two axes must not entangle: which base tasks get kept must not depend
    on the variants_per_base_task value, and vice versa (separate RNG instances)."""
    data_dir = make_arc1d_dataset_dict(
        n_base_tasks=10, n_variants_per_base_task=20, categories=["1d_move_1p"]
    )

    def selected_base_task_ids(variants_per_base_task: int | None) -> set[int]:
        dm = _make_dm(
            data_dir,
            base_tasks_per_category=3,
            variants_per_base_task=variants_per_base_task,
            data_seed=7,
        )
        dm.setup()
        return {dm.train_dataset[i]["task_id"] // 10000 for i in range(len(dm.train_dataset))}

    assert selected_base_task_ids(None) == selected_base_task_ids(1)

    # The pairing this experiment actually uses: exactly one (original) row per
    # selected base task.
    dm = _make_dm(
        data_dir, base_tasks_per_category=3, variants_per_base_task=1, data_seed=7
    )
    dm.setup()
    assert len(dm.train_dataset) == 3 * 1 * 3  # 3 base tasks x 1 variant x 3 pairs
    for i in range(len(dm.train_dataset)):
        assert dm.train_dataset[i]["task_id"] % 10000 == 0
