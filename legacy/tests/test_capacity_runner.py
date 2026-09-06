"""Focused tests for seeded capacity runner scheduling helpers."""

from pathlib import Path

from scripts.run_arc1d_capacity import (
    ScheduledRun,
    build_process_env,
    build_scheduled_runs,
    build_train_command,
    discover_configs,
    load_config,
    load_experiment_name,
    parse_csv_ints,
    resolve_max_parallel,
    resolve_output_root,
)


def test_load_experiment_name_resolves_base_config_interpolation(tmp_path: Path):
    """Runner discovery should read the resolved experiment_name from config inheritance."""
    base_path = tmp_path / "base.yaml"
    config_path = tmp_path / "child.yaml"
    base_path.write_text(
        "\n".join(
            [
                "task: move_1p",
                "model: rnn",
                "experiment_name: ${task}_${model}",
            ]
        )
    )
    config_path.write_text(f"_base_: {base_path}\n")

    assert load_experiment_name(config_path) == "move_1p_rnn"


def test_build_scheduled_runs_expands_each_config_per_seed(tmp_path: Path):
    """One config and three seeds should become three scheduled runs."""
    config_path = tmp_path / "run.yaml"
    config_path.write_text("experiment_name: mc_1d_move_1p_rnn\n")

    runs = build_scheduled_runs([config_path], [42, 43, 44])

    assert [run.run_name for run in runs] == [
        "mc_1d_move_1p_rnn_seed_42",
        "mc_1d_move_1p_rnn_seed_43",
        "mc_1d_move_1p_rnn_seed_44",
    ]


def test_build_train_command_includes_seed_and_unique_experiment_name():
    """Seeded runs should inject both the seed and the run-specific experiment name."""
    run = ScheduledRun(
        config_path=Path("configs/experiments/arc1d_capacity_multiclass/1d_move_1p/rnn.yaml"),
        seed=43,
        base_experiment_name="mc_1d_move_1p_rnn",
    )

    command = build_train_command(run)

    assert command[-2:] == ["seed=43", "experiment_name=mc_1d_move_1p_rnn_seed_43"]


def test_build_process_env_uses_single_gpu_or_cpu_fallback():
    """GPU mode exposes one GPU id; CPU mode hides all GPUs."""
    gpu_env = build_process_env(2)
    cpu_env = build_process_env(None)

    assert gpu_env["CUDA_VISIBLE_DEVICES"] == "2"
    assert cpu_env["CUDA_VISIBLE_DEVICES"] == ""


def test_resolve_max_parallel_defaults_to_gpu_count():
    """GPU mode should default concurrency to the number of visible GPUs."""
    assert resolve_max_parallel(None, [0, 1, 2]) == 3
    assert resolve_max_parallel(2, [0, 1, 2]) == 2
    assert resolve_max_parallel(8, [0, 1, 2]) == 3


def test_parse_csv_ints_parses_seeds_and_gpus():
    """Comma-separated integer parsing should preserve order."""
    assert parse_csv_ints("42,43,44", "--seeds") == [42, 43, 44]
    assert parse_csv_ints("0,2", "--gpus") == [0, 2]


def test_augmented_capacity_tiers_are_discoverable_and_named():
    """Each augmented tier should expose the full 18-task by 3-model sweep."""
    tiers = ["small", "medium", "large"]

    for tier in tiers:
        root = Path(f"configs/experiments/arc1d_capacity_augmented_{tier}")
        configs = discover_configs(root)
        names = [load_experiment_name(config_path) for config_path in configs]

        assert len(configs) == 54
        assert len(set(names)) == 54
        assert all(name.startswith(f"aug_{tier}_") for name in names)
        assert resolve_output_root(configs[0]) == Path(f"outputs/arc1d_capacity_augmented_{tier}")


def test_augmented_capacity_tiers_match_baseline_model_configs():
    """Augmented tiers should differ from baseline tiers only by dataset and naming."""
    tiers = ["small", "medium", "large"]

    for tier in tiers:
        baseline_root = Path(f"configs/experiments/arc1d_capacity_{tier}")
        augmented_root = Path(f"configs/experiments/arc1d_capacity_augmented_{tier}")
        baseline_configs = {
            path.relative_to(baseline_root): path for path in discover_configs(baseline_root)
        }
        augmented_configs = {
            path.relative_to(augmented_root): path for path in discover_configs(augmented_root)
        }

        assert augmented_configs.keys() == baseline_configs.keys()

        for relative_path, baseline_path in baseline_configs.items():
            baseline_cfg = load_config(baseline_path)
            augmented_cfg = load_config(augmented_configs[relative_path])

            assert augmented_cfg.data_dir == "data/arc_1d_augmented"
            assert augmented_cfg.task_categories == baseline_cfg.task_categories
            assert augmented_cfg.backbone_model == baseline_cfg.backbone_model
            assert augmented_cfg.max_steps == baseline_cfg.max_steps
            assert augmented_cfg.batch_size == baseline_cfg.batch_size
            assert augmented_cfg.non_background_loss_weight == 2.0
