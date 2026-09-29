"""Focused tests for config override support in the training entrypoint."""

from pathlib import Path

from training.config import load_config, locate_saved_run


def test_load_config_applies_dotlist_overrides_before_resolution(tmp_path: Path):
    """Overriding experiment_name should also update interpolated output_path."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "seed: 42",
                "output_dir: outputs",
                "project_name: proj",
                "experiment_name: baseline",
                "output_path: ${output_dir}/${project_name}/${experiment_name}",
            ]
        )
    )

    cfg = load_config(
        str(config_path),
        ["seed=7", "experiment_name=baseline_seed_7"],
    )

    assert cfg.seed == 7
    assert cfg.experiment_name == "baseline_seed_7"
    assert cfg.output_path == "outputs/proj/baseline_seed_7"


def test_a_moved_run_folder_is_found_from_its_own_config(tmp_path: Path):
    """A saved run's config.yaml still names the folder it was first written to; evaluation
    scripts must use the folder the file is in now (runs get moved, renamed or fetched)."""
    run_dir = tmp_path / "renamed_run"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text("output_path: outputs/original_name/run\n")
    (run_dir / "results.txt").write_text("done\n")
    cfg = load_config(str(run_dir / "config.yaml"))
    locate_saved_run(cfg, str(run_dir / "config.yaml"))
    assert cfg.output_path == str(run_dir.resolve())

    # An experiment config (not a saved run) keeps its own output_path.
    experiment_cfg = tmp_path / "experiment.yaml"
    experiment_cfg.write_text("output_path: outputs/exp/run\n")
    cfg = load_config(str(experiment_cfg))
    locate_saved_run(cfg, str(experiment_cfg))
    assert cfg.output_path == "outputs/exp/run"
