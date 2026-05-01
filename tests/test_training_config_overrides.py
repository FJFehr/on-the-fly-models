"""Focused tests for config override support in the training entrypoint."""

from pathlib import Path

from training.config import load_config


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
