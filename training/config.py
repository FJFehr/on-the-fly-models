"""Config loading, normalization, and output path helpers.

This module owns the full config lifecycle:
- loading YAML with `_base_` inheritance
- flattening it into the keyword arguments the model and data module receive
- creating and saving run output directories
"""

import os
from pathlib import Path

from omegaconf import OmegaConf


def ensure_output_path(output_path: str) -> None:
    """Create the run output directory if it does not already exist."""
    os.makedirs(output_path, exist_ok=True)


def save_resolved_config(cfg) -> None:
    """Persist the resolved config next to other run artefacts.

    The saved config is the fully resolved version after `_base_` merging and
    interpolation, so the run directory remains self-documenting.
    """
    ensure_output_path(cfg.output_path)
    OmegaConf.save(cfg, os.path.join(cfg.output_path, "config.yaml"))


def build_runtime_config_dict(cfg) -> dict:
    """Resolve the config into the flat keyword arguments passed to the model and data module.

    Keys under an optional `training:` section are lifted to the top level. Model and data
    module constructors take the keys they need and ignore the rest.
    """
    runtime_cfg = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(runtime_cfg, dict):
        raise ValueError("Resolved config must be a dictionary.")
    training_cfg = runtime_cfg.pop("training", None)
    if isinstance(training_cfg, dict):
        runtime_cfg.update(training_cfg)
    return runtime_cfg


def load_config(config_path: str, overrides: list[str] | None = None):
    """Load a config file, apply `_base_` and command-line overrides, and resolve it.

    This is the single entry point for config loading so every run benefits
    from the same merge and normalization behavior.
    """
    cfg = OmegaConf.load(config_path)
    if "_base_" in cfg:
        # `_base_` keeps experiment files small by allowing per-run overrides
        # on top of a shared base config.
        base_cfg = OmegaConf.load(cfg._base_)
        del cfg["_base_"]
        cfg = OmegaConf.merge(base_cfg, cfg)

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    OmegaConf.resolve(cfg)
    return cfg


def locate_saved_run(cfg, config_path: str) -> None:
    """Point output_path at the folder of a saved run's config.yaml.

    A finished run's config.yaml records where the run was written, but the folder may have
    been moved, renamed or copied from another machine since. When `config_path` is such a
    file (a config.yaml next to a results.txt), its own folder is the run's folder.
    """
    folder = Path(config_path).resolve().parent
    if Path(config_path).name == "config.yaml" and (folder / "results.txt").exists():
        cfg.output_path = str(folder)
