"""Config loading, normalization, and output path helpers.

This module owns the full config lifecycle:
- loading YAML with `_base_` inheritance
- normalizing grouped/aliased sections into the runtime shape
- creating and saving run output directories
"""

import os

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


def apply_grouped_config_aliases(cfg) -> None:
    """Translate grouped experiment config sections into the runtime shape.

    Some experiment files still use older or grouped naming conventions.
    This helper normalizes them before runtime flattening happens.
    """
    if "name" in cfg and "model" not in cfg:
        cfg["model"] = cfg["name"]


def build_runtime_config_dict(cfg) -> dict:
    """Flatten grouped config sections into the kwargs expected by the runtime.

    The training runtime passes one large `**kwargs` mapping into the selected
    model and datamodule. This helper converts the human-friendly YAML shape
    into that runtime shape while preserving backward compatibility.
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    if not isinstance(cfg_dict, dict):
        msg = "Resolved config must be a dictionary."
        raise ValueError(msg)

    runtime_cfg = dict(cfg_dict)

    # Lift the grouped `training:` section to top-level runtime kwargs so model
    # and datamodule constructors can continue receiving a flat config shape.
    training_cfg = runtime_cfg.pop("training", None)
    if isinstance(training_cfg, dict):
        runtime_cfg.update(training_cfg)

    if runtime_cfg.get("model") not in {"hyper_model", "binary_hyper_model"}:
        # Older meta-model paths expect some grouped config values to be lifted
        # into flat runtime keys. The simplified hypermodel path keeps its
        # structured `hyper_model` / `target_model` sections intact.
        hyper_model_cfg = runtime_cfg.pop("hyper_model", None)
        if isinstance(hyper_model_cfg, dict):
            runtime_cfg.update(hyper_model_cfg)

        target_model_cfg = runtime_cfg.pop("target_model", None)
        if isinstance(target_model_cfg, dict):
            rnn_cfg = target_model_cfg.get("rnn")
            if isinstance(rnn_cfg, dict):
                runtime_cfg["target_rnn_hidden_dim"] = rnn_cfg["hidden_dim"]
                runtime_cfg["target_rnn_bidirectional"] = rnn_cfg["bidirectional"]
                runtime_cfg["target_rnn_num_layers"] = rnn_cfg["num_layers"]
            cnn_cfg = target_model_cfg.get("cnn")
            if isinstance(cnn_cfg, dict):
                runtime_cfg["target_cnn_hidden_channels"] = cnn_cfg["hidden_channels"]
                runtime_cfg["target_cnn_kernel_size"] = cnn_cfg.get("kernel_size", 3)
                runtime_cfg["target_cnn_num_layers"] = cnn_cfg.get("num_layers", 1)
                runtime_cfg["target_cnn_use_skip_connections"] = cnn_cfg.get(
                    "use_skip_connections", False
                )

    return runtime_cfg


def load_config(config_path: str, overrides: list[str] | None = None):
    """Load a config file, apply `_base_`, aliases, and interpolation resolution.

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

    apply_grouped_config_aliases(cfg)
    OmegaConf.resolve(cfg)
    return cfg
