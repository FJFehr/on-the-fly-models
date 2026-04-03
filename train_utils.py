"""Shared helpers for config-driven training entrypoints."""

import os

import lightning as pl
from lightning.pytorch.callbacks import Callback
from omegaconf import OmegaConf


class StopOnMetricThreshold(Callback):
    """Stop training once a monitored validation metric reaches a threshold."""

    def __init__(self, monitor: str, threshold: float, mode: str = "max") -> None:
        super().__init__()
        if mode not in {"max", "min"}:
            msg = f"Unsupported mode {mode!r}. Expected 'max' or 'min'."
            raise ValueError(msg)

        self.monitor = monitor
        self.threshold = threshold
        self.mode = mode

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return

        metric = trainer.callback_metrics.get(self.monitor)
        if metric is None:
            return

        current_value = float(metric.detach().cpu())
        reached_threshold = (
            current_value >= self.threshold
            if self.mode == "max"
            else current_value <= self.threshold
        )

        if reached_threshold:
            trainer.should_stop = True
            print(
                f"Stopping early because {self.monitor} reached "
                f"{current_value:.6f} (threshold {self.threshold:.6f})."
            )


def write_results_file(
    output_path: str,
    checkpoint_path: str | None,
    val_results: list[dict],
    test_results: list[dict],
) -> None:
    """Write final evaluation results to a plain-text file in the run directory."""
    results_path = os.path.join(output_path, "results.txt")
    lines = ["Final evaluation results"]

    if checkpoint_path:
        lines.append(f"checkpoint: {checkpoint_path}")
    else:
        lines.append("checkpoint: none")

    if val_results:
        lines.append("validation_metrics:")
        for metric_name, metric_value in sorted(val_results[0].items()):
            lines.append(f"{metric_name}: {metric_value}")
    else:
        lines.append("validation_metrics: none")

    if test_results:
        lines.append("test_metrics:")
        for metric_name, metric_value in sorted(test_results[0].items()):
            lines.append(f"{metric_name}: {metric_value}")
    else:
        lines.append("test_metrics: none")

    with open(results_path, "w", encoding="utf-8") as results_file:
        results_file.write("\n".join(lines) + "\n")


def apply_grouped_config_aliases(cfg) -> None:
    """Translate grouped experiment config sections into the runtime shape."""
    if "name" in cfg and "model" not in cfg:
        cfg["model"] = cfg["name"]


def build_runtime_config_dict(cfg) -> dict:
    """Flatten grouped config sections into the kwargs expected by the runtime."""
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    if not isinstance(cfg_dict, dict):
        msg = "Resolved config must be a dictionary."
        raise ValueError(msg)

    runtime_cfg = dict(cfg_dict)

    training_cfg = runtime_cfg.pop("training", None)
    if isinstance(training_cfg, dict):
        runtime_cfg.update(training_cfg)

    if runtime_cfg.get("model") not in {"hyper_model", "binary_hyper_model"}:
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


def load_config(config_path: str):
    """Load a config file, apply `_base_`, aliases, and interpolation resolution."""
    cfg = OmegaConf.load(config_path)
    if "_base_" in cfg:
        base_cfg = OmegaConf.load(cfg._base_)
        del cfg["_base_"]
        cfg = OmegaConf.merge(base_cfg, cfg)

    apply_grouped_config_aliases(cfg)
    OmegaConf.resolve(cfg)
    return cfg
