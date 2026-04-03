# train.py
# ---------------------------------------------------------------------------
# Config-driven training entrypoint for on-the-fly-models.
#
# This script is the single entrypoint for all experiments. It loads a
# self-contained YAML config file, looks up the model and data module
# from their respective registries, and runs training + evaluation via
# PyTorch Lightning.
#
# Usage:
#   uv run python train.py --config configs/mlp_arc1d.yaml
#
# Key design decisions:
#   - One YAML config = one experiment. Configs are verbose and self-contained
#     so every experiment is fully reproducible from its config alone.
#   - The resolved config is saved to the output directory, so experiment
#     outputs are self-documenting.
#   - Models and data modules are looked up by string key from registries
#     (see models/__init__.py and data_modules/__init__.py). Adding a new
#     model or data module requires zero changes to this file.
#   - Checkpoint resume is handled by Lightning natively via ckpt_path="last"
#     and ModelCheckpoint(save_last=True).
#   - WandB run ID management is handled by WandbLogger(resume="allow"),
#     eliminating the need for manual wandb_id.txt files.
# ---------------------------------------------------------------------------

import argparse
import os

import lightning as pl
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf

import wandb

# Import the registries — these map config string keys to classes
from data_modules import DATA_REGISTRY
from models import MODEL_REGISTRY
from visualisation import figure_to_wandb_image, render_val_example_figure


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


def log_hard_val_examples(model, datamodule, output_path, wandb_logger=None, num_hard_examples=3):
    """Find the hardest validation failures and save them as images.

    Runs a full pass over the validation set, collects examples where the
    model's prediction doesn't exactly match the target, ranks them by
    per-position accuracy (lowest = hardest), and saves the top N as PNG
    images. Optionally logs them to wandb if a logger is available.
    """
    model.eval()
    device = next(model.parameters()).device
    val_loader = datamodule.val_dataloader()

    wrong_examples = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(device)
            targets = batch["targets"].to(device)

            logits = model.format_logits(model(inputs), targets)
            preds = model.decode_logits(logits)
            targets_long = targets.long()

            exact_matches = (preds == targets_long).all(dim=1)
            position_accs = (preds == targets_long).float().mean(dim=1)

            for i in range(inputs.size(0)):
                if not exact_matches[i]:
                    wrong_examples.append(
                        {
                            "input": inputs[i].cpu().long().tolist(),
                            "target": targets_long[i].cpu().long().tolist(),
                            "prediction": preds[i].cpu().long().tolist(),
                            "position_accuracy": position_accs[i].item(),
                            "task_category": batch["task_category"][i],
                            "task_id": batch["task_id"][i].item(),
                        }
                    )

    if not wrong_examples:
        print("No wrong validation examples found — skipping hard example logging.")
        return

    # Sort by position accuracy ascending (hardest first)
    wrong_examples.sort(key=lambda x: x["position_accuracy"])
    hard_examples = wrong_examples[:num_hard_examples]

    # Save to disk
    hard_dir = os.path.join(output_path, "hard_examples")
    os.makedirs(hard_dir, exist_ok=True)

    wandb_payload = {}
    for idx, ex in enumerate(hard_examples):
        num_wrong = sum(1 for t, p in zip(ex["target"], ex["prediction"], strict=True) if t != p)
        seq_len = len(ex["target"])
        caption = (
            f"{ex['task_category']}:{ex['task_id']} | "
            f"pos_acc={ex['position_accuracy']:.2f} | "
            f"{num_wrong}/{seq_len} wrong"
        )

        fig = render_val_example_figure(
            input_sequence=ex["input"],
            target_sequence=ex["target"],
            prediction_sequence=ex["prediction"],
        )

        filename = (
            f"hard_{idx}_{ex['task_category']}_{ex['task_id']}"
            f"_acc{ex['position_accuracy']:.2f}.png"
        )
        fig.savefig(os.path.join(hard_dir, filename), dpi=150, bbox_inches="tight")

        wandb_payload[f"hard_example_{idx}"] = figure_to_wandb_image(fig, caption=caption)

    # Log to wandb if available
    if (
        wandb_logger is not None
        and hasattr(wandb_logger, "experiment")
        and isinstance(wandb_logger.experiment, wandb.sdk.wandb_run.Run)
    ):
        wandb_logger.experiment.log(wandb_payload)

    print(
        f"Logged {len(hard_examples)} hard validation examples "
        f"({len(wrong_examples)} total failures) to {hard_dir}"
    )


class MetaTaskVisualizationCallback(Callback):
    """Meta-model specific task-level visualization logging during validation."""

    def __init__(self, output_path: str, wandb_logger=None):
        super().__init__()
        self.output_path = output_path
        self.wandb_logger = wandb_logger
        self.has_logged_pretrain_snapshot = False

    def emit_task_visualizations(
        self,
        pl_module: pl.LightningModule,
        datamodule,
        example_split_prefix: str,
        example_key_prefix: str,
        attention_split_prefix: str,
        attention_key_prefix: str,
    ) -> None:
        train_task_ids = pl_module.select_representative_task_records_from_dataset(
            datamodule.train_dataset,
            split_name="train",
            limit=pl_module.num_periodic_train_task_examples,
        )
        val_task_ids = pl_module.select_representative_task_records_from_dataset(
            datamodule.val_dataset,
            split_name="val",
            limit=pl_module.num_periodic_val_task_examples,
        )
        train_records = pl_module.collect_task_records_from_dataset_by_task_ids(
            datamodule.train_dataset,
            train_task_ids,
        )
        val_records = pl_module.collect_task_records_from_dataset_by_task_ids(
            datamodule.val_dataset,
            val_task_ids,
        )

        if getattr(pl_module, "log_task_examples", False):
            pl_module.log_task_gallery(
                split_name=example_split_prefix.format(split_name="train"),
                records=train_records,
                output_path=self.output_path,
                wandb_logger=self.wandb_logger,
                key_prefix=example_key_prefix.format(split_name="train_task"),
            )
            pl_module.log_task_gallery(
                split_name=example_split_prefix.format(split_name="val"),
                records=val_records,
                output_path=self.output_path,
                wandb_logger=self.wandb_logger,
                key_prefix=example_key_prefix.format(split_name="val_task"),
            )

        if getattr(pl_module, "log_task_attention", False):
            pl_module.log_task_attention_gallery(
                split_name=attention_split_prefix.format(split_name="train"),
                records=train_records,
                output_path=self.output_path,
                wandb_logger=self.wandb_logger,
                key_prefix=attention_key_prefix.format(split_name="train_task_attention"),
            )
            pl_module.log_task_attention_gallery(
                split_name=attention_split_prefix.format(split_name="val"),
                records=val_records,
                output_path=self.output_path,
                wandb_logger=self.wandb_logger,
                key_prefix=attention_key_prefix.format(split_name="val_task_attention"),
            )

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.has_logged_pretrain_snapshot:
            return
        if not getattr(pl_module, "supports_task_visualization", False):
            return
        if not (
            getattr(pl_module, "log_task_examples", False)
            or getattr(pl_module, "log_task_attention", False)
        ):
            return

        self.emit_task_visualizations(
            pl_module=pl_module,
            datamodule=trainer.datamodule,
            example_split_prefix="pretrain_{split_name}",
            example_key_prefix="pretrain_{split_name}",
            attention_split_prefix="pretrain_{split_name}",
            attention_key_prefix="pretrain_{split_name}",
        )
        self.has_logged_pretrain_snapshot = True

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        if not getattr(pl_module, "supports_task_visualization", False):
            return

        every_n_epochs = getattr(pl_module, "log_task_examples_every_n_epochs", 1)
        should_log_examples = (
            getattr(pl_module, "log_task_examples", False)
            and every_n_epochs >= 1
            and trainer.current_epoch % every_n_epochs == 0
        )
        attention_every_n_epochs = getattr(
            pl_module,
            "log_task_attention_every_n_epochs",
            every_n_epochs,
        )
        should_log_attention = (
            getattr(pl_module, "log_task_attention", False)
            and attention_every_n_epochs >= 1
            and trainer.current_epoch % attention_every_n_epochs == 0
        )
        if not should_log_examples and not should_log_attention:
            return

        self.emit_task_visualizations(
            pl_module=pl_module,
            datamodule=trainer.datamodule,
            example_split_prefix="{split_name}",
            example_key_prefix="{split_name}",
            attention_split_prefix="{split_name}",
            attention_key_prefix="{split_name}_task_attention",
        )


def apply_grouped_config_aliases(cfg) -> None:
    """Translate grouped experiment config sections into the flat runtime shape."""
    if "name" in cfg and "model" not in cfg:
        cfg["model"] = cfg["name"]


def build_runtime_config_dict(cfg) -> dict:
    """Flatten grouped config sections into the kwargs expected by the runtime."""
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    if not isinstance(cfg_dict, dict):
        msg = "Resolved config must be a dictionary."
        raise ValueError(msg)

    runtime_cfg = dict(cfg_dict)

    hyper_model_cfg = runtime_cfg.pop("hyper_model", None)
    if isinstance(hyper_model_cfg, dict):
        runtime_cfg.update(hyper_model_cfg)

    training_cfg = runtime_cfg.pop("training", None)
    if isinstance(training_cfg, dict):
        runtime_cfg.update(training_cfg)

    target_model_cfg = runtime_cfg.pop("target_model", None)
    if isinstance(target_model_cfg, dict):
        rnn_cfg = target_model_cfg.get("rnn")
        if isinstance(rnn_cfg, dict):
            runtime_cfg["target_rnn_hidden_dim"] = rnn_cfg["hidden_dim"]
            runtime_cfg["target_rnn_bidirectional"] = rnn_cfg["bidirectional"]
            runtime_cfg["target_rnn_num_layers"] = rnn_cfg["num_layers"]
            runtime_cfg["target_rnn_use_skip_connections"] = rnn_cfg.get(
                "use_skip_connections",
                False,
            )
        cnn_cfg = target_model_cfg.get("cnn")
        if isinstance(cnn_cfg, dict):
            runtime_cfg["target_cnn_hidden_channels"] = cnn_cfg["hidden_channels"]
            runtime_cfg["target_cnn_kernel_size"] = cnn_cfg.get("kernel_size", 3)
            runtime_cfg["target_cnn_num_layers"] = cnn_cfg.get("num_layers", 1)
            runtime_cfg["target_cnn_use_skip_connections"] = cnn_cfg.get(
                "use_skip_connections", False
            )

    return runtime_cfg


def main():
    """Main training function.

    Parses the --config CLI argument, loads the YAML config, instantiates
    the model and data module from registries, sets up logging and
    checkpointing, runs training, and evaluates on the test set.
    """

    # -----------------------------------------------------------------------
    # 1. Parse CLI arguments
    # -----------------------------------------------------------------------
    # The only CLI argument is --config, which points to a self-contained
    # YAML file with ALL experiment parameters.
    parser = argparse.ArgumentParser(
        description="Train a model using a self-contained YAML config."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to the experiment YAML config file (e.g. configs/mlp_arc1d.yaml)",
    )
    cli_args = parser.parse_args()

    # -----------------------------------------------------------------------
    # 2. Load and resolve the config
    # -----------------------------------------------------------------------
    cfg = OmegaConf.load(cli_args.config)

    # Support config hierarchy: if _base_ is set, load the base config and
    # merge the experiment config on top. This lets experiment configs stay
    # minimal — only the fields that differ from the base need to be specified.
    if "_base_" in cfg:
        base_cfg = OmegaConf.load(cfg._base_)
        del cfg["_base_"]
        cfg = OmegaConf.merge(base_cfg, cfg)

    apply_grouped_config_aliases(cfg)
    OmegaConf.resolve(cfg)

    # Convert the OmegaConf DictConfig to a plain Python dict for passing
    # as **kwargs to model and data module constructors.
    cfg_dict = build_runtime_config_dict(cfg)

    # -----------------------------------------------------------------------
    # 3. Save the resolved config to the output directory
    # -----------------------------------------------------------------------
    # Every experiment output should be fully self-contained and reproducible.
    # We save the resolved config (with all interpolations expanded) so that
    # anyone can see exactly what parameters were used for this run.
    os.makedirs(cfg.output_path, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(cfg.output_path, "config.yaml"))

    # -----------------------------------------------------------------------
    # 4. Seed everything for reproducibility
    # -----------------------------------------------------------------------
    # Lightning's seed_everything sets seeds for Python, NumPy, and PyTorch
    # (both CPU and GPU) to ensure reproducible results.
    pl.seed_everything(cfg.seed)

    # -----------------------------------------------------------------------
    # 5. Instantiate model from registry
    # -----------------------------------------------------------------------
    # Look up the model class using the "model" key from the config.
    # The full config dict is passed as **kwargs — the model picks out
    # the parameters it needs (input_dim, hidden_dim, etc.) and ignores
    # the rest via its **kwargs parameter.
    model_cls = MODEL_REGISTRY[cfg.model]
    model = model_cls(**cfg_dict)

    # Print model architecture and parameter count
    model_repr = str(model)
    print("Model architecture:")
    print(model_repr)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    with open(os.path.join(cfg.output_path, "model.txt"), "w", encoding="utf-8") as model_file:
        model_file.write(model_repr)
        model_file.write("\n")
        model_file.write(f"Total parameters: {total_params:,}\n")
        model_file.write(f"Trainable parameters: {trainable_params:,}\n")

    # -----------------------------------------------------------------------
    # 6. Instantiate data module from registry
    # -----------------------------------------------------------------------
    # Same pattern as the model: look up by the "data" config key,
    # pass the full config as **kwargs, and the data module extracts
    # what it needs (data_dir, batch_size, num_workers).
    dm_cls = DATA_REGISTRY[cfg.data]
    dm = dm_cls(**cfg_dict)

    # -----------------------------------------------------------------------
    # 7. Set up checkpointing callback
    # -----------------------------------------------------------------------
    # ModelCheckpoint saves the best model (by val_accuracy) and also
    # saves a "last.ckpt" after every epoch. The "last.ckpt" file enables
    # automatic resume when re-running the same experiment.
    monitor_metric = cfg.primary_metric

    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_metric,  # Metric to monitor (e.g. "val_accuracy")
        mode="max",  # Higher is better for accuracy
        every_n_epochs=1,  # Check after every epoch
        dirpath=cfg.output_path,  # Save checkpoints to the experiment output dir
        filename="best_model",  # Name for the best model checkpoint
        save_last=True,  # Also save last.ckpt for resume support
    )

    callbacks = [checkpoint_callback]

    if cfg.get("stop_on_perfect_val_exact_match", False):
        callbacks.append(
            StopOnMetricThreshold(
                monitor=monitor_metric,
                threshold=1.0,
                mode="max",
            )
        )

    # -----------------------------------------------------------------------
    # 8. Set up WandB logger
    # -----------------------------------------------------------------------
    # WandbLogger handles experiment tracking. resume="allow" means:
    #   - If a run with this project/name already exists, resume it
    #   - If not, create a new run
    # This eliminates the need for manual wandb_id.txt file management.
    wandb_logger = WandbLogger(
        project=cfg.project_name,  # WandB project name
        entity=cfg.get("entity"),  # WandB entity (team/user), optional
        resume="allow",  # Auto-resume if run already exists
        log_model=False,  # Don't upload model artifacts to WandB
    )

    # Log all config parameters as hyperparameters in WandB so they
    # appear in the experiment dashboard and are searchable/filterable.
    wandb_logger.log_hyperparams(cfg_dict)
    wandb_logger.log_hyperparams(
        {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        }
    )
    if hasattr(wandb_logger, "experiment") and isinstance(
        wandb_logger.experiment, wandb.sdk.wandb_run.Run
    ):
        wandb_logger.experiment.summary["total_parameters"] = total_params
        wandb_logger.experiment.summary["trainable_parameters"] = trainable_params
        wandb_logger.experiment.summary["model_architecture"] = model_repr

    if getattr(model, "supports_task_visualization", False):
        callbacks.append(
            MetaTaskVisualizationCallback(
                output_path=cfg.output_path,
                wandb_logger=wandb_logger,
            )
        )

    # -----------------------------------------------------------------------
    # 9. Create Lightning Trainer
    # -----------------------------------------------------------------------
    trainer = pl.Trainer(
        accelerator=cfg.accelerator,  # "auto", "gpu", "cpu", etc.
        logger=wandb_logger,  # Log metrics to WandB
        max_steps=cfg.max_steps,  # Stop after this many training steps
        callbacks=callbacks,  # Save checkpoints and optionally stop on perfect val exact match
        num_sanity_val_steps=0,  # Skip sanity validation (faster startup)
        overfit_batches=1 if cfg.get("overfit_single_batch", False) else 0,
    )

    # -----------------------------------------------------------------------
    # 10. Train the model
    # -----------------------------------------------------------------------
    # ckpt_path="last" tells Lightning to look for a "last.ckpt" file
    # in the checkpoint directory and resume from it if found. If no
    # checkpoint exists, training starts from scratch.
    trainer.fit(model=model, datamodule=dm, ckpt_path="last")

    # -----------------------------------------------------------------------
    # 10.5. Log hard validation examples
    # -----------------------------------------------------------------------
    # Load the best checkpoint so we evaluate the best model (same one used
    # for testing), not the last training step weights.
    best_ckpt = checkpoint_callback.best_model_path or checkpoint_callback.last_model_path
    if best_ckpt:
        ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["state_dict"])

    if getattr(model, "supports_hard_val_examples", True):
        log_hard_val_examples(
            model=model,
            datamodule=dm,
            output_path=cfg.output_path,
            wandb_logger=wandb_logger,
            num_hard_examples=cfg.get("num_hard_examples", 3),
        )

    # -----------------------------------------------------------------------
    # 11. Evaluate on the validation and test sets
    # -----------------------------------------------------------------------
    # Use the best checkpoint for the final validation and test report.
    test_ckpt_path = checkpoint_callback.best_model_path or checkpoint_callback.last_model_path
    val_results = trainer.validate(
        model=model,
        datamodule=dm,
        ckpt_path=test_ckpt_path if test_ckpt_path else None,
    )
    test_results = trainer.test(
        model=model,
        datamodule=dm,
        ckpt_path=test_ckpt_path if test_ckpt_path else None,
    )
    if getattr(model, "supports_task_visualization", False):
        train_task_ids = model.selected_representative_task_ids.get("train", [])
        if not train_task_ids:
            train_task_ids = model.select_representative_task_records_from_dataset(
                dm.train_dataset,
                split_name="train",
                limit=model.num_periodic_train_task_examples,
            )
        val_task_ids = model.selected_representative_task_ids.get("val", [])
        if not val_task_ids:
            val_task_ids = model.select_representative_task_records_from_dataset(
                dm.val_dataset,
                split_name="val",
                limit=model.num_periodic_val_task_examples,
            )

        train_final_records = model.collect_task_records_from_dataset_by_task_ids(
            dm.train_dataset,
            train_task_ids,
        )
        val_final_records = model.collect_task_records_from_dataset_by_task_ids(
            dm.val_dataset,
            val_task_ids,
        )
        val_hard_final_records = model.select_hard_task_records(
            dm.val_dataloader(),
            limit=model.num_final_hard_val_task_examples,
        )

        if getattr(model, "log_task_examples", False):
            model.log_task_gallery(
                split_name="train_final",
                records=train_final_records,
                output_path=cfg.output_path,
                wandb_logger=wandb_logger,
                key_prefix="train_final_task",
            )
            model.log_task_gallery(
                split_name="val_final",
                records=val_final_records,
                output_path=cfg.output_path,
                wandb_logger=wandb_logger,
                key_prefix="val_final_task",
            )
            model.log_task_gallery(
                split_name="val_hard_final",
                records=val_hard_final_records,
                output_path=cfg.output_path,
                wandb_logger=wandb_logger,
                key_prefix="val_hard_final_task",
            )

        if getattr(model, "log_task_attention", False):
            model.log_task_attention_gallery(
                split_name="train_final",
                records=train_final_records,
                output_path=cfg.output_path,
                wandb_logger=wandb_logger,
                key_prefix="train_final_task_attention",
            )
            model.log_task_attention_gallery(
                split_name="val_final",
                records=val_final_records,
                output_path=cfg.output_path,
                wandb_logger=wandb_logger,
                key_prefix="val_final_task_attention",
            )
            model.log_task_attention_gallery(
                split_name="val_hard_final",
                records=val_hard_final_records,
                output_path=cfg.output_path,
                wandb_logger=wandb_logger,
                key_prefix="val_hard_final_task_attention",
            )
    write_results_file(
        output_path=cfg.output_path,
        checkpoint_path=test_ckpt_path if test_ckpt_path else None,
        val_results=val_results,
        test_results=test_results,
    )


if __name__ == "__main__":
    main()
