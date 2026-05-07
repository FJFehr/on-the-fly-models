"""Trainer, callbacks, and checkpoint infrastructure.

This module owns the Lightning training setup:
- repo-specific callbacks (early stop, task visualisation)
- trainer and checkpoint callback construction
- checkpoint path resolution and loading
- post-training orchestration (`run_post_training_artifacts`)
"""

import os
from pathlib import Path

import lightning as pl
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint

from training.logging import (
    export_hard_validation_examples,
    log_final_task_visualizations,
)

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def _select_representative_task_ids(tasks: list[dict], limit: int) -> list[tuple[str, int]]:
    """Select at most one (category, task_id) pair per category in dataset order."""
    if limit < 1:
        return []

    seen_categories = set()
    selected = []
    for task in tasks:
        task_category = task["task_category"]
        if task_category in seen_categories:
            continue
        seen_categories.add(task_category)
        selected.append((task_category, task["task_id"]))
        if len(selected) >= limit:
            break
    return selected


class StopOnMetricThreshold(Callback):
    """Stop training once a monitored validation metric reaches a threshold.

    Used for experiments where reaching exact-match `1.0` is "good enough"
    and we want to stop spending compute immediately.
    """

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


class TaskVisualizationCallback(Callback):
    """Shared task-visualisation callback for task-level models.

    The callback does not decide *how* to render a task — that remains model-
    owned. Its job is only to decide *when* task examples/attention plots
    should be emitted during training and to route the request to the model.
    """

    def __init__(self, output_path: str, wandb_logger=None):
        super().__init__()
        self.output_path = output_path
        self.wandb_logger = wandb_logger
        self.has_logged_pretrain_snapshot = False

    def _snapshot_split_name(self, split_name: str, epoch_index: int) -> str:
        return f"{split_name}_epoch_{epoch_index:04d}"

    def _resolve_task_ids(
        self,
        pl_module: pl.LightningModule,
        dataset,
        split_name: str,
        limit: int,
    ) -> list[int]:
        selected_task_ids = getattr(pl_module, "selected_representative_task_ids", {}).get(
            split_name,
            [],
        )
        if selected_task_ids:
            return selected_task_ids

        if hasattr(pl_module, "select_representative_task_records_from_dataset"):
            return pl_module.select_representative_task_records_from_dataset(
                dataset,
                split_name=split_name,
                limit=limit,
            )

        selected_task_ids = _select_representative_task_ids(getattr(dataset, "tasks", []), limit)
        pl_module.selected_representative_task_ids[split_name] = selected_task_ids
        return selected_task_ids

    def emit_task_visualizations(
        self,
        pl_module: pl.LightningModule,
        datamodule,
        example_split_prefix: str,
        attention_split_prefix: str,
    ) -> None:
        try:
            self._emit_task_visualizations_impl(
                pl_module, datamodule, example_split_prefix, attention_split_prefix
            )
        except Exception as exc:
            print(
                f"[TaskVisualizationCallback] WARNING: skipping task viz — {exc}",
                flush=True,
            )

    def _emit_task_visualizations_impl(
        self,
        pl_module: pl.LightningModule,
        datamodule,
        example_split_prefix: str,
        attention_split_prefix: str,
    ) -> None:
        train_task_ids = self._resolve_task_ids(
            pl_module,
            datamodule.train_dataset,
            split_name="train",
            limit=pl_module.num_periodic_train_task_examples,
        )
        val_task_ids = self._resolve_task_ids(
            pl_module,
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
            train_example_split = example_split_prefix.format(split_name="train")
            val_example_split = example_split_prefix.format(split_name="val")
            pl_module.log_task_gallery(
                split_name=train_example_split,
                records=train_records,
                output_path=self.output_path,
                wandb_logger=self.wandb_logger,
                key_prefix="train_task",
            )
            pl_module.log_task_gallery(
                split_name=val_example_split,
                records=val_records,
                output_path=self.output_path,
                wandb_logger=self.wandb_logger,
                key_prefix="val_task",
            )

        if getattr(pl_module, "log_task_attention", False):
            train_attention_split = attention_split_prefix.format(split_name="train")
            val_attention_split = attention_split_prefix.format(split_name="val")
            pl_module.log_task_attention_gallery(
                split_name=train_attention_split,
                records=train_records,
                output_path=self.output_path,
                wandb_logger=self.wandb_logger,
                key_prefix="train_task_attention",
            )
            pl_module.log_task_attention_gallery(
                split_name=val_attention_split,
                records=val_records,
                output_path=self.output_path,
                wandb_logger=self.wandb_logger,
                key_prefix="val_task_attention",
            )

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.is_global_zero:
            return
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
            example_split_prefix=self._snapshot_split_name("{split_name}", 0),
            attention_split_prefix=self._snapshot_split_name("{split_name}", 0),
        )
        self.has_logged_pretrain_snapshot = True

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.is_global_zero:
            return
        if trainer.sanity_checking:
            return
        if not getattr(pl_module, "supports_task_visualization", False):
            return

        completed_epochs = trainer.current_epoch + 1
        every_n_epochs = getattr(pl_module, "log_task_examples_every_n_epochs", 1)
        should_log_examples = (
            getattr(pl_module, "log_task_examples", False)
            and every_n_epochs >= 1
            and completed_epochs % every_n_epochs == 0
        )
        attention_every_n_epochs = getattr(
            pl_module,
            "log_task_attention_every_n_epochs",
            every_n_epochs,
        )
        should_log_attention = (
            getattr(pl_module, "log_task_attention", False)
            and attention_every_n_epochs >= 1
            and completed_epochs % attention_every_n_epochs == 0
        )
        if not should_log_examples and not should_log_attention:
            return

        self.emit_task_visualizations(
            pl_module=pl_module,
            datamodule=trainer.datamodule,
            example_split_prefix=self._snapshot_split_name(
                "{split_name}",
                completed_epochs,
            ),
            attention_split_prefix=self._snapshot_split_name(
                "{split_name}",
                completed_epochs,
            ),
        )


# ---------------------------------------------------------------------------
# Trainer and checkpoint construction
# ---------------------------------------------------------------------------


def create_checkpoint_callback(cfg) -> ModelCheckpoint:
    """Create the shared best/last checkpoint callback.

    Every run keeps both:
    - `best_model.ckpt` for metric-selected evaluation
    - `last.ckpt` for seamless resume
    """
    metric = cfg.primary_metric
    mode = "min" if metric.endswith("loss") else "max"
    return ModelCheckpoint(
        monitor=metric,
        mode=mode,
        every_n_epochs=1,
        dirpath=cfg.output_path,
        filename="best_model",
        save_last=True,
    )


def build_callbacks(
    cfg, model: pl.LightningModule, wandb_logger=None
) -> tuple[list[Callback], ModelCheckpoint]:
    """Build the shared callback list for one run."""
    checkpoint_callback = create_checkpoint_callback(cfg)
    callbacks: list[Callback] = [checkpoint_callback]

    if cfg.get("stop_on_perfect_val_exact_match", False):
        callbacks.append(
            StopOnMetricThreshold(
                monitor="val_query_exact_match",
                threshold=1.0,
                mode="max",
            )
        )

    if getattr(model, "supports_task_visualization", False):
        callbacks.append(
            TaskVisualizationCallback(
                output_path=cfg.output_path,
                wandb_logger=wandb_logger,
            )
        )

    return callbacks, checkpoint_callback


def build_trainer(
    cfg,
    runtime_cfg: dict,
    callbacks: list[Callback] | None = None,
    wandb_logger=None,
    *,
    evaluation: bool = False,
) -> pl.Trainer:
    """Create the shared Lightning trainer."""
    trainer_kwargs = {
        "accelerator": runtime_cfg["accelerator"],
        "devices": runtime_cfg.get("devices", 1),
        "logger": wandb_logger if wandb_logger not in {None, False} else False,
        "callbacks": callbacks or [],
        "enable_model_summary": False,
        "num_sanity_val_steps": 0,
    }

    if evaluation:
        # Run evaluation on one device so Lightning does not wrap dataloaders
        # in a DistributedSampler that can duplicate samples across ranks.
        trainer_kwargs["devices"] = 1
        return pl.Trainer(**trainer_kwargs)

    if runtime_cfg.get("gradient_clip_val") is not None:
        trainer_kwargs["gradient_clip_val"] = runtime_cfg["gradient_clip_val"]

    return pl.Trainer(
        **trainer_kwargs,
        max_steps=runtime_cfg["max_steps"],
        overfit_batches=1 if cfg.get("overfit_single_batch", False) else 0,
        log_every_n_steps=1,
    )


# ---------------------------------------------------------------------------
# Checkpoint resolution and loading
# ---------------------------------------------------------------------------


def resolve_resume_checkpoint_path(output_path: str) -> str | None:
    """Return the last-checkpoint path when a resumable run already exists."""
    checkpoint_path = os.path.join(output_path, "last.ckpt")
    if os.path.exists(checkpoint_path):
        return checkpoint_path
    return None


def resolve_best_checkpoint_path(checkpoint_callback: ModelCheckpoint) -> str | None:
    """Return the best available checkpoint path after training.

    Prefer the monitored best checkpoint, but fall back to `last.ckpt` so
    evaluation still runs when no "best" path was recorded.
    """
    return checkpoint_callback.best_model_path or checkpoint_callback.last_model_path or None


def resolve_evaluation_checkpoint_path(output_path: str, checkpoint_name: str) -> str:
    """Resolve validation checkpoint selectors to a concrete checkpoint path."""
    output_dir = Path(output_path)

    def find_named_checkpoint(prefix: str) -> Path | None:
        exact_match = output_dir / f"{prefix}.ckpt"
        if exact_match.exists():
            return exact_match

        matching_paths = sorted(output_dir.glob(f"{prefix}*.ckpt"))
        if matching_paths:
            return matching_paths[0]
        return None

    if checkpoint_name in {"best", "last"}:
        checkpoint_label = "best_model" if checkpoint_name == "best" else "last"
        checkpoint_path = find_named_checkpoint(checkpoint_label)
        if checkpoint_path is None:
            msg = f"Checkpoint {checkpoint_label}*.ckpt does not exist in {output_dir}."
            raise FileNotFoundError(msg)
        return str(checkpoint_path)

    if checkpoint_name == "auto":
        for prefix in ("best_model", "last"):
            checkpoint_path = find_named_checkpoint(prefix)
            if checkpoint_path is not None:
                return str(checkpoint_path)
        msg = f"No checkpoint found in {output_dir}."
        raise FileNotFoundError(msg)

    checkpoint_path = Path(checkpoint_name)
    if not checkpoint_path.exists():
        msg = f"Checkpoint {checkpoint_path} does not exist."
        raise FileNotFoundError(msg)
    return str(checkpoint_path)


def load_checkpoint_state(model: pl.LightningModule, checkpoint_path: str | None) -> None:
    """Load model state from a saved Lightning checkpoint when available."""
    if not checkpoint_path:
        return
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])


# ---------------------------------------------------------------------------
# Post-training orchestration
# ---------------------------------------------------------------------------


def run_post_training_artifacts(
    cfg,
    trainer: pl.Trainer,
    model: pl.LightningModule,
    datamodule,
    checkpoint_callback: ModelCheckpoint,
    wandb_logger=None,
) -> tuple[list[dict], list[dict], str | None]:
    """Run post-fit loading, visualisation, validation, and testing.

    Keeping this sequence together avoids scattering "after fit" decisions
    across `train.py`:
    - choose the checkpoint to evaluate
    - load that checkpoint into the in-memory model
    - emit optional local/W&B artefacts
    - run final validation and test passes
    """
    checkpoint_path = resolve_best_checkpoint_path(checkpoint_callback)
    load_checkpoint_state(model, checkpoint_path)

    export_hard_validation_examples(
        model=model,
        datamodule=datamodule,
        output_path=cfg.output_path,
        wandb_logger=wandb_logger,
        num_hard_examples=cfg.get("num_hard_examples", 3),
        key_prefix="val_hard_validate",
        snapshot_label="post_fit_validate",
    )

    val_results = trainer.validate(
        model=model,
        datamodule=datamodule,
        ckpt_path=checkpoint_path if checkpoint_path else None,
    )
    test_results = trainer.test(
        model=model,
        datamodule=datamodule,
        ckpt_path=checkpoint_path if checkpoint_path else None,
    )

    log_final_task_visualizations(
        model=model,
        datamodule=datamodule,
        output_path=cfg.output_path,
        wandb_logger=wandb_logger,
    )
    return val_results, test_results, checkpoint_path
