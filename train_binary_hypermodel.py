"""Dedicated training entrypoint for simplified binary HyperModel experiments.

This script mirrors the main training entrypoint's core experiment-management
behavior for the simpler binary HyperModel track. It loads a self-contained
YAML config, instantiates the configured model and datamodule, saves the
resolved config to the run directory, tracks the run in Weights & Biases, and
writes a final plain-text evaluation summary.
"""

import argparse
import os

import lightning as pl
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf

import wandb

from data_modules import DATA_REGISTRY
from models import MODEL_REGISTRY
from train_utils import (
    StopOnMetricThreshold,
    build_runtime_config_dict,
    load_config,
    write_results_file,
)


class MetaTaskVisualizationCallback(Callback):
    """Periodic task-example logging for task-level meta-learning runs."""

    def __init__(self, output_path: str, wandb_logger=None):
        super().__init__()
        self.output_path = output_path
        self.wandb_logger = wandb_logger
        self.has_logged_pretrain_snapshot = False

    def emit_task_visualizations(
        self,
        pl_module: pl.LightningModule,
        datamodule,
        split_prefix: str,
        key_prefix: str,
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
        pl_module.log_task_gallery(
            split_name=split_prefix.format(split_name="train"),
            records=train_records,
            output_path=self.output_path,
            wandb_logger=self.wandb_logger,
            key_prefix=key_prefix.format(split_name="train_task"),
        )
        pl_module.log_task_gallery(
            split_name=split_prefix.format(split_name="val"),
            records=val_records,
            output_path=self.output_path,
            wandb_logger=self.wandb_logger,
            key_prefix=key_prefix.format(split_name="val_task"),
        )

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.has_logged_pretrain_snapshot:
            return
        if not getattr(pl_module, "supports_task_visualization", False):
            return
        if not getattr(pl_module, "log_task_examples", False):
            return

        self.emit_task_visualizations(
            pl_module=pl_module,
            datamodule=trainer.datamodule,
            split_prefix="pretrain_{split_name}",
            key_prefix="pretrain_{split_name}",
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
        if not should_log_examples:
            return

        self.emit_task_visualizations(
            pl_module=pl_module,
            datamodule=trainer.datamodule,
            split_prefix="{split_name}",
            key_prefix="{split_name}",
        )


def main() -> None:
    """Run the simplified binary hypermodel training flow."""

    # -----------------------------------------------------------------------
    # 1. Parse CLI arguments
    # -----------------------------------------------------------------------
    # The only CLI argument is --config, which points to a self-contained
    # YAML file with all experiment parameters.
    parser = argparse.ArgumentParser(
        description="Train the simplified binary HyperModel from a YAML config."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to the experiment YAML config file.",
    )
    cli_args = parser.parse_args()

    # -----------------------------------------------------------------------
    # 2. Load and resolve the config
    # -----------------------------------------------------------------------
    cfg = load_config(cli_args.config)
    cfg_dict = build_runtime_config_dict(cfg)

    # -----------------------------------------------------------------------
    # 3. Save the resolved config to the output directory
    # -----------------------------------------------------------------------
    # Each run should be self-documenting, so persist the fully resolved
    # config alongside checkpoints and result artifacts.
    os.makedirs(cfg.output_path, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(cfg.output_path, "config.yaml"))

    # -----------------------------------------------------------------------
    # 4. Seed everything for reproducibility
    # -----------------------------------------------------------------------
    pl.seed_everything(cfg.seed)

    # -----------------------------------------------------------------------
    # 5. Instantiate model from registry
    # -----------------------------------------------------------------------
    model_cls = MODEL_REGISTRY[cfg.model]
    model = model_cls(**cfg_dict)

    # Capture a readable model summary for both the terminal and run artifacts.
    model_repr = str(model)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print("Model architecture:")
    print(model_repr)
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
    dm_cls = DATA_REGISTRY[cfg.data]
    dm = dm_cls(**cfg_dict)

    # -----------------------------------------------------------------------
    # 7. Set up checkpointing callback
    # -----------------------------------------------------------------------
    checkpoint_dir = os.path.join(cfg.output_path, "checkpoints")
    monitor_metric = cfg.primary_metric
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_metric,
        mode="max",
        every_n_epochs=1,
        dirpath=checkpoint_dir,
        filename="best_model",
        save_last=True,
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
    # This script relies on the model's Lightning self.log(...) calls for the
    # actual metric stream. The entrypoint only needs to attach a logger and
    # record run-level metadata such as the config and parameter counts.
    wandb_logger = WandbLogger(
        project=cfg.project_name,
        entity=cfg.get("entity"),
        resume="allow",
        log_model=False,
    )
    wandb_logger.log_hyperparams(cfg_dict)
    wandb_logger.log_hyperparams(
        {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        }
    )
    if hasattr(wandb_logger, "experiment") and isinstance(
        wandb_logger.experiment,
        wandb.sdk.wandb_run.Run,
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
        accelerator=cfg.accelerator,
        logger=wandb_logger,
        max_steps=cfg.max_steps,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        overfit_batches=1 if cfg.get("overfit_single_batch", False) else 0,
        log_every_n_steps=1,
    )

    # -----------------------------------------------------------------------
    # 10. Train the model
    # -----------------------------------------------------------------------
    last_ckpt_path = os.path.join(checkpoint_dir, "last.ckpt")
    trainer.fit(
        model=model,
        datamodule=dm,
        ckpt_path=last_ckpt_path if os.path.exists(last_ckpt_path) else None,
    )

    # -----------------------------------------------------------------------
    # 11. Evaluate on the validation and test sets
    # -----------------------------------------------------------------------
    # Use the best checkpoint when available so the final report matches the
    # checkpoint selected by the monitored validation metric.
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

    if getattr(model, "supports_task_visualization", False) and getattr(
        model,
        "log_task_examples",
        False,
    ):
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

    # -----------------------------------------------------------------------
    # 12. Write the final plain-text results summary
    # -----------------------------------------------------------------------
    write_results_file(
        output_path=cfg.output_path,
        checkpoint_path=test_ckpt_path if test_ckpt_path else None,
        val_results=val_results,
        test_results=test_results,
    )


if __name__ == "__main__":
    main()
