"""Thin config-driven training entrypoint for all experiments.

This file intentionally keeps very little logic of its own.

The design goal is:
- `train.py` explains the high-level run lifecycle
- `train_utils.py` owns reusable runtime helpers
- model classes own model-specific training behavior and visualisation details

That split keeps this file readable when you want to answer
"what happens during a run?" without also reading every low-level helper.
"""

import argparse

import lightning as pl
import torch

from data_modules import DATA_REGISTRY
from models import MODEL_REGISTRY
from training.config import (
    build_runtime_config_dict,
    ensure_output_path,
    load_config,
    save_resolved_config,
)
from training.logging import (
    create_wandb_logger,
    describe_model,
    write_model_summary,
    write_results_file,
)
from training.trainer import (
    build_callbacks,
    build_trainer,
    resolve_resume_checkpoint_path,
    run_post_training_artifacts,
)


def configure_torch_runtime(runtime_cfg: dict) -> None:
    """Apply repo-level Torch runtime settings from config.

    `float32_matmul_precision` is optional. When set on CUDA hosts it removes
    PyTorch's Tensor Core warning while keeping the trade-off explicit in the
    experiment config.
    """
    precision = runtime_cfg.get("float32_matmul_precision")
    if precision is None:
        return

    allowed_precisions = {"highest", "high", "medium"}
    if precision not in allowed_precisions:
        msg = (
            "float32_matmul_precision must be one of "
            f"{sorted(allowed_precisions)}, got {precision!r}."
        )
        raise ValueError(msg)

    torch.set_float32_matmul_precision(precision)


def parse_args() -> argparse.Namespace:
    """Parse the minimal CLI surface for a training run.

    The runtime is config-driven, so the entrypoint only needs the path to one
    experiment YAML file. Everything else is expected to come from that config.
    """
    parser = argparse.ArgumentParser(
        description="Train a model using a self-contained YAML config.",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to the experiment YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the shared training flow for any registered model/data combination.

    The lifecycle is:
    1. read and resolve the YAML config
    2. instantiate the configured model and datamodule
    3. build trainer infrastructure (logger, callbacks, checkpoints)
    4. fit the model, optionally resuming from `last.ckpt`
    5. run post-training artefact generation plus final validation/test
    6. write a plain-text summary file for the run directory
    """
    cli_args = parse_args()

    # Resolve config inheritance, aliases, and grouped sections before anything
    # is instantiated. The resulting runtime dict is what gets passed to the
    # model and datamodule constructors.
    cfg = load_config(cli_args.config)
    runtime_cfg = build_runtime_config_dict(cfg)
    configure_torch_runtime(runtime_cfg)

    # The output directory is treated as the canonical home for this run:
    # config snapshot, model summary, checkpoints, visualisations, and results.
    ensure_output_path(cfg.output_path)
    save_resolved_config(cfg)

    # Seed after the config is loaded so the chosen seed comes from the run
    # definition rather than from the shell environment.
    pl.seed_everything(cfg.seed)

    # Both the model and datamodule are registry-driven. `train.py` does not
    # know any model-specific constructor details beyond the registry key.
    model = MODEL_REGISTRY[cfg.model](**runtime_cfg)
    model_summary = describe_model(model)
    write_model_summary(cfg.output_path, model_summary)

    datamodule = DATA_REGISTRY[cfg.data](**runtime_cfg)

    # Logging and callback setup are runtime concerns, so they are assembled
    # here before building the Lightning trainer.
    wandb_logger = create_wandb_logger(cfg, runtime_cfg, model_summary)
    callbacks, checkpoint_callback = build_callbacks(cfg, model, wandb_logger=wandb_logger)
    trainer = build_trainer(cfg, runtime_cfg, callbacks, wandb_logger)

    # Resume from `last.ckpt` when the run directory already contains one.
    # If it does not exist, Lightning starts from scratch.
    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=resolve_resume_checkpoint_path(cfg.output_path),
    )

    # Post-training helper owns:
    # - reloading the best/last checkpoint into the model
    # - emitting optional visualisation artefacts
    # - running final validation and test passes
    val_results, test_results, checkpoint_path = run_post_training_artifacts(
        cfg=cfg,
        trainer=trainer,
        model=model,
        datamodule=datamodule,
        checkpoint_callback=checkpoint_callback,
        wandb_logger=wandb_logger,
    )

    # Keep a small text summary in the run directory so the most important
    # final metrics can be inspected without opening W&B or Lightning logs.
    write_results_file(
        output_path=cfg.output_path,
        checkpoint_path=checkpoint_path,
        val_results=val_results,
        test_results=test_results,
    )


if __name__ == "__main__":
    main()
