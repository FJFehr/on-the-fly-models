"""Thin config-driven validation entrypoint for saved runs."""

import argparse
from pathlib import Path

import lightning as pl

from data_modules import DATA_REGISTRY
from models import MODEL_REGISTRY
from training.config import build_runtime_config_dict, ensure_output_path, load_config
from training.logging import (
    create_wandb_logger,
    describe_model,
    export_hard_validation_examples,
    write_model_summary,
)
from training.trainer import (
    build_trainer,
    load_checkpoint_state,
    resolve_evaluation_checkpoint_path,
)


def parse_args() -> argparse.Namespace:
    """Parse the minimal CLI surface for a validation run."""
    parser = argparse.ArgumentParser(
        description="Validate or test a saved run from a config and checkpoint."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to the experiment YAML config file.",
    )
    parser.add_argument(
        "--checkpoint",
        default="auto",
        type=str,
        help="Checkpoint to load: auto, best, last, or an explicit path.",
    )
    parser.add_argument(
        "--mode",
        default="both",
        choices=("validate", "test", "both"),
        help="Which evaluation pass to run.",
    )
    parser.add_argument(
        "--log-hard-examples",
        action="store_true",
        help="Export the hardest validation failures for the loaded checkpoint.",
    )
    parser.add_argument(
        "--num-hard-examples",
        default=3,
        type=int,
        help="Number of hard validation failures to export when enabled.",
    )
    parser.add_argument(
        "--log-to-wandb",
        action="store_true",
        help="Create an evaluation W&B run and log metrics plus hard examples.",
    )
    return parser.parse_args()


def main() -> None:
    """Run shared evaluation flow for a saved checkpoint."""
    cli_args = parse_args()
    cfg = load_config(cli_args.config)
    runtime_cfg = build_runtime_config_dict(cfg)
    checkpoint_path = resolve_evaluation_checkpoint_path(cfg.output_path, cli_args.checkpoint)

    pl.seed_everything(cfg.seed)

    model = MODEL_REGISTRY[cfg.model](**runtime_cfg)
    ensure_output_path(cfg.output_path)
    model_summary = describe_model(model)
    write_model_summary(cfg.output_path, model_summary)

    datamodule = DATA_REGISTRY[cfg.data](**runtime_cfg)
    wandb_logger = None
    if cli_args.log_to_wandb:
        run_name = f"{Path(cfg.output_path).name}-{cli_args.mode}-evaluation"
        wandb_logger = create_wandb_logger(
            cfg,
            runtime_cfg,
            model_summary,
            run_name=run_name,
            job_type="evaluation",
            resume="never",
        )
    trainer = build_trainer(
        cfg,
        runtime_cfg,
        wandb_logger=wandb_logger,
        evaluation=True,
    )

    print(f"Using checkpoint: {checkpoint_path}")

    if cli_args.log_hard_examples:
        load_checkpoint_state(model, checkpoint_path)
        export_hard_validation_examples(
            model=model,
            datamodule=datamodule,
            output_path=cfg.output_path,
            wandb_logger=wandb_logger,
            num_hard_examples=cli_args.num_hard_examples,
            key_prefix="validate_hard_example",
            snapshot_label="validate_cli",
        )

    if cli_args.mode in {"validate", "both"}:
        val_results = trainer.validate(
            model=model,
            datamodule=datamodule,
            ckpt_path=checkpoint_path,
        )
        print("Validation results:")
        print(val_results)

    if cli_args.mode in {"test", "both"}:
        test_results = trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=checkpoint_path,
        )
        print("Test results:")
        print(test_results)

    experiment = getattr(wandb_logger, "experiment", None)
    if experiment is not None:
        experiment.finish()


if __name__ == "__main__":
    main()
