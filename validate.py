"""Validate or test a saved run from a config file and checkpoint.

This script mirrors the config-resolution and model/datamodule construction
used by ``train.py``, but skips training entirely. It can also export the
hardest validation failures from a saved checkpoint so debugging a run does
not require re-running the full training loop.
"""

import argparse
from pathlib import Path

import lightning as pl
import torch
from omegaconf import OmegaConf

from data_modules import DATA_REGISTRY
from models import MODEL_REGISTRY
from train import (
    apply_grouped_config_aliases,
    build_runtime_config_dict,
    log_hard_val_examples,
)


def resolve_config(config_path: str):
    """Load a config file, merge its base config, and flatten grouped aliases."""
    cfg = OmegaConf.load(config_path)
    if "_base_" in cfg:
        base_cfg = OmegaConf.load(cfg._base_)
        del cfg["_base_"]
        cfg = OmegaConf.merge(base_cfg, cfg)

    apply_grouped_config_aliases(cfg)
    OmegaConf.resolve(cfg)
    return cfg, build_runtime_config_dict(cfg)


def resolve_checkpoint_path(output_path: str, checkpoint_name: str) -> str:
    """Resolve ``best`` / ``last`` / ``auto`` checkpoint selectors to a file path."""
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
        checkpoint_path = find_named_checkpoint(
            checkpoint_label
        )
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


def load_checkpoint_weights(model, checkpoint_path: str) -> None:
    """Load raw checkpoint weights into the already-instantiated model."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])


def export_hard_validation_examples(
    model,
    datamodule,
    output_path: str,
    num_hard_examples: int,
) -> None:
    """Export hard validation failures for standard and meta-learning models.

    Standard per-example models reuse the ``train.py`` helper directly.
    Meta-models do not expose that path; instead they return task records
    which we render through the model's own gallery helpers.
    """
    datamodule.setup(stage="validate")

    if getattr(model, "supports_hard_val_examples", True):
        log_hard_val_examples(
            model=model,
            datamodule=datamodule,
            output_path=output_path,
            wandb_logger=None,
            num_hard_examples=num_hard_examples,
        )
        return

    if not hasattr(model, "select_hard_task_records"):
        print("Model does not support hard validation example export.")
        return

    hard_task_records = model.select_hard_task_records(
        datamodule.val_dataloader(),
        limit=num_hard_examples,
    )
    if not hard_task_records:
        print("No hard validation task examples found.")
        return

    if getattr(model, "log_task_examples", False) and hasattr(model, "log_task_gallery"):
        model.log_task_gallery(
            split_name="val_hard_validate",
            records=hard_task_records,
            output_path=output_path,
            wandb_logger=None,
            key_prefix="val_hard_validate_task",
        )

    if getattr(model, "log_task_attention", False) and hasattr(
        model, "log_task_attention_gallery"
    ):
        model.log_task_attention_gallery(
            split_name="val_hard_validate",
            records=hard_task_records,
            output_path=output_path,
            wandb_logger=None,
            key_prefix="val_hard_validate_attention",
        )


def main():
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
    cli_args = parser.parse_args()

    # Build the runtime config exactly the same way as the training entrypoint
    # so model/datamodule construction stays consistent between scripts.
    cfg, cfg_dict = resolve_config(cli_args.config)
    checkpoint_path = resolve_checkpoint_path(cfg.output_path, cli_args.checkpoint)

    pl.seed_everything(cfg.seed)

    model_cls = MODEL_REGISTRY[cfg.model]
    model = model_cls(**cfg_dict)
    dm_cls = DATA_REGISTRY[cfg.data]
    datamodule = dm_cls(**cfg_dict)

    trainer = pl.Trainer(
        accelerator=cfg.accelerator,
        logger=False,
        num_sanity_val_steps=0,
    )

    print(f"Using checkpoint: {checkpoint_path}")

    if cli_args.log_hard_examples:
        load_checkpoint_weights(model, checkpoint_path)
        export_hard_validation_examples(
            model=model,
            datamodule=datamodule,
            output_path=cfg.output_path,
            num_hard_examples=cli_args.num_hard_examples,
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


if __name__ == "__main__":
    main()
