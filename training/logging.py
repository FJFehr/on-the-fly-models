"""Run artifact logging: model summaries, W&B, results, hard examples, task galleries.

This module owns everything that produces output artefacts during or after a
training run — writing to disk, logging to W&B, or printing to stdout.
"""

import os

import lightning as pl
import torch
from lightning.pytorch.loggers import WandbLogger

from visualisation import figure_to_wandb_image, render_val_example_figure


def _log_wandb_payload(
    wandb_logger,
    payload: dict,
    *,
    step: int | None = None,
) -> None:
    """Log payload to W&B when an active experiment logger is available."""
    experiment = getattr(wandb_logger, "experiment", None)
    if not payload or experiment is None or not hasattr(experiment, "log"):
        return
    if step is None:
        experiment.log(payload)
        return
    experiment.log(payload, step=step)


# ---------------------------------------------------------------------------
# Model summary
# ---------------------------------------------------------------------------


def count_parameters(module: torch.nn.Module, *, trainable_only: bool | None = None) -> int:
    """Count parameters on a module with an optional trainability filter."""
    parameters = module.parameters()
    if trainable_only is True:
        parameters = (parameter for parameter in parameters if parameter.requires_grad)
    elif trainable_only is False:
        parameters = (parameter for parameter in parameters if not parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)


def _format_parameter_count(num_parameters: int) -> str:
    return f"{num_parameters:,}"


def _describe_hyper_projection(hypermodel) -> str:
    parts = [
        f"Linear({hypermodel.hyper_output_dim} -> {hypermodel.bottleneck_dim})",
        "GELU",
        f"Linear({hypermodel.bottleneck_dim} -> {hypermodel.total_target_params})",
    ]
    if hypermodel.noise_std > 0.0:
        parts.insert(2, f"Noise(std={hypermodel.noise_std})")
    return " + ".join(parts)


def _build_generic_model_summary(model: torch.nn.Module) -> dict[str, int | str]:
    model_repr = str(model)
    total_params = count_parameters(model)
    trainable_params = count_parameters(model, trainable_only=True)
    summary_text = "\n".join(
        [
            "Model architecture:",
            model_repr,
            f"Total parameters: {_format_parameter_count(total_params)}",
            f"Trainable parameters: {_format_parameter_count(trainable_params)}",
        ]
    )
    return {
        "model_repr": model_repr,
        "summary_text": summary_text,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
    }


def _build_hypermodel_summary(model: torch.nn.Module) -> dict[str, int | str]:
    """Build a hypermodel-aware summary for the simplified binary HyperModel path."""
    hypermodel = model.hypermodel
    hypernetwork = hypermodel.hypernetwork
    target_model = hypermodel.target_model
    shared_task_token_embedder = getattr(model, "shared_task_token_embedder", None)

    hypernetwork_backbone_params = count_parameters(hypernetwork, trainable_only=True)
    if hypermodel.use_vae:
        bottleneck_modules = [hypermodel.hyper_mu, hypermodel.hyper_log_sigma]
    else:
        bottleneck_modules = [hypermodel.hyper_bottleneck]
    hyper_head_params = (
        sum(count_parameters(m, trainable_only=True) for m in bottleneck_modules)
        + hypermodel.pool_query.numel()
        + count_parameters(hypermodel.hyper_out, trainable_only=True)
    )
    shared_embedding_params = (
        count_parameters(shared_task_token_embedder, trainable_only=True)
        if shared_task_token_embedder is not None
        else 0
    )
    hypernetwork_trainable_params = (
        hypernetwork_backbone_params + hyper_head_params + shared_embedding_params
    )
    target_non_trainable_params = count_parameters(target_model, trainable_only=False)
    total_params = count_parameters(model)

    model_repr = str(model)
    summary_lines = [
        "Model architecture:",
        f"Hypernetwork: {hypernetwork!r}",
        f"Hyper projection: {_describe_hyper_projection(hypermodel)}",
        f"Target: {target_model!r}",
    ]
    if shared_task_token_embedder is not None:
        summary_lines.append(f"Shared task embeddings: {shared_task_token_embedder!r}")
    summary_lines.extend(
        [
            "Hypernetwork trainable params: "
            f"{_format_parameter_count(hypernetwork_trainable_params)}",
            f"Target non-trainable params: {_format_parameter_count(target_non_trainable_params)}",
            f"Total params: {_format_parameter_count(total_params)}",
            "",
            model_repr,
        ]
    )
    summary_text = "\n".join(summary_lines)
    return {
        "model_repr": model_repr,
        "summary_text": summary_text,
        "total_parameters": total_params,
        "trainable_parameters": hypernetwork_trainable_params,
    }


def describe_model(model: torch.nn.Module) -> dict[str, int | str]:
    """Build and print a model summary for the current run.

    The returned dictionary is reused for disk artefacts and W&B metadata so
    we only compute parameter counts once.
    """
    if hasattr(model, "hypermodel") and hasattr(model.hypermodel, "hypernetwork"):
        summary = _build_hypermodel_summary(model)
    else:
        summary = _build_generic_model_summary(model)

    print(summary["summary_text"])
    return summary


def write_model_summary(output_path: str, model_summary: dict[str, int | str]) -> None:
    """Write the printable model summary to disk."""
    with open(os.path.join(output_path, "model.txt"), "w", encoding="utf-8") as model_file:
        model_file.write(str(model_summary["summary_text"]))
        model_file.write("\n")


# ---------------------------------------------------------------------------
# W&B logger
# ---------------------------------------------------------------------------


def create_wandb_logger(
    cfg,
    runtime_cfg: dict,
    model_summary: dict[str, int | str],
    *,
    run_name: str | None = None,
    job_type: str | None = None,
    resume: str = "allow",
) -> WandbLogger:
    """Create and prime the W&B logger for one run.

    The entrypoint relies on model-side `self.log(...)` calls for metrics.
    This helper is responsible for run-level metadata: config fields, model
    parameter counts, and the printable architecture string.
    """
    wandb_logger = WandbLogger(
        project=cfg.project_name,
        entity=cfg.get("entity"),
        name=run_name,
        job_type=job_type,
        resume=resume,
        log_model=False,
    )
    wandb_logger.log_hyperparams(runtime_cfg)
    wandb_logger.log_hyperparams(
        {
            "total_parameters": model_summary["total_parameters"],
            "trainable_parameters": model_summary["trainable_parameters"],
        }
    )

    experiment = getattr(wandb_logger, "experiment", None)
    if experiment is not None and hasattr(experiment, "summary"):
        experiment.summary["total_parameters"] = model_summary["total_parameters"]
        experiment.summary["trainable_parameters"] = model_summary["trainable_parameters"]
        experiment.summary["model_architecture"] = model_summary["model_repr"]

    return wandb_logger


# ---------------------------------------------------------------------------
# Results file
# ---------------------------------------------------------------------------


def write_results_file(
    output_path: str,
    checkpoint_path: str | None,
    val_results: list[dict],
    test_results: list[dict],
) -> None:
    """Write final evaluation results to a plain-text file in the run directory.

    This file is meant to be a quick local summary of what the run produced,
    independent of Lightning logs or W&B.
    """
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


# ---------------------------------------------------------------------------
# Hard example export
# ---------------------------------------------------------------------------


def log_hard_val_examples(
    model: pl.LightningModule,
    datamodule,
    output_path: str,
    wandb_logger=None,
    num_hard_examples: int = 3,
    key_prefix: str = "val_hard_example",
    snapshot_label: str | None = None,
) -> None:
    """Find the hardest validation failures for pair-based models and log them.

    This path is only for pair-based models that expose canonical logits and
    decoded predictions over `(batch, seq_len)`. Task-level models use their
    own task-gallery path instead.
    """
    model.eval()
    device = next(model.parameters()).device
    wrong_examples = []

    with torch.no_grad():
        for batch in datamodule.val_dataloader():
            inputs = batch["inputs"].to(device)
            targets = batch["targets"].to(device)

            logits = model.format_logits(model(inputs), targets)
            predictions = model.decode_logits(logits)
            targets_long = targets.long()
            exact_matches = (predictions == targets_long).all(dim=1)
            position_accuracies = (predictions == targets_long).float().mean(dim=1)

            for index in range(inputs.size(0)):
                if exact_matches[index]:
                    continue
                wrong_examples.append(
                    {
                        "input": inputs[index].cpu().long().tolist(),
                        "target": targets_long[index].cpu().long().tolist(),
                        "prediction": predictions[index].cpu().long().tolist(),
                        "position_accuracy": position_accuracies[index].item(),
                        "task_category": batch["task_category"][index],
                        "task_id": batch["task_id"][index].item(),
                    }
                )

    if not wrong_examples:
        print("No wrong validation examples found - skipping hard example logging.")
        return

    wrong_examples.sort(key=lambda example: example["position_accuracy"])
    hard_examples = wrong_examples[:num_hard_examples]
    hard_dir = os.path.join(output_path, "hard_examples")
    os.makedirs(hard_dir, exist_ok=True)
    wandb_payload = {}

    for index, example in enumerate(hard_examples):
        num_wrong = sum(
            1
            for target_value, prediction_value in zip(
                example["target"],
                example["prediction"],
                strict=True,
            )
            if target_value != prediction_value
        )
        sequence_length = len(example["target"])
        caption = (
            f"{example['task_category']}:{example['task_id']} | "
            f"pos_acc={example['position_accuracy']:.2f} | "
            f"{num_wrong}/{sequence_length} wrong"
        )

        figure = render_val_example_figure(
            input_sequence=example["input"],
            target_sequence=example["target"],
            prediction_sequence=example["prediction"],
        )
        filename = (
            f"hard_{index}_{example['task_category']}_{example['task_id']}"
            f"_acc{example['position_accuracy']:.2f}.png"
        )
        figure.savefig(os.path.join(hard_dir, filename), dpi=150, bbox_inches="tight")
        wandb_payload[f"{key_prefix}_{index}"] = figure_to_wandb_image(figure, caption=caption)

    if snapshot_label is not None:
        wandb_payload["hard_example_snapshot"] = snapshot_label
    _log_wandb_payload(wandb_logger, wandb_payload)

    print(
        f"Logged {len(hard_examples)} hard validation examples "
        f"({len(wrong_examples)} total failures) to {hard_dir}"
    )


def export_hard_validation_examples(
    model: pl.LightningModule,
    datamodule,
    output_path: str,
    wandb_logger=None,
    num_hard_examples: int = 3,
    key_prefix: str = "val_hard_example",
    snapshot_label: str | None = None,
) -> None:
    """Export hard validation artefacts for pair-based and task-based models."""
    datamodule.setup(stage="validate")

    if getattr(model, "supports_hard_val_examples", True):
        log_hard_val_examples(
            model=model,
            datamodule=datamodule,
            output_path=output_path,
            wandb_logger=wandb_logger,
            num_hard_examples=num_hard_examples,
            key_prefix=key_prefix,
            snapshot_label=snapshot_label,
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
            wandb_logger=wandb_logger,
            key_prefix=f"{key_prefix}_task",
        )

    if getattr(model, "log_task_attention", False) and hasattr(
        model, "log_task_attention_gallery"
    ):
        model.log_task_attention_gallery(
            split_name="val_hard_validate",
            records=hard_task_records,
            output_path=output_path,
            wandb_logger=wandb_logger,
            key_prefix=f"{key_prefix}_attention",
        )


# ---------------------------------------------------------------------------
# Final task gallery
# ---------------------------------------------------------------------------


def log_final_task_visualizations(model, datamodule, output_path: str, wandb_logger=None) -> None:
    """Emit final task galleries and attention plots for task-level models.

    This runs after training so the run directory includes a stable final
    snapshot for representative train/val tasks and, when supported, a set of
    especially hard validation tasks.
    """
    trainer = getattr(model, "trainer", None)
    if trainer is not None and not trainer.is_global_zero:
        return

    if not getattr(model, "supports_task_visualization", False):
        return

    train_task_ids = model.selected_representative_task_ids.get("train", [])
    if not train_task_ids:
        train_task_ids = model.select_representative_task_records_from_dataset(
            datamodule.train_dataset,
            split_name="train",
            limit=model.num_periodic_train_task_examples,
        )
    val_task_ids = model.selected_representative_task_ids.get("val", [])
    if not val_task_ids:
        val_task_ids = model.select_representative_task_records_from_dataset(
            datamodule.val_dataset,
            split_name="val",
            limit=model.num_periodic_val_task_examples,
        )

    train_final_records = model.collect_task_records_from_dataset_by_task_ids(
        datamodule.train_dataset,
        train_task_ids,
    )
    val_final_records = model.collect_task_records_from_dataset_by_task_ids(
        datamodule.val_dataset,
        val_task_ids,
    )
    val_hard_final_records = []
    if hasattr(model, "select_hard_task_records"):
        val_hard_final_records = model.select_hard_task_records(
            datamodule.val_dataloader(),
            limit=getattr(model, "num_final_hard_val_task_examples", 3),
        )

    if getattr(model, "log_task_examples", False):
        model.log_task_gallery(
            split_name="train_final",
            records=train_final_records,
            output_path=output_path,
            wandb_logger=wandb_logger,
            key_prefix="train_task",
        )
        model.log_task_gallery(
            split_name="val_final",
            records=val_final_records,
            output_path=output_path,
            wandb_logger=wandb_logger,
            key_prefix="val_task",
        )
        if val_hard_final_records:
            model.log_task_gallery(
                split_name="val_hard_final",
                records=val_hard_final_records,
                output_path=output_path,
                wandb_logger=wandb_logger,
                key_prefix="val_hard_final_task",
            )

    if getattr(model, "log_task_attention", False):
        model.log_task_attention_gallery(
            split_name="train_final",
            records=train_final_records,
            output_path=output_path,
            wandb_logger=wandb_logger,
            key_prefix="train_task_attention",
        )
        model.log_task_attention_gallery(
            split_name="val_final",
            records=val_final_records,
            output_path=output_path,
            wandb_logger=wandb_logger,
            key_prefix="val_task_attention",
        )
        if val_hard_final_records:
            model.log_task_attention_gallery(
                split_name="val_hard_final",
                records=val_hard_final_records,
                output_path=output_path,
                wandb_logger=wandb_logger,
                key_prefix="val_hard_final_task_attention",
            )
