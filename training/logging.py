"""Run artifact logging: model summaries, W&B, results, hard examples, task galleries.

This module owns everything that produces output artefacts during or after a
training run — writing to disk, logging to W&B, or printing to stdout.
"""

import os
from pathlib import Path

import lightning as pl
import numpy as np
import torch
from lightning.pytorch.loggers import WandbLogger

from visualisation import figure_to_wandb_image
from visualisation.core.embedding_clusters import (
    DEFAULT_GROUP_STYLES,
    compute_linear_probe_accuracy,
    render_embedding_cluster_figure,
    render_single_projection_figures,
)


def log_wandb_payload(
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


def _write_wandb_summary(experiment, summary_values: dict[str, int | str]) -> None:
    """Write summary metadata when the experiment exposes a dict-like summary.

    On non-zero distributed ranks Lightning can return a dummy experiment
    object whose `.summary` is a no-op method instead of a mapping. In that
    case we should skip summary assignment entirely.
    """
    if experiment is None:
        return

    summary = getattr(experiment, "summary", None)
    if summary is None or not hasattr(summary, "__setitem__"):
        return

    for key, value in summary_values.items():
        summary[key] = value


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


def describe_model(model: torch.nn.Module) -> dict[str, int | str]:
    """Print and return the model architecture with its parameter counts.

    Lists trainable/total parameters for each top-level submodule (and, for the
    hypernetwork, each of its own submodules), then the full module tree.
    """
    lines = ["Parameters (trainable / total):"]
    for name, child in model.named_children():
        children = [(name, child)]
        if name == "hypernetwork":
            children = [(f"{name}.{sub}", module) for sub, module in child.named_children()]
        for child_name, module in children:
            trainable = _format_parameter_count(count_parameters(module, trainable_only=True))
            total = _format_parameter_count(count_parameters(module))
            lines.append(f"  {child_name}: {trainable} / {total}")
    total_params = count_parameters(model)
    trainable_params = count_parameters(model, trainable_only=True)
    model_repr = str(model)
    summary_text = "\n".join(
        [
            *lines,
            f"Total trainable parameters: {_format_parameter_count(trainable_params)}",
            f"Total parameters: {_format_parameter_count(total_params)}",
            "",
            model_repr,
        ]
    )
    print(summary_text)
    return {
        "model_repr": model_repr,
        "summary_text": summary_text,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
    }


def write_model_summary(output_path: str, model_summary: dict[str, int | str]) -> None:
    """Write the printable model summary to disk."""
    with open(os.path.join(output_path, "model.txt"), "w", encoding="utf-8") as model_file:
        model_file.write(str(model_summary["summary_text"]))
        model_file.write("\n")


# ---------------------------------------------------------------------------
# W&B logger
# ---------------------------------------------------------------------------


def _is_global_zero_process() -> bool:
    """Return whether this process should own external logging side effects."""
    rank = os.environ.get("RANK")
    if rank is None:
        return True
    return rank == "0"


def create_wandb_logger(
    cfg,
    runtime_cfg: dict,
    model_summary: dict[str, int | str],
    *,
    run_name: str | None = None,
    job_type: str | None = None,
    resume: str = "allow",
) -> WandbLogger | bool:
    """Create and prime the W&B logger for one run.

    The entrypoint relies on model-side `self.log(...)` calls for metrics.
    This helper is responsible for run-level metadata: config fields, model
    parameter counts, and the printable architecture string.
    """
    if not _is_global_zero_process():
        return False

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

    _write_wandb_summary(
        getattr(wandb_logger, "experiment", None),
        {
            "total_parameters": model_summary["total_parameters"],
            "trainable_parameters": model_summary["trainable_parameters"],
            "model_architecture": model_summary["model_repr"],
        },
    )

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


def _select_hard_task_records(
    model: pl.LightningModule,
    dataloader,
    *,
    limit: int | None = None,
    one_per_category: bool = False,
) -> list[dict]:
    """Select failed task records ordered from hardest to easiest."""
    if not hasattr(model, "collect_task_records_from_dataloader"):
        return []

    records = model.collect_task_records_from_dataloader(dataloader)
    failed_records = [record for record in records if not record["query_exact_match"]]
    failed_records.sort(
        key=lambda record: (
            record["query_accuracy"],
            record["task_category"],
            record["task_id"],
        )
    )

    if not one_per_category:
        if limit is None or limit < 1:
            return failed_records
        return failed_records[:limit]

    selected_records = []
    seen_categories = set()
    for record in failed_records:
        task_category = record["task_category"]
        if task_category in seen_categories:
            continue
        seen_categories.add(task_category)
        selected_records.append(record)
        if limit is not None and limit > 0 and len(selected_records) >= limit:
            break
    return selected_records


def export_hard_validation_examples(
    model: pl.LightningModule,
    datamodule,
    output_path: str,
    wandb_logger=None,
    num_hard_examples: int = 3,
    key_prefix: str = "val_hard_example",
    split: str = "val",
) -> None:
    """Export hard example artefacts for pair-based and task-based models.

    split: "val" (default) or "test" — controls which dataloader is used.
    """
    if split not in {"val", "test"}:
        msg = f"split must be 'val' or 'test', got {split!r}."
        raise ValueError(msg)
    stage = "test" if split == "test" else "validate"
    datamodule.setup(stage=stage)

    # The direct model exports its own hard examples; the hypernetwork saves a gallery of
    # its worst validation tasks instead.
    if hasattr(model, "export_hard_examples"):
        model.export_hard_examples(
            datamodule=datamodule,
            output_path=output_path,
            wandb_logger=wandb_logger,
            num_examples=num_hard_examples,
            key_prefix=key_prefix,
            split=split,
        )
        return

    if not getattr(model, "log_task_examples", False):
        return
    dataloader = datamodule.test_dataloader() if split == "test" else datamodule.val_dataloader()
    hard_task_records = _select_hard_task_records(model, dataloader, limit=num_hard_examples)
    if not hard_task_records:
        print("No hard validation task examples found.")
        return
    model.log_task_gallery(
        split_name="val_hard_validate",
        records=hard_task_records,
        output_path=output_path,
        wandb_logger=wandb_logger,
        key_prefix=f"{key_prefix}_task",
    )


# ---------------------------------------------------------------------------
# Final task gallery
# ---------------------------------------------------------------------------


def log_final_task_visualizations(model, datamodule, output_path: str, wandb_logger=None) -> None:
    """Emit final task galleries for the hypernetwork.

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
    val_hard_final_records = _select_hard_task_records(
        model,
        datamodule.val_dataloader(),
        limit=None,
        one_per_category=True,
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



# ---------------------------------------------------------------------------
# Embedding cluster maps (disentanglement diagnostic)
# ---------------------------------------------------------------------------


def log_embedding_cluster_plots(
    model,
    datamodule,
    output_path: str,
    wandb_logger=None,
    *,
    holdout_dataloader=None,
    holdout_label: str = "held-out",
) -> None:
    """Emit a PCA/t-SNE/UMAP cluster map of the pooled task latent, colored by task category.

    Validation-only, end-of-run diagnostic for whether the hypernetwork's pooled task
    representation is disentangled across task categories. Only runs for models exposing
    `supports_embedding_visualization` (the hypernetwork) with `log_embedding_clusters`
    explicitly opted in.

    Pass `holdout_dataloader` (e.g. a zero-shot compositional-generalisation eval set) to
    additionally overlay those embeddings on the same projection: reference validation points
    render fully opaque, drawn last (on top); holdout points render behind them at a
    higher-than-normal alpha (translucent but still legible), so it's visually clear where the
    model places examples it never trained on relative to the solid validation clusters (see
    visualisation.core.embedding_clusters.render_embedding_cluster_figure's `groups` argument).
    Omit it (default) for the original single-group behaviour, used by every other experiment.

    Logs the combined multi-panel (PCA + t-SNE + UMAP) figure to disk and to W&B as before,
    plus each available projection a second time as its own separate W&B image -- easier to
    inspect one projection at a time than a single wide combined panel.
    """
    try:
        trainer = model.trainer
    except RuntimeError:
        # LightningModule.trainer raises (not AttributeError) when unattached, so plain
        # getattr(..., None) doesn't catch it -- this path is hit by standalone eval scripts
        # (e.g. scripts/eval_compositional_holdout.py) that never call Trainer.fit().
        trainer = None
    if trainer is not None and not trainer.is_global_zero:
        return

    if not getattr(model, "supports_embedding_visualization", False):
        return

    if not getattr(model, "log_embedding_clusters", False):
        return

    records = model.collect_embedding_records_from_dataloader(datamodule.val_dataloader())
    if not records:
        return

    cluster_dir = os.path.join(output_path, "embedding_clusters")
    os.makedirs(cluster_dir, exist_ok=True)

    # Dump the raw (reference-only, no holdout) vectors alongside the figure -- this run's own
    # embeddings.npz, saved naturally as part of the run that computed them, no separate
    # checkpoint-reload step needed. Same schema legacy/scripts/dump_embedding_clusters.py
    # (only still useful for a run that predates this) produces, so every consumer
    # (visualisation.paper.plot_embedding_clusters et al.) reads either the same way.
    np.savez_compressed(
        os.path.join(cluster_dir, "embeddings.npz"),
        vectors=torch.stack([record["pooled_embedding"] for record in records]).numpy(),
        task_categories=np.array([record["task_category"] for record in records]),
        task_ids=np.array([record["task_id"] for record in records]),
    )

    groups: list[str] | None = None
    if holdout_dataloader is not None:
        holdout_records = model.collect_embedding_records_from_dataloader(holdout_dataloader)
        groups = ["reference"] * len(records) + [holdout_label] * len(holdout_records)
        records = records + holdout_records

    vectors = torch.stack([record["pooled_embedding"] for record in records]).numpy()
    task_categories = [record["task_category"] for record in records]

    title = "Pooled task latent (validation)"
    filename = "pooled_task_latent.png"
    group_styles = None
    if groups is not None:
        title = f"Pooled task latent (validation + {holdout_label})"
        filename = f"pooled_task_latent_with_{holdout_label.replace(' ', '_')}.png"
        group_styles = {"reference": DEFAULT_GROUP_STYLES["reference"], holdout_label: DEFAULT_GROUP_STYLES["new"]}

    figure = render_embedding_cluster_figure(
        vectors,
        task_categories,
        title=title,
        groups=groups,
        group_styles=group_styles,
    )
    figure.savefig(os.path.join(cluster_dir, filename), dpi=150, bbox_inches="tight")

    key_stem = Path(filename).stem
    wandb_payload = {f"embedding_clusters/{key_stem}": figure_to_wandb_image(figure, caption=title)}

    single_projection_figures = render_single_projection_figures(
        vectors, task_categories, groups=groups, group_styles=group_styles
    )
    for projection_name, single_figure in single_projection_figures.items():
        projection_key = projection_name.lower().replace("-", "")  # "t-SNE" -> "tsne"
        wandb_payload[f"embedding_clusters/{key_stem}_{projection_key}"] = figure_to_wandb_image(
            single_figure, caption=f"{title} — {projection_name}"
        )

    probe_message = ""
    try:
        probe = compute_linear_probe_accuracy(vectors, task_categories)
    except ValueError as error:
        print(f"Skipped linear probe accuracy: {error}")
    else:
        wandb_payload["embedding_clusters/linear_probe_accuracy"] = probe["mean_accuracy"]
        summary_path = os.path.join(cluster_dir, "linear_probe_summary.txt")
        with open(summary_path, "w") as summary_file:
            summary_file.write(_format_linear_probe_summary(probe))
        probe_message = f" | linear probe accuracy: {probe['mean_accuracy']:.3f}"

    log_wandb_payload(wandb_logger, wandb_payload)

    print(f"Logged pooled-task-latent embedding cluster map to {cluster_dir}{probe_message}")


def _format_linear_probe_summary(probe: dict[str, object]) -> str:
    """Return a concise human-readable summary for `linear_probe_summary.txt`."""
    fold_accuracies = ", ".join(f"{score:.4f}" for score in probe["fold_accuracies"])
    lines = [
        "Linear probe summary (task-category classification from pooled task latent)",
        f"n_classes: {probe['n_classes']}",
        f"n_splits: {probe['n_splits']}",
        f"mean_accuracy: {probe['mean_accuracy']:.6f}",
        f"fold_accuracies: [{fold_accuracies}]",
    ]
    return "\n".join(lines) + "\n"
