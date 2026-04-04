"""Offline task-attention visualiser for hypernetwork checkpoints."""

import argparse
import os

import lightning as pl
from omegaconf import OmegaConf

from training.config import apply_grouped_config_aliases, build_runtime_config_dict
from data_modules import DATA_REGISTRY
from models import MODEL_REGISTRY
from visualisation import render_task_attention_figure, resolve_attention_matrix


def resolve_checkpoint_path(output_path: str, checkpoint_path: str | None) -> str:
    """Resolve the checkpoint path from the CLI or the standard output directory."""
    if checkpoint_path is not None:
        if not os.path.exists(checkpoint_path):
            msg = f"Checkpoint path {checkpoint_path!r} does not exist."
            raise FileNotFoundError(msg)
        return checkpoint_path

    candidate_paths = [
        os.path.join(output_path, "best_model.ckpt"),
        os.path.join(output_path, "last.ckpt"),
    ]
    for candidate_path in candidate_paths:
        if os.path.exists(candidate_path):
            return candidate_path

    msg = (
        "Could not find a checkpoint automatically. Pass --checkpoint-path or ensure "
        "best_model.ckpt or last.ckpt exists in the experiment output directory."
    )
    raise FileNotFoundError(msg)


def get_split_dataloader(datamodule, split_name: str):
    """Return the requested dataloader from the configured datamodule."""
    if split_name == "train":
        return datamodule.train_dataloader()
    if split_name == "val":
        return datamodule.val_dataloader()
    if split_name == "test":
        return datamodule.test_dataloader()

    msg = f"Unsupported split {split_name!r}. Expected one of 'train', 'val', or 'test'."
    raise ValueError(msg)


def main(cli_args) -> None:
    cfg = OmegaConf.load(cli_args.config)
    apply_grouped_config_aliases(cfg)
    cfg_dict = build_runtime_config_dict(cfg)
    pl.seed_everything(cfg_dict["seed"])

    checkpoint_path = resolve_checkpoint_path(cfg.output_path, cli_args.checkpoint_path)
    model_cls = MODEL_REGISTRY[cfg.model]
    model = model_cls.load_from_checkpoint(checkpoint_path, map_location="cpu", **cfg_dict)
    model.eval()

    datamodule_cls = DATA_REGISTRY[cfg.data]
    datamodule = datamodule_cls(**cfg_dict)
    datamodule.setup()
    dataloader = get_split_dataloader(datamodule, cli_args.split)
    batch = next(iter(dataloader))

    if cli_args.batch_item < 0 or cli_args.batch_item >= batch["support_inputs"].shape[0]:
        msg = (
            f"batch_item must be in [0, {batch['support_inputs'].shape[0] - 1}] but got "
            f"{cli_args.batch_item}."
        )
        raise ValueError(msg)

    _, predictions, targets = model.predict_batch(batch)
    task_record = model.build_task_records(batch, predictions.cpu(), targets.cpu())[
        cli_args.batch_item
    ]
    attention_record = model.build_task_attention_record(task_record)

    layer_indices = (
        range(len(attention_record["attentions"]))
        if cli_args.layer_index is None
        else [cli_args.layer_index]
    )

    output_dir = cli_args.output_dir or os.path.join(
        cfg.output_path, "attention_plots", cli_args.split
    )
    os.makedirs(output_dir, exist_ok=True)

    for layer_index in layer_indices:
        if layer_index < 0 or layer_index >= len(attention_record["attentions"]):
            msg = (
                f"layer_index must be in [0, {len(attention_record['attentions']) - 1}] but "
                f"got {layer_index}."
            )
            raise ValueError(msg)

        attention_matrix, head_label = resolve_attention_matrix(
            attention_record["attentions"][layer_index],
            reduction=cli_args.head_reduction,
            head_index=cli_args.head_index,
        )
        figure = render_task_attention_figure(
            attention_matrix=attention_matrix,
            token_labels=attention_record["token_labels"],
            token_metadata=attention_record["token_metadata"],
            task_category=attention_record["task_category"],
            task_id=attention_record["task_id"],
            layer_index=layer_index,
            head_label=head_label,
            query_exact_match=attention_record.get("query_exact_match"),
            query_accuracy=attention_record.get("query_accuracy"),
        )
        output_path = os.path.join(
            output_dir,
            f"{cli_args.split}_{attention_record['task_category']}_"
            f"{attention_record['task_id']}_layer{layer_index}_{head_label}.png",
        )
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--checkpoint-path", default=None, type=str)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--batch-item", default=0, type=int)
    parser.add_argument("--layer-index", default=None, type=int)
    parser.add_argument("--head-reduction", default="mean", choices=["mean", "max"])
    parser.add_argument("--head-index", default=None, type=int)
    parser.add_argument("--output-dir", default=None, type=str)
    main(parser.parse_args())
