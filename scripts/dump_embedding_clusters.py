"""Dump the pooled task-latent embeddings from a trained checkpoint to disk.

Loads a trained hypernetwork checkpoint and runs the same
`collect_embedding_records_from_dataloader` pass that
`training.logging.log_embedding_cluster_plots` uses internally to build its
end-of-run cluster-map PNG -- except here the raw vectors are saved to a
.npz instead of (only) being rendered into a figure. Once dumped, cluster
maps can be re-plotted in any style, split any way, as many times as
needed, without retraining or even reloading the checkpoint again -- pair
two dumps with visualisation/plot_embedding_clusters.py, the default tool
for this (see its docstring), for the usual "compare two runs" figure.

Requires `save_checkpoints: true` for the run being loaded (the default in
most of this repo's configs is false -- see the leaf config actually used,
e.g. configs/experiments/arc1d_v2_hypernetwork_multitask/dim6_notd.yaml via
its base.yaml, was rerun with `save_checkpoints=true` specifically to make
this possible).

Usage
-----
Run as a module (not `python scripts/dump_embedding_clusters.py` directly --
that leaves the repo root off sys.path and data_modules/models fail to
import; same pre-existing quirk affects scripts/eval_compositional_holdout.py):

    uv run python -m scripts.dump_embedding_clusters \\
        --config configs/experiments/arc1d_v2_hypernetwork_multitask/dim6_notd.yaml \\
        --experiment-name v2_hyper_multitask_dim6_notd_ckpt_seed1 \\
        --out configs/experiments/arc1d_v2_hypernetwork_multitask/embeddings_dim6_notd.npz
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from data_modules import DATA_REGISTRY
from models import MODEL_REGISTRY
from training.config import build_runtime_config_dict, load_config
from training.trainer import load_checkpoint_state, resolve_evaluation_checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the trained leaf config.")
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Override experiment_name (e.g. to point at a *_ckpt_seed1 rerun without "
        "editing the leaf config's own experiment_name).",
    )
    parser.add_argument(
        "--checkpoint", default="best", help="'best', 'last', 'auto', or an explicit path.",
    )
    parser.add_argument("--out", required=True, type=Path, help="Output .npz path.")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = [f"experiment_name={args.experiment_name}"] if args.experiment_name else None
    cfg = load_config(args.config, overrides=overrides)
    runtime_cfg = build_runtime_config_dict(cfg)

    model = MODEL_REGISTRY[cfg.model](**runtime_cfg)
    checkpoint_path = resolve_evaluation_checkpoint_path(cfg.output_path, args.checkpoint)
    load_checkpoint_state(model, checkpoint_path)
    model.to(args.device)
    model.eval()

    datamodule = DATA_REGISTRY[cfg.data](**runtime_cfg)
    datamodule.setup()

    records = model.collect_embedding_records_from_dataloader(datamodule.val_dataloader())
    if not records:
        msg = "No embedding records collected -- val_dataloader() returned nothing."
        raise RuntimeError(msg)

    vectors = torch.stack([r["pooled_embedding"] for r in records]).numpy()
    task_categories = np.array([r["task_category"] for r in records])
    task_ids = np.array([r["task_id"] for r in records])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, vectors=vectors, task_categories=task_categories, task_ids=task_ids,
    )
    print(
        f"Saved {vectors.shape[0]} records (dim={vectors.shape[1]}) from {checkpoint_path} "
        f"to {args.out}"
    )


if __name__ == "__main__":
    main()
