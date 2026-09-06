"""Preset: Task ID vs w/o Task ID paired cluster maps, dim=6 or dim=4 (Experiment 4).

Thin wrapper around visualisation.plot_embedding_clusters -- the actual
rendering logic (theme, palette, pairing layout) lives there and is shared
by every experiment; this just fills in this experiment's two .npz paths
and panel labels so it can be rerun here with no arguments beyond --dim.

Reads the raw pooled-embedding vectors dumped alongside training
(training.logging.log_embedding_cluster_plots, `log_embedding_clusters:
true`) or, for these specific historical runs, by the older
scripts/dump_embedding_clusters.py from checkpoints rerun with
save_checkpoints=true specifically so this never needs to retrain again --
rerun this script any time with no cluster access needed, ~15s locally.

dim=4's notd checkpoint (dim4_notd_ckpt_seed1: 47.1% test EM, notably lower
than the main grid's notd seeds' 67-74%) crashed mid-training on a
transient FileNotFoundError and resumed from its last checkpoint, which
likely disrupted the Muon LR schedule/optimizer state. Still usable for the
cluster map (checkpoint captured a real, if weaker, trained model), but
don't read dim=4's accuracy gap as representative of notd's typical
performance -- see results_per_task_dim4.csv for that.

Usage
-----
    uv run python experiments/02_hypernetwork_multitask/plot_embedding_clusters.py
    uv run python experiments/02_hypernetwork_multitask/plot_embedding_clusters.py --dim 4
    uv run python experiments/02_hypernetwork_multitask/plot_embedding_clusters.py \\
        --projections t-SNE   # only the pairing actually used in the paper
"""

import argparse
from pathlib import Path

from visualisation.plot_embedding_clusters import render_paired_cluster_maps

NPZ_DIR = Path("outputs/results/02_hypernetwork_multitask")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=6, choices=[4, 6])
    parser.add_argument(
        "--projections", nargs="+", default=None, choices=["PCA", "t-SNE", "UMAP"],
        help="Subset to render (default: all three available).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_paired_cluster_maps(
        left_npz=NPZ_DIR / f"embeddings_dim{args.dim}_frozentd.npz",
        left_label="Task ID",
        right_npz=NPZ_DIR / f"embeddings_dim{args.dim}_notd.npz",
        right_label="w/o Task ID",
        out_dir=Path("outputs/figures/02_hypernetwork_multitask"),
        out_prefix=f"cluster_dim{args.dim}_paired",
        projections=args.projections,
    )


if __name__ == "__main__":
    main()
