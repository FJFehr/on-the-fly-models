"""Preset: dim4 Task ID vs w/o Task ID paired cluster maps (Experiment 4).

Same pattern as plot_embedding_clusters.py (the dim6 preset) -- thin
wrapper around visualisation.plot_embedding_clusters's general tool, just
filling in this experiment's dim=4 .npz paths and panel labels.

Reads embeddings_dim4_frozentd.npz / embeddings_dim4_notd.npz, dumped via
scripts/dump_embedding_clusters.py from checkpoints rerun with
save_checkpoints=true specifically for this (dim4_frozentd_ckpt_seed1:
95.7% test EM; dim4_notd_ckpt_seed1: 47.1% test EM, notably lower than the
main grid's notd seeds (67-74%) -- that checkpoint's run crashed mid-training
on a transient FileNotFoundError and resumed from its last checkpoint, which
likely disrupted the Muon LR schedule/optimizer state. Still usable for the
cluster map (checkpoint captured a real, if weaker, trained model), but
don't read this pairing's accuracy gap as representative of dim4 notd's
typical performance -- see per_task_dim4.csv for that).

Usage
-----
    uv run python configs/experiments/arc1d_v2_hypernetwork_multitask/plot_embedding_clusters_dim4.py
    uv run python configs/experiments/arc1d_v2_hypernetwork_multitask/plot_embedding_clusters_dim4.py \\
        --projections t-SNE
"""

import argparse
from pathlib import Path

from visualisation.plot_embedding_clusters import render_paired_cluster_maps

HERE = Path(__file__).parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--projections", nargs="+", default=None, choices=["PCA", "t-SNE", "UMAP"],
        help="Subset to render (default: all three available).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_paired_cluster_maps(
        left_npz=HERE / "embeddings_dim4_frozentd.npz",
        left_label="Task ID",
        right_npz=HERE / "embeddings_dim4_notd.npz",
        right_label="w/o Task ID",
        out_dir=HERE,
        out_prefix="cluster_dim4_paired",
        projections=args.projections,
    )


if __name__ == "__main__":
    main()
