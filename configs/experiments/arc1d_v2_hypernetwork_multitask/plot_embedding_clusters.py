"""Preset: dim6 Task ID vs No task ID paired cluster maps (Experiment 4).

Thin wrapper around visualisation.plot_embedding_clusters -- the actual
rendering logic (theme, palette, pairing layout) lives there and is shared
by every experiment; this just fills in this experiment's two .npz paths
and panel labels so it can be rerun here with no arguments.

Reads the raw pooled-embedding vectors dumped by scripts/dump_embedding_clusters.py
(embeddings_dim6_frozentd.npz / embeddings_dim6_notd.npz, next to this
script). Those were themselves dumped from checkpoints rerun with
save_checkpoints=true specifically so this never needs to retrain again --
rerun this script any time with no cluster access needed, ~15s locally.

Usage
-----
    uv run python configs/experiments/arc1d_v2_hypernetwork_multitask/plot_embedding_clusters.py
    uv run python configs/experiments/arc1d_v2_hypernetwork_multitask/plot_embedding_clusters.py \\
        --projections t-SNE   # only the pairing actually used in the paper
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
        left_npz=HERE / "embeddings_dim6_frozentd.npz",
        left_label="Task ID",
        right_npz=HERE / "embeddings_dim6_notd.npz",
        right_label="No task ID",
        out_dir=HERE,
        out_prefix="cluster_dim6_paired",
        projections=args.projections,
    )


if __name__ == "__main__":
    main()
