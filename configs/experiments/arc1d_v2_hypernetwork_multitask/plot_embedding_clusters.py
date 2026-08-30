"""Themed, individual cluster-map plots for the joint hypernetwork's pooled
task latent (Experiment 4).

Reads the raw pooled-embedding vectors dumped by scripts/dump_embedding_clusters.py
(one .npz per condition, e.g. embeddings_dim6_notd.npz) and renders each
available projection (PCA / t-SNE / UMAP) as its own standalone figure,
styled to match plot_capacity_cliff.py / plot_per_task.py (Nimbus Roman,
larger labels, PNG+PDF). Reuses
visualisation.embedding_clusters.render_single_projection_figures for the
actual projection + scatter logic rather than reimplementing it -- this
script only supplies the theme and the file-splitting.

Why a checkpoint dump instead of just re-plotting from the original PNG:
save_checkpoints was false for the original run, so the raw vectors that
produced cluster_dim6_notd.png/cluster_dim6_frozentd.png were never saved --
only rendered once into that PNG. dim6_notd and dim6_frozentd were rerun
once with save_checkpoints=true specifically so this script (and any future
restyle) never needs to retrain again.

Usage
-----
    uv run python configs/experiments/arc1d_v2_hypernetwork_multitask/plot_embedding_clusters.py
    uv run python configs/experiments/arc1d_v2_hypernetwork_multitask/plot_embedding_clusters.py \\
        --projections t-SNE   # only render the ones actually used in the paper
"""

import argparse
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from visualisation.embedding_clusters import render_single_projection_figures
from visualisation.style import apply_latex_style

HERE = Path(__file__).parent
CONDITIONS = {
    "notd": "embeddings_dim6_notd.npz",
    "frozentd": "embeddings_dim6_frozentd.npz",
}


def apply_theme() -> None:
    apply_latex_style()
    plt.rcParams.update(
        {
            "font.serif": ["Nimbus Roman", "Times New Roman", "Liberation Serif", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.titlesize": 19,
            "axes.labelsize": 17,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
            "legend.title_fontsize": 14,
            "pdf.fonttype": 42,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--projections",
        nargs="+",
        default=None,
        choices=["PCA", "t-SNE", "UMAP"],
        help="Subset to render (default: all three available).",
    )
    parser.add_argument(
        "--conditions", nargs="+", default=None, choices=list(CONDITIONS),
        help="Subset to render (default: all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_theme()

    conditions = args.conditions or list(CONDITIONS)
    for condition in conditions:
        npz_path = HERE / CONDITIONS[condition]
        if not npz_path.exists():
            print(f"Skipping {condition}: {npz_path} not found (run dump_embedding_clusters.py).")
            continue

        data = np.load(npz_path, allow_pickle=True)
        vectors, task_categories = data["vectors"], list(data["task_categories"])

        figures = render_single_projection_figures(vectors, task_categories)
        for name, figure in figures.items():
            if args.projections and name not in args.projections:
                continue
            projection_key = name.lower().replace("-", "")  # "t-SNE" -> "tsne"
            out_stem = HERE / f"cluster_dim6_{condition}_{projection_key}"
            figure.savefig(out_stem.with_suffix(".png"))
            figure.savefig(out_stem.with_suffix(".pdf"))
            print(f"Saved {out_stem}.png / .pdf")


if __name__ == "__main__":
    main()
