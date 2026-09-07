"""Preset: Task ID vs w/o Task ID paired cluster maps, dim=6 or dim=4.

Thin wrapper around visualisation.paper.plot_embedding_clusters -- the actual
rendering logic (theme, palette, pairing layout) lives there and is shared
by every experiment; this just fills in this experiment's two .npz paths
and panel labels so it can be rerun here with no arguments beyond --dim.

Reads the raw pooled-embedding vectors dumped alongside training
(training.logging.log_embedding_cluster_plots, `log_embedding_clusters:
true` -- the default in this experiment's base.yaml). That dump happens
automatically at the end of every run, per seed, at
outputs/02_hypernetwork_multitask/<run>_seed<N>/embedding_clusters/
embeddings.npz -- no checkpoint reload needed, no separate dump step.
--seed reads directly from there; without it, this falls back to the flat,
non-seed-specific .npz files under outputs/results/02_hypernetwork_multitask/
(a single representative seed someone chose and copied there by hand,
kept for the paper figure's reproducibility without needing live outputs/).

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
        --dim 4 --seed 3   # this seed's own live outputs/ dump, not the flat file
    uv run python experiments/02_hypernetwork_multitask/plot_embedding_clusters.py \\
        --projections t-SNE   # only the pairing actually used in the paper
"""

import argparse
from pathlib import Path

from visualisation.paper.plot_embedding_clusters import render_paired_cluster_maps

NPZ_DIR = Path("outputs/results/02_hypernetwork_multitask")
OUTPUTS_DIR = Path("outputs/02_hypernetwork_multitask")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=6, choices=[4, 6])
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Read this seed's own embeddings.npz straight from outputs/ instead of "
        "the flat, pre-picked file under outputs/results/. Requires that seed's run "
        "directory to still exist (fetch it from the cluster node first if not).",
    )
    parser.add_argument(
        "--projections", nargs="+", default=None, choices=["PCA", "t-SNE", "UMAP"],
        help="Subset to render (default: all three available).",
    )
    return parser.parse_args()


def _npz_path(dim: int, cond: str, seed: int | None) -> Path:
    if seed is None:
        return NPZ_DIR / f"embeddings_dim{dim}_{cond}.npz"
    run = f"hyper_multitask_dim{dim}_{cond}_seed{seed}"
    return OUTPUTS_DIR / run / "embedding_clusters" / "embeddings.npz"


def main() -> None:
    args = parse_args()
    out_prefix = f"cluster_dim{args.dim}_paired"
    if args.seed is not None:
        out_prefix += f"_seed{args.seed}"
    render_paired_cluster_maps(
        left_npz=_npz_path(args.dim, "frozentd", args.seed),
        left_label="Task ID",
        right_npz=_npz_path(args.dim, "notd", args.seed),
        right_label="w/o Task ID",
        out_dir=Path("outputs/figures/02_hypernetwork_multitask"),
        out_prefix=out_prefix,
        projections=args.projections,
    )


if __name__ == "__main__":
    main()
