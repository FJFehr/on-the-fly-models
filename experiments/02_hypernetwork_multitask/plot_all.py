"""Regenerate every figure and report for experiment 2 (hypernetwork) in one pass.

The single entry point to run after a training sweep finishes (see run.sh):
rescans outputs/, refreshes results_per_task_dim4_combined.csv and results.csv,
prints the linear-probe/exact-match summary, and renders the per-task figures
(5-condition combined, and the headline 3-condition version) plus every seed's
cluster maps (PCA/t-SNE/UMAP, --originals-only to match the paper's 5/category
look). No need to remember four separate scripts' own invocations.

Cluster maps read straight from each seed's own outputs/.../embedding_clusters/
embeddings.npz (plot_embedding_clusters.py --all-seeds) -- always live, no
--from-csv equivalent for those since there's no intermediate CSV to skip past.

Usage
-----
    uv run python experiments/02_hypernetwork_multitask/plot_all.py
    # rescan, refresh, plot

    uv run python experiments/02_hypernetwork_multitask/plot_all.py --from-csv
    # replot per-task from the CSV; linear-probe report and cluster maps still rescan

    uv run python experiments/02_hypernetwork_multitask/plot_all.py --skip-clusters
    # skip the (slower) t-SNE/UMAP cluster maps
"""

import argparse
from pathlib import Path

import plot_embedding_clusters as clusters
import plot_per_task as per_task
import plot_per_task_combined as per_task_combined
import report_linear_probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Rescan this dir and refresh the per-task CSV before plotting (default: outputs).",
    )
    parser.add_argument(
        "--from-csv",
        action="store_true",
        help="Skip the per-task rescan; replot from results_per_task_dim4_combined.csv as "
        "committed. The linear-probe report and cluster maps still rescan outputs/ (they "
        "have no CSV to fall back to).",
    )
    parser.add_argument(
        "--skip-clusters",
        action="store_true",
        help="Skip cluster-map rendering (PCA/t-SNE/UMAP x 5 seeds is the slow part).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("== per-task breakdown, dim=4 (individual, joint-direct, hypernetwork) ==")
    if args.from_csv:
        task_records = per_task_combined.read_csv(per_task_combined.CSV_PATH)
    else:
        task_records = per_task_combined.extract_records(args.outputs_dir)
        per_task_combined.write_csv(task_records, per_task_combined.CSV_PATH)
    figures_dir = Path("outputs/figures/02_hypernetwork_multitask")
    combined_series = per_task_combined.build_series(task_records)
    per_task_combined.plot(combined_series, figures_dir / "per_task_dim4_combined.png")
    headline_records = [r for r in task_records if r["condition"] in per_task.CONDITIONS]
    headline_series = per_task.build_series(headline_records)
    per_task.plot(headline_series, figures_dir / "per_task_dim4.png")

    print("\n== linear-probe accuracy + test exact match, mean +- 1 s.d. across seeds ==")
    report_linear_probe.run()

    if args.skip_clusters:
        print("\n--skip-clusters: leaving cluster maps as they are.")
    else:
        print("\n== cluster maps (PCA/t-SNE/UMAP), every seed, real 5/category originals ==")
        for seed in clusters.SEEDS:
            print(f"  -- seed {seed} --")
            clusters.render_one(
                dim=4, seed=seed, out_dir=clusters.FIGURES_DIR / f"seed{seed}",
                projections=None, max_per_category=None, originals_only=True,
            )

    print("\nDone. Figures in outputs/figures/02_hypernetwork_multitask/")
    print(f"CSVs in {per_task_combined.CSV_PATH.parent}")


if __name__ == "__main__":
    main()
