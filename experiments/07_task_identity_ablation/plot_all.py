"""Experiment 7: aggregate every arm's results into CSVs, summary tables and figures.

Reads (whatever exists; missing runs are skipped and counted):
  in-distribution   INDIST_RUNS results.txt (test query metrics) and model.txt (parameters)
  leave-one-out     common.loo_results_path: held-out category, zero task vector for every arm
  compositional     outputs/07_task_identity_ablation/compositional/*/results.txt
  clusters          outputs/07_task_identity_ablation/representations/ (dump_representations.py)

Writes outputs/results/07_task_identity_ablation/*.csv and
outputs/figures/07_task_identity_ablation/:
  loo_per_category.png        held-out token accuracy per category, all 5 arms
  clusters_<space>_tsne.png   t-SNE of one seed's representations, one panel per arm
  task_embedding_similarity.png  cosine similarity of each learned task-embedding table

Every summary table has the trainable and total parameter columns; differences are judged
against the seed s.d. (see the README's hypotheses).

Usage:
    PYTHONPATH=. uv run python experiments/07_task_identity_ablation/plot_all.py
    PYTHONPATH=. uv run python experiments/07_task_identity_ablation/plot_all.py --cluster-seed 2
"""

import argparse
import csv
import statistics
from pathlib import Path

import numpy as np
from common import (
    ALL_CATEGORIES,
    ARM_LABELS,
    ARMS,
    EXPERIMENT,
    INDIST_RUNS,
    SEEDS,
    TD_ARMS,
    loo_results_path,
    parse_results,
    read_parameter_counts,
)
from matplotlib import pyplot as plt

from models.embedding import TASK_CATEGORY_INDEX
from visualisation.core.embedding_clusters import compute_tsne_2d
from visualisation.core.style import apply_latex_style, format_task_category
from visualisation.paper.arc_paper import PAPER_COLORS, lighten
from visualisation.paper.plot_embedding_clusters import MARKER_STYLE, build_display_categories

RESULTS_DIR = Path(f"outputs/results/{EXPERIMENT}")
FIGURES_DIR = Path(f"outputs/figures/{EXPERIMENT}")
OUTPUTS_DIR = Path(f"outputs/{EXPERIMENT}")
SPACES = ("support", "support_no_id", "task", "weights")
# notd and frozentd_latent keep experiment 5's colours.
COLORS = {
    "notd": PAPER_COLORS[5],
    "frozentd_latent": PAPER_COLORS[9],
    "learnedtd_latent": PAPER_COLORS[8],
    "frozentd_input": PAPER_COLORS[2],
    "learnedtd_input": PAPER_COLORS[1],
}


def mean_sd(values: list[float]) -> str:
    if not values:
        return "n/a"
    return f"{statistics.mean(values):.3f} ± {statistics.pstdev(values):.3f} (n={len(values)})"


def write_csv(rows: list[dict], name: str) -> None:
    if not rows:
        print(f"No rows for {name}")
        return
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / name, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {RESULTS_DIR / name}")


def parameters_by_arm() -> dict[str, dict[str, int]]:
    params = {}
    for arm in ARMS:
        for seed in SEEDS:
            run_dir = Path(INDIST_RUNS[arm].format(seed=seed))
            if (run_dir / "model.txt").exists():
                params[arm] = read_parameter_counts(run_dir)
                break
    return params


def params_text(params: dict, arm: str) -> str:
    p = params.get(arm, {})
    return f"{p.get('params_trainable', '?'):>6} / {p.get('params_total', '?'):>6}"


def indist(params: dict) -> None:
    rows = []
    for arm in ARMS:
        for seed in SEEDS:
            path = Path(INDIST_RUNS[arm].format(seed=seed)) / "results.txt"
            if path.exists():
                r = parse_results(path)
                rows.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        **params.get(arm, {}),
                        "test_query_accuracy": r["test_query_accuracy"],
                        "test_query_exact_match": r["test_query_exact_match"],
                    }
                )
    write_csv(rows, "indist.csv")
    print("\n== In-distribution (test query), mean ± s.d. over seeds; params trainable / total ==")
    for arm in ARMS:
        acc = [r["test_query_accuracy"] for r in rows if r["arm"] == arm]
        em = [r["test_query_exact_match"] for r in rows if r["arm"] == arm]
        label = f"{ARM_LABELS[arm]:16s} {params_text(params, arm)}"
        print(f"  {label}  acc {mean_sd(acc)}  EM {mean_sd(em)}")


def leave_one_out(params: dict) -> None:
    rows = []
    for arm in ARMS:
        for held_out in ALL_CATEGORIES:
            for seed in SEEDS:
                path = loo_results_path(arm, held_out, seed)
                if not path.exists():
                    continue
                r = parse_results(path)
                others = [c for c in ALL_CATEGORIES if c != held_out]
                rows.append(
                    {
                        "arm": arm,
                        "category": held_out,
                        "seed": seed,
                        "holdout_accuracy": r[f"val_query_accuracy_by_task_{held_out}"],
                        "holdout_exact_match": r[f"val_query_exact_match_by_task_{held_out}"],
                        "indist_accuracy": statistics.mean(
                            r[f"val_query_accuracy_by_task_{c}"] for c in others
                        ),
                    }
                )
    write_csv(rows, "leave_one_out.csv")

    # Per-category seed means, then the macro mean over categories (as experiment 5).
    means = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for category in ALL_CATEGORIES:
            values = [r for r in rows if r["arm"] == arm and r["category"] == category]
            if values:
                means[arm][category] = {
                    key: statistics.mean(v[key] for v in values)
                    for key in ("holdout_accuracy", "holdout_exact_match", "indist_accuracy")
                } | {"sd": statistics.pstdev(v["holdout_accuracy"] for v in values)}

    print(
        "\n== Leave-one-out, zero task vector on the held-out category (macro over categories) =="
    )
    for arm in ARMS:
        cats = means[arm].values()
        if not cats:
            print(f"  {ARM_LABELS[arm]:16s} no runs")
            continue
        held = statistics.mean(c["holdout_accuracy"] for c in cats)
        em = statistics.mean(c["holdout_exact_match"] for c in cats)
        ind = statistics.mean(c["indist_accuracy"] for c in cats)
        print(
            f"  {ARM_LABELS[arm]:16s} {params_text(params, arm)}  held-out acc {held:.3f}  "
            f"EM {em:.4f}  in-dist acc {ind:.3f}  drop {ind - held:.3f}  ({len(cats)} categories)"
        )

    print("  Paired per-category comparisons (held-out token accuracy, seed means):")
    for a, b in (
        ("learnedtd_latent", "frozentd_latent"),
        ("frozentd_input", "frozentd_latent"),
        ("learnedtd_input", "frozentd_input"),
        *((arm, "notd") for arm in TD_ARMS),
    ):
        shared = [c for c in ALL_CATEGORIES if c in means[a] and c in means[b]]
        if shared:
            ahead = sum(
                means[a][c]["holdout_accuracy"] > means[b][c]["holdout_accuracy"] for c in shared
            )
            print(f"    {ARM_LABELS[a]} ahead of {ARM_LABELS[b]} in {ahead} of {len(shared)}")
    plot_loo(means)


def plot_loo(means: dict) -> None:
    apply_latex_style()
    categories = [c for c in ALL_CATEGORIES if any(c in means[arm] for arm in ARMS)]
    if not categories:
        return
    x = np.arange(len(categories))
    width = 0.8 / len(ARMS)
    fig, ax = plt.subplots(figsize=(15.0, 5.0))
    for i, arm in enumerate(ARMS):
        offset = (i - (len(ARMS) - 1) / 2) * width
        values = np.array(
            [
                means[arm].get(c, {"holdout_accuracy": np.nan})["holdout_accuracy"]
                for c in categories
            ]
        )
        sds = np.array([means[arm].get(c, {"sd": 0})["sd"] for c in categories])
        ax.bar(
            x + offset,
            values,
            width=width,
            color=lighten(COLORS[arm]),
            edgecolor=COLORS[arm],
            linewidth=1.2,
            label=ARM_LABELS[arm],
            zorder=3,
        )
        ax.errorbar(
            x + offset,
            values,
            yerr=sds,
            fmt="none",
            ecolor="0.25",
            elinewidth=0.9,
            capsize=2,
            zorder=4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([format_task_category(c) for c in categories], rotation=40, ha="right")
    ax.set_ylabel("Held-out zero-shot token accuracy")
    ax.set_ylim(0, 1.04)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="y", alpha=0.3, linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(ARMS), frameon=False)
    save(fig, "loo_per_category")


def compositional(params: dict) -> None:
    rows = []
    for path in sorted((OUTPUTS_DIR / "compositional").glob("*/results.txt")):
        name = path.parent.name  # <arm>_seed<N> or <arm>_<mode>_seed<N>
        stem, _, seed = name.rpartition("_seed")
        arm = next(a for a in sorted(ARMS, key=len, reverse=True) if stem.startswith(a))
        mode = stem[len(arm) + 1 :] or "none"
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) == 4 and parts[0].startswith("1d_comp_"):
                rows.append(
                    {
                        "arm": arm,
                        "mode": mode,
                        "seed": int(seed),
                        "category": parts[0],
                        "exact_match": float(parts[2]),
                        "accuracy": float(parts[3]),
                    }
                )
    write_csv(rows, "compositional.csv")
    print("\n== Compositional (macro over composite categories) ==")
    for arm in ARMS:
        for mode in sorted({r["mode"] for r in rows if r["arm"] == arm}):
            per_seed = {}
            for r in rows:
                if r["arm"] == arm and r["mode"] == mode:
                    per_seed.setdefault(r["seed"], []).append(r)
            acc = [statistics.mean(r["accuracy"] for r in v) for v in per_seed.values()]
            em = [statistics.mean(r["exact_match"] for r in v) for v in per_seed.values()]
            print(
                f"  {ARM_LABELS[arm]:16s} {mode:9s} {params_text(params, arm)}  "
                f"acc {mean_sd(acc)}  EM {mean_sd(em)}"
            )


def clusters(cluster_seed: int) -> None:
    metrics_path = OUTPUTS_DIR / "representations" / "cluster_metrics.csv"
    if not metrics_path.exists():
        print("\nNo cluster metrics yet (run dump_representations.py)")
        return
    with open(metrics_path) as f:
        rows = list(csv.DictReader(f))
    print("\n== Cluster metrics by space: linear probe | silhouette, mean ± s.d. over seeds ==")
    for space in SPACES:
        print(f"  {space}")
        for arm in ARMS:
            arm_rows = [r for r in rows if r["arm"] == arm]
            probe = [float(r[f"{space}_linear_probe"]) for r in arm_rows]
            sil = [float(r[f"{space}_silhouette"]) for r in arm_rows]
            print(f"    {ARM_LABELS[arm]:16s} probe {mean_sd(probe)} | silhouette {mean_sd(sil)}")

    data = {}
    for arm in ARMS:
        path = OUTPUTS_DIR / "representations" / f"{arm}_seed{cluster_seed}.npz"
        if path.exists():
            data[arm] = np.load(path, allow_pickle=True)
    if not data:
        return
    for space in SPACES:
        plot_cluster_grid(data, space, cluster_seed)
    plot_similarity(data, cluster_seed)


def plot_cluster_grid(data: dict, space: str, seed: int) -> None:
    """t-SNE of the non-augmented originals (5 per category), one panel per arm."""
    apply_latex_style()
    arms = list(data)
    fig, axes = plt.subplots(1, len(arms), figsize=(4.2 * len(arms), 4.6), squeeze=False)
    handles = {}
    for ax, arm in zip(axes[0], arms, strict=True):
        keep = data[arm]["task_ids"] % 10000 == 0
        categories = list(data[arm]["task_categories"][keep])
        coords = compute_tsne_2d(data[arm][space][keep])
        display, unique, colours = build_display_categories(categories)
        for category in unique:
            idx = [i for i, d in enumerate(display) if d == category]
            handles[category] = ax.scatter(
                coords[idx, 0],
                coords[idx, 1],
                color=colours[category],
                s=MARKER_STYLE["s"] / 3,
                alpha=MARKER_STYLE["alpha"],
            )
        ax.set_title(ARM_LABELS[arm])
        ax.set_xticks([])
        ax.set_yticks([])
    fig.legend(
        handles.values(),
        handles.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=7,
        frameon=False,
    )
    fig.tight_layout()
    save(fig, f"clusters_{space}_tsne_seed{seed}")


def plot_similarity(data: dict, seed: int) -> None:
    """Cosine similarity between the trained categories' task embeddings, per td arm."""
    arms = [arm for arm in TD_ARMS if arm in data and "task_embedding_table" in data[arm]]
    if not arms:
        return
    apply_latex_style()
    labels = [format_task_category(c) for c in ALL_CATEGORIES]
    rows = [TASK_CATEGORY_INDEX[c] for c in ALL_CATEGORIES]
    fig, axes = plt.subplots(1, len(arms), figsize=(5.0 * len(arms), 5.0), squeeze=False)
    for ax, arm in zip(axes[0], arms, strict=True):
        table = data[arm]["task_embedding_table"][rows]
        unit = table / np.linalg.norm(table, axis=1, keepdims=True)
        image = ax.imshow(unit @ unit.T, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_title(ARM_LABELS[arm])
        ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
        ax.set_yticks(range(len(labels)), labels if ax is axes[0][0] else [], fontsize=7)
    fig.colorbar(image, ax=axes[0].tolist(), shrink=0.8, label="cosine similarity")
    save(fig, f"task_embedding_similarity_seed{seed}")


def save(fig: plt.Figure, stem: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(FIGURES_DIR / f"{stem}{suffix}", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIGURES_DIR / stem}.png / .pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cluster-seed", type=int, default=1, help="Seed shown in the cluster figures."
    )
    args = parser.parse_args()
    params = parameters_by_arm()
    indist(params)
    compositional(params)
    leave_one_out(params)
    clusters(args.cluster_seed)


if __name__ == "__main__":
    main()
