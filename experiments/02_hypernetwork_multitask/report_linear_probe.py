"""Linear-probe + exact-match summary: mean +- 1 s.d. across seeds, and results.csv.

The `mean_accuracy` in each run's embedding_clusters/linear_probe_summary.txt
(5-fold cross-validated task-category classification from the pooled task
latent -- how disentangled it is, see training.logging.log_embedding_cluster_plots)
and `test_query_exact_match` in its results.txt are both single-seed numbers.
This scans every seed found for every dim present in outputs/, prints
mean +- population s.d. per (dim, condition) for both metrics, and refreshes
results.csv (one row per seed, the same columns it's always had --
condition, dim, params_hypernetwork, run_id, test_exact_match,
linear_probe_accuracy) so both live entirely from outputs/, not a
hand-maintained file. run_id is left blank on a rescan (wandb's own ID,
not reconstructable locally); everything else downstream only ever reads
the other five columns.

Only reflects whatever dim(s)/seed(s) actually exist under outputs/ --
dim=6 won't appear in a fresh scan unless dim=6 has been (re)run and its
outputs/ tree is present locally.

Usage
-----
    uv run python experiments/02_hypernetwork_multitask/report_linear_probe.py
    uv run python experiments/02_hypernetwork_multitask/report_linear_probe.py --dim 4
"""

import argparse
import csv
import statistics
from pathlib import Path

OUTPUTS_DIR = Path("outputs/02_hypernetwork_multitask")
CSV_PATH = Path("outputs/results/02_hypernetwork_multitask/results.csv")
CSV_FIELDS = [
    "condition", "dim", "params_hypernetwork", "run_id",
    "test_exact_match", "linear_probe_accuracy",
]
CONDITIONS = ("frozentd", "notd")
DIMS = (4, 6)
SEEDS = (1, 2, 3, 4, 5)

# Total params (trainable + the frozen task-indicator projection for frozentd), read
# directly from each condition's own model.txt -- see "Hypernetwork size" in this
# experiment's README. Fixed per (dim, condition), not something a rescan can
# recompute from results.txt/linear_probe_summary.txt. dim=4 is the matched-scale
# recipe (task_encoding.embedding_dim/hyper_model/hyper_head all sized to the dim=4
# target it generates -- 10,156 trainable params, identical for both conditions since
# the frozen projection isn't trainable; the 72-param notd/frozentd delta below is
# entirely that non-trainable projection). dim=6 is still the older, larger, fixed-width
# hypernetwork (not yet resized to match -- see README's "Open follow-ups").
PARAMS = {
    (4, "notd"): 11360,
    (4, "frozentd"): 11432,
    (6, "notd"): 336726,
    (6, "frozentd"): 337878,
}


def read_mean_accuracy(path: Path) -> float | None:
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        if line.startswith("mean_accuracy:"):
            return float(line.split(":", 1)[1].strip())
    return None


def parse_test_exact_match(results_path: Path) -> float | None:
    if not results_path.exists():
        return None
    in_test = False
    for line in results_path.read_text().splitlines():
        stripped = line.strip()
        if stripped == "test_metrics:":
            in_test = True
            continue
        if in_test and stripped.startswith("test_query_exact_match:"):
            return float(stripped.split(":", 1)[1].strip())
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dim", type=int, default=None, choices=[4, 6],
        help="Restrict to this size (default: report every dim actually found).",
    )
    return parser.parse_args()


def run(dims: list[int] | None = None) -> list[dict]:
    """Rescan outputs/, print the mean +- s.d. report, refresh results.csv,
    and return the rows written (possibly empty). dims=None auto-discovers
    every dim actually present."""
    dims = dims if dims is not None else list(DIMS)

    csv_rows = []
    for dim in dims:
        for cond in CONDITIONS:
            probe_values = []
            exact_match_values = []
            missing = []
            for seed in SEEDS:
                run_name = f"hyper_multitask_dim{dim}_{cond}_seed{seed}"
                run_dir = OUTPUTS_DIR / run_name
                probe_path = run_dir / "embedding_clusters" / "linear_probe_summary.txt"
                probe = read_mean_accuracy(probe_path)
                exact_match = parse_test_exact_match(run_dir / "results.txt")
                if probe is None and exact_match is None:
                    missing.append(seed)
                    continue
                if probe is not None:
                    probe_values.append(probe)
                if exact_match is not None:
                    exact_match_values.append(exact_match)
                csv_rows.append(
                    {
                        "condition": "frozen_td" if cond == "frozentd" else cond,
                        "dim": dim,
                        "params_hypernetwork": PARAMS[(dim, cond)],
                        "run_id": "",
                        "test_exact_match": exact_match if exact_match is not None else "",
                        "linear_probe_accuracy": probe if probe is not None else "",
                    }
                )

            label = "frozen_td" if cond == "frozentd" else cond
            if not probe_values and not exact_match_values:
                continue
            parts = [f"{label:10s} dim={dim}:"]
            metrics = (("exact_match", exact_match_values), ("linear_probe", probe_values))
            for metric_name, values in metrics:
                if values:
                    mean = statistics.mean(values)
                    std = statistics.pstdev(values)
                    per_seed = ", ".join(f"{v * 100:.1f}" for v in values)
                    parts.append(
                        f"{metric_name}: mean={mean * 100:.2f}% std={std * 100:.2f}pp "
                        f"n={len(values)} [{per_seed}]"
                    )
            if missing:
                parts.append(f"(missing seeds: {missing})")
            print("  ".join(parts))

    if csv_rows:
        CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nWrote {len(csv_rows)} rows to {CSV_PATH}")
    else:
        print("No results found under outputs/ -- results.csv left untouched.")
    return csv_rows


def main() -> None:
    args = parse_args()
    run([args.dim] if args.dim is not None else None)


if __name__ == "__main__":
    main()
