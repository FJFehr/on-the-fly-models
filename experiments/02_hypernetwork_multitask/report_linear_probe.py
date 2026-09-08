"""Linear-probe accuracy across seeds: mean +- 1 s.d.

The `mean_accuracy` in each run's embedding_clusters/linear_probe_summary.txt
(5-fold cross-validated task-category classification from the pooled task
latent -- how disentangled it is, see training.logging.log_embedding_cluster_plots)
is a single seed's number. This aggregates across every seed found for a
size/condition into mean +- population s.d., the same statistic quoted for
every other result in this experiment.

Usage
-----
    uv run python experiments/02_hypernetwork_multitask/report_linear_probe.py
    uv run python experiments/02_hypernetwork_multitask/report_linear_probe.py --dim 4
"""

import argparse
import statistics
from pathlib import Path

OUTPUTS_DIR = Path("outputs/02_hypernetwork_multitask")
CONDITIONS = ("frozentd", "notd")
SEEDS = (1, 2, 3, 4, 5)


def read_mean_accuracy(path: Path) -> float | None:
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        if line.startswith("mean_accuracy:"):
            return float(line.split(":", 1)[1].strip())
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=4, choices=[4, 6])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for cond in CONDITIONS:
        values = []
        missing = []
        for seed in SEEDS:
            run = f"hyper_multitask_dim{args.dim}_{cond}_seed{seed}"
            path = OUTPUTS_DIR / run / "embedding_clusters" / "linear_probe_summary.txt"
            acc = read_mean_accuracy(path)
            if acc is None:
                missing.append(seed)
            else:
                values.append(acc)

        label = "frozen_td" if cond == "frozentd" else cond
        if not values:
            print(f"{label:10s} dim={args.dim}: no linear_probe_summary.txt found for any seed")
            continue
        mean = statistics.mean(values)
        std = statistics.pstdev(values)
        per_seed = ", ".join(f"{v * 100:.1f}" for v in values)
        missing_note = f"  (missing seeds: {missing})" if missing else ""
        print(
            f"{label:10s} dim={args.dim}: mean={mean * 100:.2f}%  std={std * 100:.2f}pp  "
            f"n={len(values)}  [{per_seed}]{missing_note}"
        )


if __name__ == "__main__":
    main()
