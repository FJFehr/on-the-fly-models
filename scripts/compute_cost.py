"""Report the GPU compute each experiment used: minutes per run and total GPU-hours.

Every training run writes compute.json (GPU model and wall-clock minutes, see
training.logging.write_compute_record). Runs from before that file existed fall back to their
file timestamps: from config.yaml (written when the run starts) to results.txt (written when it
ends). Each run uses one GPU, so GPU-hours are the summed wall-clock hours.

    python scripts/compute_cost.py                       # every experiment under outputs/
    python scripts/compute_cost.py 02_hypernetwork_multitask --by-cell
    python scripts/compute_cost.py --outputs-dir /other/outputs --csv cost.csv
"""

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


def run_cost(run_dir: Path) -> tuple[float, str] | None:
    """(wall-clock minutes, GPU model) of one finished run, or None if it did not finish."""
    results = run_dir / "results.txt"
    if not results.exists():
        return None
    record = run_dir / "compute.json"
    if record.exists():
        data = json.loads(record.read_text())
        return float(data["wall_clock_minutes"]), data.get("gpu", "unknown")
    config = run_dir / "config.yaml"
    if not config.exists():
        return None
    minutes = (results.stat().st_mtime - config.stat().st_mtime) / 60
    return minutes, "not recorded (from file times)"


def cell_name(run_name: str) -> str:
    """The run name without its seed, e.g. joint_td_dim4_seed3 -> joint_td_dim4."""
    return re.sub(r"_seed\d+$", "", run_name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "experiments", nargs="*", help="folders under the outputs directory (default: all)"
    )
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--by-cell", action="store_true", help="one row per config (seeds pooled)")
    parser.add_argument("--csv", type=Path, help="also write the table to this CSV file")
    args = parser.parse_args()

    names = args.experiments or sorted(
        d.name for d in args.outputs_dir.iterdir() if d.is_dir() and any(d.glob("*/results.txt"))
    )
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    gpus: dict[tuple[str, str], set[str]] = defaultdict(set)
    for name in names:
        for run_dir in sorted((args.outputs_dir / name).iterdir()):
            cost = run_cost(run_dir) if run_dir.is_dir() else None
            if cost is None:
                continue
            key = (name, cell_name(run_dir.name) if args.by_cell else "")
            groups[key].append(cost[0])
            gpus[key].add(cost[1])

    header = ["experiment", "cell", "runs", "median_minutes_per_run", "total_gpu_hours", "gpu"]
    rows = [
        [
            name,
            cell,
            len(m),
            round(statistics.median(m), 1),
            round(sum(m) / 60, 1),
            ", ".join(sorted(gpus[(name, cell)])),
        ]
        for (name, cell), m in sorted(groups.items())
    ]
    if not args.by_cell:
        header.pop(1)
        rows = [row[:1] + row[2:] for row in rows]

    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for row in rows:
        print("| " + " | ".join(str(v) for v in row) + " |")
    total = sum(sum(m) for m in groups.values()) / 60
    print(f"\nTotal: {sum(len(m) for m in groups.values())} runs, {total:.1f} GPU-hours")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            csv.writer(f).writerows([header, *rows])


if __name__ == "__main__":
    main()
