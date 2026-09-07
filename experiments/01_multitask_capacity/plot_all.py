"""Regenerate every figure for experiment 1 (multi-task capacity) in one pass.

The single entry point to run after a training sweep finishes: rescans
outputs/, refreshes both CSVs, and renders the capacity-cliff plot plus a
per-task breakdown for every model size actually present in the data --
no need to remember plot_capacity_cliff.py's own invocation and then
plot_per_task.py once per --dim.

Usage
-----
    uv run python experiments/01_multitask_capacity/plot_all.py             # rescan, refresh, plot
    uv run python experiments/01_multitask_capacity/plot_all.py --from-csv  # replot from the CSVs
"""

import argparse
from pathlib import Path

import plot_capacity_cliff as cliff
import plot_per_task as per_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Rescan this outputs/ dir and refresh both CSVs before plotting (default: outputs).",
    )
    parser.add_argument(
        "--from-csv",
        action="store_true",
        help="Skip the rescan; replot from the CSVs already committed next to this script.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("== capacity-cliff (Individual vs. Joint, all sizes) ==")
    if args.from_csv:
        cliff_records = cliff.read_csv(cliff.CSV_PATH)
    else:
        cliff_records = cliff.extract_records(args.outputs_dir)
        cliff.write_csv(cliff_records, cliff.CSV_PATH)
    cliff_series = cliff.build_series(cliff_records)
    dims_sorted = sorted(cliff.DIM_PARAMS, key=lambda d: cliff.DIM_PARAMS[d])
    for cond in ("individual", "notd", "td"):
        for dim in dims_sorted:
            s = cliff_series[cond].get(dim)
            if s:
                print(
                    f"  {cond:10s} dim={dim:>2s} ({cliff.DIM_PARAMS[dim]:>5,} params): "
                    f"mean={s['mean']:.3f} std={s['std']:.3f} n={s['n']}"
                )
    cliff.plot(cliff_series, cliff.PLOT_PATH)

    print("\n== per-task breakdown, every size present in the data ==")
    if args.from_csv:
        task_records = per_task.read_csv(per_task.CSV_PATH)
    else:
        task_records = per_task.extract_records(args.outputs_dir)
        per_task.write_csv(task_records, per_task.CSV_PATH)
    available_dims = sorted({r["dim"] for r in task_records}, key=int)
    if not available_dims:
        print("  No per-task data found -- skipping.")
    for dim in available_dims:
        series = per_task.build_series(task_records, dim)
        out_path = Path(f"outputs/figures/01_multitask_capacity/per_task_dim{dim}.png")
        per_task.plot(series, dim, out_path)
        print(f"  dim={dim:>2s}: {out_path}")

    print("\nDone. Figures in outputs/figures/01_multitask_capacity/")
    print(f"CSVs in {cliff.CSV_PATH.parent}")


if __name__ == "__main__":
    main()
