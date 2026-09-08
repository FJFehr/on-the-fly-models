"""Regenerate every figure for experiment 1 (multi-task capacity) in one pass.

The single entry point to run after a training sweep finishes (see run.sh):
rescans outputs/, refreshes every CSV, and renders every figure --
capacity-cliff, a per-task breakdown for every model size present, and both
ablations (Canon, optimizer) if their data is present. No need to remember
five separate scripts' own invocations.

Usage
-----
    uv run python experiments/01_multitask_capacity/plot_all.py             # rescan, refresh, plot
    uv run python experiments/01_multitask_capacity/plot_all.py --from-csv  # replot from the CSVs
"""

import argparse
from pathlib import Path

import plot_canon_ablation as canon_ablation
import plot_capacity_cliff as cliff
import plot_optimizer_ablation as optimizer_ablation
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

    print("\n== Canon ablation (with vs. without Canon) ==")
    if args.from_csv:
        canon_records = canon_ablation.read_csv(canon_ablation.CSV_PATH)
    else:
        canon_records = canon_ablation.extract_records(args.outputs_dir)
        canon_ablation.write_csv(canon_records, canon_ablation.CSV_PATH)
    if canon_records:
        canon_ablation.plot(canon_ablation.build_series(canon_records), canon_ablation.PLOT_PATH)
    else:
        print("  No canon-ablation data found (configs/nocanon/) -- skipping.")

    print("\n== Optimizer ablation (Muon vs. AdamW) ==")
    if args.from_csv:
        opt_records = optimizer_ablation.read_csv(optimizer_ablation.CSV_PATH)
    else:
        opt_records = optimizer_ablation.extract_records(args.outputs_dir)
        optimizer_ablation.write_csv(opt_records, optimizer_ablation.CSV_PATH)
    if opt_records:
        optimizer_ablation.plot(
            optimizer_ablation.build_series(opt_records), optimizer_ablation.PLOT_PATH
        )
    else:
        print("  No optimizer-ablation data found (configs/adamw/) -- skipping.")

    print("\nDone. Figures in outputs/figures/01_multitask_capacity/")
    print(f"CSVs in {cliff.CSV_PATH.parent}")


if __name__ == "__main__":
    main()
