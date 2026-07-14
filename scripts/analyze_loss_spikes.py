"""Correlate loss spikes against gradient norm, epoch position, and batch composition.

Reads the `train_step_diagnostics.jsonl` file written by
`HyperModelLightning._write_step_diagnostics` (one record per training step,
containing per-inner-supervision-step loss/grad-norm plus the batch's
task-category/task-id composition) and reports, for each detected spike:

- the max/mean gradient norm across that step's inner supervision passes,
  relative to the run's overall grad-norm distribution (exploding-gradient check)
- how far into its epoch the spiking batch fell (epoch-boundary/reshuffle check)
- the task-category composition and task_ids of the spiking batch (data-item check)

Epoch boundaries are derived directly from transitions in the record's `epoch`
field rather than assumed from dataset_size // batch_size, so this works for
any run regardless of config.

Usage:
    python scripts/analyze_loss_spikes.py path/to/train_step_diagnostics.jsonl
    python scripts/analyze_loss_spikes.py path/to/train_step_diagnostics.jsonl --window 20 --z 3.0
"""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def load_records(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records.sort(key=lambda r: r["global_step"])
    return records


def compute_epoch_boundaries(records: list[dict]) -> dict[int, int]:
    """Map global_step -> steps since the start of its epoch."""
    steps_into_epoch = {}
    epoch_start_step = records[0]["global_step"]
    current_epoch = records[0]["epoch"]
    for rec in records:
        if rec["epoch"] != current_epoch:
            current_epoch = rec["epoch"]
            epoch_start_step = rec["global_step"]
        steps_into_epoch[rec["global_step"]] = rec["global_step"] - epoch_start_step
    return steps_into_epoch


def detect_spikes(records: list[dict], window: int, z_thresh: float) -> list[dict]:
    losses = [statistics.mean(r["inner_losses"]) for r in records]
    spikes = []
    for i in range(len(records)):
        lo = max(0, i - window)
        history = losses[lo:i]
        if len(history) < max(5, window // 2):
            continue
        median = statistics.median(history)
        mad = statistics.median([abs(x - median) for x in history]) or 1e-8
        # Robust z-score using MAD (1.4826 makes MAD ~ comparable to std for normal data).
        z = (losses[i] - median) / (1.4826 * mad)
        if z > z_thresh:
            spikes.append({"index": i, "z": z, "loss": losses[i], "median": median})
    return spikes


def summarize(records: list[dict], spikes: list[dict], steps_into_epoch: dict[int, int]) -> None:
    all_grad_norms = [gn for r in records for gn in r["inner_grad_norms"]]
    baseline_max = statistics.median([max(r["inner_grad_norms"]) for r in records])
    baseline_p95 = sorted(all_grad_norms)[int(0.95 * len(all_grad_norms))]

    print(f"Loaded {len(records)} steps.")
    print(f"Baseline grad norm: median-of-per-step-max={baseline_max:.4g}, p95-overall={baseline_p95:.4g}")
    print(f"Detected {len(spikes)} spike(s).\n")

    if not spikes:
        print("No spikes detected — try a smaller --z or --window, or a longer run.")
        return

    header = f"{'step':>8} {'epoch':>6} {'batch_idx':>10} {'into_epoch':>11} {'loss':>8} {'z':>6} {'grad_max':>10} {'grad_max/baseline':>18}  dominant_task_category"
    print(header)
    print("-" * len(header))
    for spike in spikes:
        rec = records[spike["index"]]
        grad_max = max(rec["inner_grad_norms"])
        dominant_category, count = Counter(rec["task_category_counts"]).most_common(1)[0]
        print(
            f"{rec['global_step']:>8} {rec['epoch']:>6} {rec['batch_idx']:>10} "
            f"{steps_into_epoch[rec['global_step']]:>11} {spike['loss']:>8.4g} {spike['z']:>6.2f} "
            f"{grad_max:>10.4g} {grad_max / baseline_max:>18.2f}x  {dominant_category} ({count})"
        )

    print("\nRecurring task_ids across spikes (id: count of spikes it appears in):")
    id_hits = Counter()
    for spike in spikes:
        for tid in records[spike["index"]]["task_ids"]:
            id_hits[tid] += 1
    for tid, count in id_hits.most_common(10):
        if count > 1:
            print(f"  {tid}: appears in {count}/{len(spikes)} spiking batches")

    print("\nSpike position within epoch (steps_into_epoch, 0 = first batch of epoch):")
    positions = [steps_into_epoch[records[s["index"]]["global_step"]] for s in spikes]
    print(f"  {positions}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("diagnostics_path", type=Path)
    parser.add_argument("--window", type=int, default=20, help="Rolling-history window (steps) for spike detection.")
    parser.add_argument("--z", type=float, default=3.0, help="Robust z-score threshold to flag a spike.")
    args = parser.parse_args()

    records = load_records(args.diagnostics_path)
    if not records:
        print(f"No records found in {args.diagnostics_path}")
        return
    steps_into_epoch = compute_epoch_boundaries(records)
    spikes = detect_spikes(records, window=args.window, z_thresh=args.z)
    summarize(records, spikes, steps_into_epoch)


if __name__ == "__main__":
    main()
