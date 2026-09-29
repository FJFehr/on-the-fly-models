"""Experiment 7, compositional: score every arm's in-distribution checkpoints zero-shot on
experiment 4's 10 composite categories.

A composite category chains two base rules, so td arms get both components' task embeddings:
the multi-hot task vector over its two base categories (common.COMPOSITE_COMPONENTS), or with
--mode mean their average. No new task columns are needed, so unlike experiment 4 nothing is
padded and num_tasks stays 18. notd has no task vector and is scored as in experiment 4, which
checks this script against experiment 4's published numbers.

Metrics are the same as scripts/eval_compositional_holdout.py (per-task query exact match and
token accuracy, averaged per category), whose dataloader this reuses. Every arm is scored from
best_model.ckpt, as experiment 4 did.

Usage:
    PYTHONPATH=. uv run python experiments/07_task_identity_ablation/eval_compositional.py
    PYTHONPATH=. uv run python experiments/07_task_identity_ablation/eval_compositional.py \\
        --arms learnedtd_input --seeds 1 --mode mean
"""

import argparse
from collections import defaultdict
from pathlib import Path

import torch
from common import (
    ARMS,
    COMPOSITE_COMPONENTS,
    EXPERIMENT,
    INDIST_RUNS,
    SEEDS,
    composite_task_vector,
    load_run,
)

from scripts.eval_compositional_holdout import build_holdout_dataloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--mode", default="multihot", choices=["multihot", "mean"])
    parser.add_argument("--holdout-data-dir", default="data/arc_1d_compositional_holdout")
    parser.add_argument("--holdout-split", default="holdout_test")
    parser.add_argument(
        "--output-dir", type=Path, default=Path(f"outputs/{EXPERIMENT}/compositional")
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def evaluate(run_dir: str, arm: str, args: argparse.Namespace) -> str:
    vectors = None
    if arm != "notd":
        vectors = {c: composite_task_vector(c, args.mode) for c in COMPOSITE_COMPONENTS}
    cfg, _, model = load_run(run_dir, "best_model.ckpt", category_vectors=vectors)
    model.to(args.device)
    dataloader = build_holdout_dataloader(
        args.holdout_data_dir, args.holdout_split, cfg.batch_size, cfg.padding_idx
    )

    exact = defaultdict(list)
    accuracy = defaultdict(list)
    with torch.no_grad():
        for batch in dataloader:
            batch = {
                k: v.to(args.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            _, predictions, targets = model.predict_batch(batch)
            for record in model.build_task_records(batch, predictions, targets):
                exact[record["task_category"]].append(record["query_exact_match"])
                accuracy[record["task_category"]].append(record["query_accuracy"])

    lines = [
        f"Compositional zero-shot eval: run={run_dir}, arm={arm}, task vector={args.mode}",
        "",
        f"{'category':<32}{'n':>6}{'exact_match':>14}{'seq_accuracy':>14}",
    ]
    for category in sorted(exact):
        n = len(exact[category])
        lines.append(
            f"{category:<32}{n:>6}{sum(exact[category]) / n:>14.3f}"
            f"{sum(accuracy[category]) / n:>14.3f}"
        )
    everything = [m for matches in exact.values() for m in matches]
    lines += ["", f"{'overall':<32}{len(everything):>6}{sum(everything) / len(everything):>14.3f}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    for arm in args.arms:
        for seed in args.seeds:
            # notd has no task vector, so its result does not depend on --mode.
            name = f"{arm}_seed{seed}" if arm == "notd" else f"{arm}_{args.mode}_seed{seed}"
            out_path = args.output_dir / name / "results.txt"
            run_dir = INDIST_RUNS[arm].format(seed=seed)
            if out_path.exists():
                print(f"SKIP   {name} (already evaluated)")
                continue
            if not (Path(run_dir) / "best_model.ckpt").exists():
                print(f"SKIP   {name} (no checkpoint in {run_dir})")
                continue
            print(f"RUN    {name}")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(evaluate(run_dir, arm, args))
            print(f"DONE   {name}")


if __name__ == "__main__":
    main()
