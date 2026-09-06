"""Independently verify there is no train/eval content overlap in a built ARC-1D dataset.

scripts/augment_arc_1d.py already drops augmented train examples that exactly
match a dev/test example at build time (filter_held_out_contamination). This
script is a standalone, independent regression guard for that property: it
re-derives the same exact-content fingerprint directly from the final,
on-disk dataset splits and reports any overlap it finds, rather than trusting
that the build-time filter ran (or ran correctly). It also checks the
compositional holdout set (built by a wholly separate generator,
data_modules/arc1d_compositional.py, with disjoint category names) against
train, for the same reason.

A fingerprint is the exact (support_inputs, support_outputs, query_input,
query_output) content of an example, ignoring task_id -- identical to
scripts/augment_arc_1d.py::_example_fingerprint. Two examples with the same
fingerprint are, for training purposes, the same example: a model that sees
one at train time has seen the other's content at eval time too.

Usage
-----
    python scripts/check_data_leakage.py
    python scripts/check_data_leakage.py --data-dir data/arc_1d_looped_augmented
    python scripts/check_data_leakage.py --holdout-dir data/arc_1d_compositional_holdout

Exits non-zero if any overlap is found.
"""

import argparse
import sys
from pathlib import Path

from datasets import DatasetDict, load_from_disk

DEFAULT_DATA_DIR = "data/arc_1d_looped_augmented"
DEFAULT_HOLDOUT_DIR = "data/arc_1d_compositional_holdout"


def example_fingerprint(task: dict) -> tuple:
    """Hashable fingerprint of an example's full content (ignores task_id).

    Kept identical to scripts/augment_arc_1d.py::_example_fingerprint --
    two examples must be judged the same way in both places, or this script
    would not actually be checking what the build-time filter checks.
    """
    return (
        tuple(tuple(s) for s in task["support_inputs"]),
        tuple(tuple(s) for s in task["support_outputs"]),
        tuple(task["query_input"]),
        tuple(task["query_output"]),
    )


def fingerprints(dataset) -> set[tuple]:
    return {example_fingerprint(task) for task in dataset}


def check_overlap(name_a: str, set_a: set[tuple], name_b: str, set_b: set[tuple]) -> int:
    overlap = set_a & set_b
    if overlap:
        print(f"  LEAKAGE: {len(overlap)} example(s) shared between {name_a} and {name_b}")
    else:
        print(f"  OK: {name_a} ({len(set_a)}) vs {name_b} ({len(set_b)}) -- no overlap")
    return len(overlap)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Main built dataset (train/dev/test splits).")
    parser.add_argument(
        "--holdout-dir",
        default=DEFAULT_HOLDOUT_DIR,
        help="Compositional holdout dataset (a single holdout_test split), checked against --data-dir's train. "
        "Pass '' to skip.",
    )
    args = parser.parse_args()

    total_leaks = 0

    data_path = Path(args.data_dir)
    if not data_path.is_dir():
        print(f"'{args.data_dir}' not found -- build it first (see scripts/augment_arc_1d.py).")
        sys.exit(2)

    print(f"Loading {args.data_dir} ...")
    dataset: DatasetDict = load_from_disk(str(data_path))
    fps = {split_name: fingerprints(split) for split_name, split in dataset.items()}
    for split_name, fp_set in fps.items():
        print(f"  {split_name}: {len(dataset[split_name])} examples, {len(fp_set)} distinct")

    print(f"\nChecking splits within {args.data_dir}:")
    if "train" in fps:
        for other in ("dev", "test"):
            if other in fps:
                total_leaks += check_overlap("train", fps["train"], other, fps[other])
    else:
        print("  no 'train' split found -- nothing to check here.")

    if args.holdout_dir:
        holdout_path = Path(args.holdout_dir)
        if holdout_path.is_dir() and "train" in fps:
            print(f"\nChecking {args.holdout_dir} against train:")
            holdout: DatasetDict = load_from_disk(str(holdout_path))
            for split_name, split in holdout.items():
                holdout_fps = fingerprints(split)
                print(f"  {split_name}: {len(split)} examples, {len(holdout_fps)} distinct")
                total_leaks += check_overlap("train", fps["train"], f"holdout/{split_name}", holdout_fps)
        elif not holdout_path.is_dir():
            print(f"\n'{args.holdout_dir}' not found -- skipping compositional holdout check.")

    print()
    if total_leaks:
        print(f"FAILED: {total_leaks} leaked example(s) found across all checks.")
        sys.exit(1)
    print("PASSED: no train/eval content overlap found.")


if __name__ == "__main__":
    main()
