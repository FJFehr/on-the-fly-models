"""Augment the 1D-ARC dataset with colour permutations and shifts.

Reads from data/arc_1d (variable-length), augments the train split only,
and writes to data/arc_1d_augmented in the same schema. Dev and test splits
are passed through unchanged so results remain comparable to the baseline.

Augmentation pipeline per task (applied in this order):
  1. colour permutations — remap non-zero colours consistently across all sequences
  2. shifts             — extend sequences with zero-padding at one end

For 1d_mirror tasks, colour 9 (the semantic pivot) is never permuted and no
other colour is remapped to 9.
"""

import argparse
import random
from collections import Counter
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk

INPUT_DIR = Path("data/arc_1d")
OUTPUT_DIR = Path("data/arc_1d_augmented")

# Category whose pivot colour must not be permuted and may not be a remap target.
FIXED_RULE_COLOURS: dict[str, int] = {
    "1d_mirror": 9,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR,
        help="Base variable-length ARC1D dataset to augment.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where the augmented dataset will be saved.",
    )
    parser.add_argument(
        "--n-color-permutations",
        type=int,
        default=21,
        help="Number of additional colour permutations per task (excluding original).",
    )
    parser.add_argument(
        "--shifts",
        type=int,
        nargs="*",
        default=[1, 2, -1, -2],
        help="Shift offsets to apply. 0 (no shift) is always included.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return parser.parse_args()


def get_task_colors(task: dict) -> list[int]:
    """Return sorted unique non-zero colour values across all sequences in the task."""
    seen: set[int] = set()
    for seq in task["support_inputs"]:
        seen.update(seq)
    for seq in task["support_outputs"]:
        seen.update(seq)
    seen.update(task["query_input"])
    seen.update(task["query_output"])
    return sorted(c for c in seen if c != 0)


def apply_color_map(task: dict, color_map: dict[int, int]) -> dict:
    """Apply a colour remapping to every sequence in the task.

    Background (0) is never in color_map and is preserved via dict.get fallback.
    """

    def remap(seq: list[int]) -> list[int]:
        return [color_map.get(v, v) for v in seq]

    return {
        **task,
        "support_inputs": [remap(s) for s in task["support_inputs"]],
        "support_outputs": [remap(s) for s in task["support_outputs"]],
        "query_input": remap(task["query_input"]),
        "query_output": remap(task["query_output"]),
    }


def generate_color_permutations(
    task: dict,
    n: int,
    rng: random.Random,
    fixed_colour: int | None = None,
) -> list[dict]:
    """Return up to n unique colour-permuted variants of the task (excluding original).

    If fixed_colour is given, that colour is never remapped and no other colour
    is mapped to it (e.g. colour 9 is the semantic pivot for 1d_mirror tasks).
    Duplicates are skipped. Returns fewer than n if the colour space is exhausted.
    """
    task_colors = get_task_colors(task)
    permutable = [c for c in task_colors if c != fixed_colour]
    if not permutable:
        return []

    available = [c for c in range(1, 10) if c != fixed_colour]
    variants: list[dict] = []
    seen_maps: set[tuple[tuple[int, int], ...]] = set()
    max_attempts = n * 20

    for _ in range(max_attempts):
        if len(variants) >= n:
            break
        rng.shuffle(available)
        color_map = {orig: available[i] for i, orig in enumerate(permutable)}
        if all(orig == mapped for orig, mapped in color_map.items()):
            continue
        map_key = tuple(sorted(color_map.items()))
        if map_key not in seen_maps:
            seen_maps.add(map_key)
            variants.append(apply_color_map(task, color_map))

    return variants


def apply_shift(task: dict, shift: int) -> dict:
    """Shift all sequences by shift positions, extending sequence_length.

    Positive shift prepends |shift| zeros (content moves right).
    Negative shift appends |shift| zeros (content moves left).
    sequence_length increases by abs(shift). No content is discarded.
    """
    if shift == 0:
        return task

    abs_shift = abs(shift)
    zeros = [0] * abs_shift

    if shift > 0:

        def do_shift(seq: list[int]) -> list[int]:
            return zeros + list(seq)
    else:

        def do_shift(seq: list[int]) -> list[int]:
            return list(seq) + zeros

    return {
        **task,
        "sequence_length": task["sequence_length"] + abs_shift,
        "support_inputs": [do_shift(s) for s in task["support_inputs"]],
        "support_outputs": [do_shift(s) for s in task["support_outputs"]],
        "query_input": do_shift(task["query_input"]),
        "query_output": do_shift(task["query_output"]),
    }


def augment_task(
    task: dict,
    n_color_perms: int,
    shifts: list[int],
    rng: random.Random,
) -> list[dict]:
    """Apply the full augmentation pipeline to one task.

    Pipeline: colour → shift.
    For 1d_mirror tasks, colour 9 is kept fixed (see FIXED_RULE_COLOURS).
    Returned tasks have new task_ids: original_task_id * 10000 + aug_index.
    """
    # Step 1: colour (original is variant 0)
    category = task.get("task_category", "")
    fixed_colour = FIXED_RULE_COLOURS.get(category)
    color_variants = [task] + generate_color_permutations(
        task, n_color_perms, rng, fixed_colour=fixed_colour
    )

    # Step 2: shifts (0 = no shift is always included)
    all_shifts = [0] + [s for s in shifts if s != 0]
    all_variants = [apply_shift(cv, s) for cv in color_variants for s in all_shifts]

    orig_id = task["task_id"]
    return [{**v, "task_id": orig_id * 10000 + i} for i, v in enumerate(all_variants)]


def augment_split(
    split: Dataset,
    n_color_perms: int,
    shifts: list[int],
    rng: random.Random,
) -> list[dict]:
    """Augment every task in the split and return the combined list."""
    augmented: list[dict] = []
    for task in split:
        augmented.extend(augment_task(task, n_color_perms, shifts, rng))
    return augmented


def print_split_summary(dataset_dict: DatasetDict) -> None:
    for split_name, dataset in dataset_dict.items():
        category_counts = Counter(dataset["task_category"])
        print(f"{split_name}: {len(dataset)} tasks")
        for category in sorted(category_counts):
            print(f"  {category}: {category_counts[category]} tasks")


def main() -> None:
    args = parse_args()

    print(f"Loading base dataset from {args.input_dir} ...")
    base = load_from_disk(str(args.input_dir))

    rng = random.Random(args.seed)
    shifts = args.shifts or []

    n_color = args.n_color_permutations
    n_shifts = 1 + len([s for s in shifts if s != 0])
    variants_per_task = (1 + n_color) * n_shifts
    print(
        f"Augmenting train split: {n_color} colour permutations, "
        f"shifts={shifts} → {variants_per_task} variants per task"
    )

    aug_train = augment_split(base["train"], n_color, shifts, rng)

    splits: dict[str, Dataset] = {"train": Dataset.from_list(aug_train)}
    for split_name in ("dev", "test"):
        if split_name in base:
            splits[split_name] = base[split_name]

    dataset_dict = DatasetDict(splits)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(args.output_dir))

    print(f"\nSaved augmented dataset to {args.output_dir}")
    print(dataset_dict)
    print_split_summary(dataset_dict)


if __name__ == "__main__":
    main()
