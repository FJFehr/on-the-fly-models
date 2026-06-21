"""Augment the 1D-ARC dataset with colour permutations, shifts, and mirroring.

Reads from data/arc_1d (variable-length), augments the train split only,
and writes to data/arc_1d_augmented in the same schema. Dev and test splits
are passed through unchanged so results remain comparable to the baseline.

Augmentation pipeline per task (applied in this order):
  1. colour permutations — remap non-zero colours consistently across all sequences
     (global mode, default) OR independently per support pair + query (--per-pair)
  2. shifts             — extend sequences with zero-padding at one end
  3. mirror             — reverse all sequences left-right (optional, default on)

For 1d_mirror tasks, colour 9 (the semantic pivot) is never permuted and no
other colour is remapped to 9. See FIXED_RULE_COLOURS.

Per-pair mode (--per-pair): each support pair and the query get independent
injective colour mappings. Colours may repeat across pairs. This expands the
augmentation space from ~21 global variants to ~9^k combinations (where k is
the number of colours in a pair), enabling patterns impossible under the global
scheme. Suitable for generating large augmented datasets (e.g. 1000 variants/task).

Some tasks require global augmentation even when --per-pair is set, because their
rule depends on colour identity being consistent across all support pairs and the
query. See GLOBAL_ONLY_TASKS. These tasks fall back to global augmentation
automatically.

After augmentation, any train example whose (support_inputs, support_outputs,
query_input, query_output) exactly matches a dev or test example is removed to
prevent data leakage.
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

# Tasks that must use global colour augmentation even when --per-pair is set.
# These tasks have a colour that is consistent across ALL support pairs and the
# query within each task instance (e.g. a direction marker or the recolour input
# colour). Per-pair augmentation would make that colour inconsistent across pairs,
# breaking the learned rule.
GLOBAL_ONLY_TASKS: frozenset[str] = frozenset({
    "1d_move_dp",      # direction marker colour consistent across all pairs
    "1d_move_2p_dp",   # direction marker colour consistent across all pairs
    "1d_scale_dp",     # scaling marker colour consistent across all pairs
    "1d_recolor_oe",   # input + two output colours identical across all pairs and query
    "1d_recolor_cmp",  # input + two output colours identical across all pairs and query
    "1d_recolor_cnt",  # run-length → output colour mapping globally consistent across all pairs
})


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
        "--mirror",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include horizontally mirrored variants of each augmented example.",
    )
    parser.add_argument(
        "--per-pair",
        action="store_true",
        default=False,
        help=(
            "Use per-pair colour augmentation: each support pair and query get "
            "independent injective colour mappings instead of one global mapping. "
            "--n-color-permutations controls how many per-pair variants to generate."
        ),
    )
    parser.add_argument(
        "--task-categories",
        nargs="+",
        default=None,
        metavar="CATEGORY",
        help=(
            "If given, only augment tasks whose task_category is in this list. "
            "Dev/test splits are filtered to the same categories. "
            "Example: --task-categories 1d_move_1p 1d_move_2p 1d_move_3p"
        ),
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
    """Return exactly n colour-permuted variants of the task (excluding original).

    If fixed_colour is given, that colour is never remapped and no other colour
    is mapped to it (e.g. colour 9 is the semantic pivot for 1d_mirror tasks).

    For tasks with a small colour space (e.g. only 2 colours), the number of
    unique injective mappings may be less than n. In that case, mappings are
    resampled (with replacement) so that exactly n variants are always returned.
    Identity mappings (no change) are always skipped and resampled.
    """
    task_colors = get_task_colors(task)
    permutable = [c for c in task_colors if c != fixed_colour]
    if not permutable:
        return []

    available = [c for c in range(1, 10) if c != fixed_colour]
    variants: list[dict] = []

    while len(variants) < n:
        rng.shuffle(available)
        color_map = {orig: available[i] for i, orig in enumerate(permutable)}
        if all(orig == mapped for orig, mapped in color_map.items()):
            continue
        variants.append(apply_color_map(task, color_map))

    return variants


def generate_per_pair_color_augmentations(
    task: dict,
    n: int,
    rng: random.Random,
    fixed_colour: int | None = None,
) -> list[dict]:
    """Return n task variants where each support pair and query are independently recoloured.

    Within each pair the mapping is injective (no colour collapse).
    Across pairs the same colour may recur freely, unlocking combinations
    impossible under a global injective mapping.

    If fixed_colour is given, that colour is never remapped and is never a
    remap target (e.g. colour 9 for 1d_mirror tasks).
    """
    available = [c for c in range(1, 10) if c != fixed_colour]
    variants: list[dict] = []

    for _ in range(n):
        new_si, new_so = [], []
        for si, so in zip(task["support_inputs"], task["support_outputs"]):
            pair_colours = sorted(
                {c for seq in (si, so) for c in seq if c != 0 and c != fixed_colour}
            )
            cmap = dict(zip(pair_colours, rng.sample(available, len(pair_colours))))
            new_si.append([cmap.get(c, c) for c in si])
            new_so.append([cmap.get(c, c) for c in so])

        q_colours = sorted(
            {
                c
                for seq in (task["query_input"], task["query_output"])
                for c in seq
                if c != 0 and c != fixed_colour
            }
        )
        qmap = dict(zip(q_colours, rng.sample(available, len(q_colours))))
        variants.append(
            {
                **task,
                "support_inputs": new_si,
                "support_outputs": new_so,
                "query_input": [qmap.get(c, c) for c in task["query_input"]],
                "query_output": [qmap.get(c, c) for c in task["query_output"]],
            }
        )

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


def apply_mirror(task: dict) -> dict:
    """Reverse all sequences left-right, preserving sequence_length."""

    def flip(seq: list[int]) -> list[int]:
        return list(reversed(seq))

    return {
        **task,
        "support_inputs": [flip(s) for s in task["support_inputs"]],
        "support_outputs": [flip(s) for s in task["support_outputs"]],
        "query_input": flip(task["query_input"]),
        "query_output": flip(task["query_output"]),
    }


def augment_task(
    task: dict,
    n_color_perms: int,
    shifts: list[int],
    rng: random.Random,
    *,
    mirror: bool = True,
    per_pair: bool = False,
) -> list[dict]:
    """Apply the full augmentation pipeline to one task.

    Pipeline: colour → shift → mirror.
    For 1d_mirror tasks, colour 9 is kept fixed (see FIXED_RULE_COLOURS).
    Returned tasks have new task_ids: original_task_id * 10000 + aug_index.

    With per_pair=True, colour augmentation uses generate_per_pair_color_augmentations
    instead of the global generate_color_permutations.
    """
    # Step 1: colour (original is variant 0)
    category = task.get("task_category", "")
    fixed_colour = FIXED_RULE_COLOURS.get(category)
    use_per_pair = per_pair and category not in GLOBAL_ONLY_TASKS
    if use_per_pair:
        color_variants = [task] + generate_per_pair_color_augmentations(
            task, n_color_perms, rng, fixed_colour=fixed_colour
        )
    else:
        color_variants = [task] + generate_color_permutations(
            task, n_color_perms, rng, fixed_colour=fixed_colour
        )

    # Step 2: shifts (0 = no shift is always included)
    all_shifts = [0] + [s for s in shifts if s != 0]
    all_variants = [apply_shift(cv, s) for cv in color_variants for s in all_shifts]

    # Step 3: mirror (each variant becomes original + mirrored)
    if mirror:
        mirrored: list[dict] = []
        for v in all_variants:
            mirrored.append(v)
            mirrored.append(apply_mirror(v))
        all_variants = mirrored

    orig_id = task["task_id"]
    return [{**v, "task_id": orig_id * 10000 + i} for i, v in enumerate(all_variants)]


def augment_split(
    split: Dataset,
    n_color_perms: int,
    shifts: list[int],
    rng: random.Random,
    *,
    mirror: bool = True,
    per_pair: bool = False,
) -> list[dict]:
    """Augment every task in the split and return the combined list."""
    augmented: list[dict] = []
    for task in split:
        augmented.extend(
            augment_task(task, n_color_perms, shifts, rng, mirror=mirror, per_pair=per_pair)
        )
    return augmented


def _example_fingerprint(task: dict) -> tuple:
    """Hashable fingerprint of the full example content (ignores task_id)."""
    return (
        tuple(tuple(s) for s in task["support_inputs"]),
        tuple(tuple(s) for s in task["support_outputs"]),
        tuple(task["query_input"]),
        tuple(task["query_output"]),
    )


def filter_held_out_contamination(
    aug_train: list[dict], base: DatasetDict
) -> list[dict]:
    """Remove augmented train examples that exactly match any dev or test example."""
    held_out: set[tuple] = set()
    for split_name in ("dev", "test"):
        if split_name in base:
            for task in base[split_name]:
                held_out.add(_example_fingerprint(task))
    if not held_out:
        return aug_train
    before = len(aug_train)
    filtered = [t for t in aug_train if _example_fingerprint(t) not in held_out]
    removed = before - len(filtered)
    if removed:
        print(f"Removed {removed} augmented train examples that matched dev/test examples.")
    return filtered


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

    if args.task_categories:
        cats = set(args.task_categories)
        print(f"Filtering to task categories: {sorted(cats)}")
        base = DatasetDict(
            {name: split.filter(lambda t: t["task_category"] in cats) for name, split in base.items()}
        )

    rng = random.Random(args.seed)
    shifts = args.shifts or []

    n_color = args.n_color_permutations
    n_shifts = 1 + len([s for s in shifts if s != 0])
    n_mirror = 2 if args.mirror else 1
    colour_mode = "per-pair" if args.per_pair else "global"
    variants_per_task = (1 + n_color) * n_shifts * n_mirror
    print(
        f"Augmenting train split: {n_color} {colour_mode} colour variants, "
        f"shifts={shifts}, mirror={args.mirror} → {variants_per_task} variants per task"
    )

    aug_train = augment_split(
        base["train"], n_color, shifts, rng, mirror=args.mirror, per_pair=args.per_pair
    )
    aug_train = filter_held_out_contamination(aug_train, base)

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
