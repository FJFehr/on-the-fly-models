"""Augment the 1D-ARC dataset with colour permutations, shifts, and mirroring.

Reads from data/arc_1d (variable-length), augments the train split, and
optionally augments dev/test with colour permutations to reduce metric variance.
Dev and test are never shifted or mirrored — only colour-permuted.

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

# Canonical output colours used when --canonical-recolor-colors is enabled.
# All recolor task instances are remapped to these fixed semantic colours so the
# model can learn a global rule rather than inferring colour roles from context.
RECOLOR_OE_ODD_COLOR = 1    # blue
RECOLOR_OE_EVEN_COLOR = 2   # red
RECOLOR_CMP_BIGGER_COLOR = 3  # green
RECOLOR_CMP_SMALLER_COLOR = 6  # magenta


def _cnt_canonical_color(count: int) -> int:
    """Canonical colour for a run of length count: 1→1, 2→2, …, 9→9, 10→1, …"""
    return ((count - 1) % 9) + 1


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
        "--dev-test-n-permutations",
        type=int,
        default=0,
        help=(
            "Number of additional colour permutations to apply to dev and test splits "
            "(0 = pass through unchanged). Shifts and mirroring are never applied to "
            "dev/test. Example: 19 gives 5 original × 20 variants = 100 examples per task."
        ),
    )
    parser.add_argument(
        "--canonical-recolor-colors",
        action="store_true",
        default=False,
        help=(
            "Fix recolor task output colours to canonical semantic values: "
            "1d_recolor_oe (odd→blue/1, even→red/2), "
            "1d_recolor_cmp (bigger→green/3, smaller→magenta/6), "
            "1d_recolor_cnt (count N→((N-1)%%9)+1). "
            "Only the input colour is permuted during augmentation. "
            "Intended for ablation experiments; off by default."
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


def apply_output_color_map(task: dict, color_map: dict[int, int]) -> dict:
    """Apply a colour remapping only to output sequences, leaving inputs unchanged.

    Background (0) is preserved via dict.get fallback.
    """

    def remap(seq: list[int]) -> list[int]:
        return [color_map.get(v, v) for v in seq]

    return {
        **task,
        "support_outputs": [remap(s) for s in task["support_outputs"]],
        "query_output": remap(task["query_output"]),
    }


def get_runs(seq: list[int]) -> list[tuple[int, int, int]]:
    """Return (start, length, color) for each contiguous non-zero run in seq."""
    runs: list[tuple[int, int, int]] = []
    i = 0
    while i < len(seq):
        if seq[i] == 0:
            i += 1
            continue
        start = i
        color = seq[i]
        while i < len(seq) and seq[i] == color:
            i += 1
        runs.append((start, i - start, color))
    return runs


def canonicalize_recolor_oe(task: dict) -> dict:
    """Remap output colours: odd-length runs → RECOLOR_OE_ODD_COLOR, even → RECOLOR_OE_EVEN_COLOR."""
    odd_color: int | None = None
    even_color: int | None = None
    for inp, out in zip(task["support_inputs"], task["support_outputs"]):
        for (_, length, _), (_, _, out_c) in zip(get_runs(inp), get_runs(out)):
            if length % 2 == 1:
                odd_color = out_c
            else:
                even_color = out_c
            if odd_color is not None and even_color is not None:
                break
        if odd_color is not None and even_color is not None:
            break
    if odd_color is None or even_color is None:
        return task
    return apply_output_color_map(task, {odd_color: RECOLOR_OE_ODD_COLOR, even_color: RECOLOR_OE_EVEN_COLOR})


def canonicalize_recolor_cmp(task: dict) -> dict:
    """Remap output colours: the colour for longer runs → green (3), shorter → magenta (6)."""
    color_lengths: dict[int, list[int]] = {}
    for inp, out in zip(task["support_inputs"], task["support_outputs"]):
        for (_, length, _), (_, _, out_c) in zip(get_runs(inp), get_runs(out)):
            color_lengths.setdefault(out_c, []).append(length)
    if len(color_lengths) != 2:
        return task
    c0, c1 = list(color_lengths)
    mean0 = sum(color_lengths[c0]) / len(color_lengths[c0])
    mean1 = sum(color_lengths[c1]) / len(color_lengths[c1])
    bigger, smaller = (c0, c1) if mean0 >= mean1 else (c1, c0)
    return apply_output_color_map(task, {bigger: RECOLOR_CMP_BIGGER_COLOR, smaller: RECOLOR_CMP_SMALLER_COLOR})


def canonicalize_recolor_cnt(task: dict) -> dict:
    """Remap output colours: a run of length N → _cnt_canonical_color(N)."""
    color_map: dict[int, int] = {}
    for inp, out in zip(task["support_inputs"], task["support_outputs"]):
        for (_, length, _), (_, _, out_c) in zip(get_runs(inp), get_runs(out)):
            color_map[out_c] = _cnt_canonical_color(length)
    return apply_output_color_map(task, color_map)


def generate_color_permutations(
    task: dict,
    n: int,
    rng: random.Random,
    fixed_colour: int | None = None,
    fixed_colours: frozenset[int] = frozenset(),
) -> list[dict]:
    """Return exactly n colour-permuted variants of the task (excluding original).

    fixed_colour: single colour never remapped (legacy; e.g. colour 9 for 1d_mirror).
    fixed_colours: additional colours to exclude from permutation (e.g. canonical
        semantic output colours set by --canonical-recolor-colors).

    Both are merged internally. For tasks with a small permutable colour space the
    number of unique injective mappings may be less than n; mappings are resampled
    (with replacement) so that exactly n variants are returned. Identity mappings
    are always skipped. Returns [] if no non-identity mapping is possible.
    """
    combined_fixed = fixed_colours | ({fixed_colour} if fixed_colour is not None else frozenset())

    task_colors = get_task_colors(task)
    permutable = [c for c in task_colors if c not in combined_fixed]
    if not permutable:
        return []

    available = [c for c in range(1, 10) if c not in combined_fixed]
    if not available:
        return []
    # Guard: single permutable colour that maps only to itself → infinite loop without this.
    if len(permutable) == 1 and available == [permutable[0]]:
        return []

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
    canonical_recolor: bool = False,
) -> list[dict]:
    """Apply the full augmentation pipeline to one task.

    Pipeline: colour → shift → mirror.
    For 1d_mirror tasks, colour 9 is kept fixed (see FIXED_RULE_COLOURS).
    Returned tasks have new task_ids: original_task_id * 10000 + aug_index.

    With per_pair=True, colour augmentation uses generate_per_pair_color_augmentations
    instead of the global generate_color_permutations.

    With canonical_recolor=True, recolor task output colours are remapped to fixed
    semantic values before augmentation so the model can learn a global rule.
    """
    # Step 1: colour (original is variant 0)
    category = task.get("task_category", "")
    fixed_colour = FIXED_RULE_COLOURS.get(category)
    use_per_pair = per_pair and category not in GLOBAL_ONLY_TASKS

    # Canonicalize recolor output colours to fixed semantic values.
    canonical_fixed: frozenset[int] = frozenset()
    if canonical_recolor:
        if category == "1d_recolor_oe":
            task = canonicalize_recolor_oe(task)
            canonical_fixed = frozenset({RECOLOR_OE_ODD_COLOR, RECOLOR_OE_EVEN_COLOR})
        elif category == "1d_recolor_cmp":
            task = canonicalize_recolor_cmp(task)
            canonical_fixed = frozenset({RECOLOR_CMP_BIGGER_COLOR, RECOLOR_CMP_SMALLER_COLOR})
        elif category == "1d_recolor_cnt":
            task = canonicalize_recolor_cnt(task)
            canonical_fixed = frozenset(
                c
                for seq in task["support_outputs"] + [task["query_output"]]
                for c in seq
                if c != 0
            )

    if use_per_pair:
        color_variants = [task] + generate_per_pair_color_augmentations(
            task, n_color_perms, rng, fixed_colour=fixed_colour
        )
    else:
        color_variants = [task] + generate_color_permutations(
            task, n_color_perms, rng, fixed_colour=fixed_colour, fixed_colours=canonical_fixed
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
    canonical_recolor: bool = False,
) -> list[dict]:
    """Augment every task in the split and return the combined list."""
    augmented: list[dict] = []
    for task in split:
        augmented.extend(
            augment_task(
                task, n_color_perms, shifts, rng,
                mirror=mirror, per_pair=per_pair, canonical_recolor=canonical_recolor,
            )
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
        base["train"], n_color, shifts, rng,
        mirror=args.mirror, per_pair=args.per_pair,
        canonical_recolor=args.canonical_recolor_colors,
    )
    aug_train = filter_held_out_contamination(aug_train, base)

    splits: dict[str, Dataset] = {"train": Dataset.from_list(aug_train)}
    for split_name in ("dev", "test"):
        if split_name not in base:
            continue
        if args.dev_test_n_permutations > 0:
            print(
                f"Augmenting {split_name} split: {args.dev_test_n_permutations} "
                f"{colour_mode} colour variants (no shifts, no mirror)"
            )
            aug_split = augment_split(
                base[split_name],
                args.dev_test_n_permutations,
                shifts=[],
                rng=rng,
                mirror=False,
                per_pair=args.per_pair,
                canonical_recolor=args.canonical_recolor_colors,
            )
            splits[split_name] = Dataset.from_list(aug_split)
        else:
            splits[split_name] = base[split_name]

    dataset_dict = DatasetDict(splits)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(args.output_dir))

    print(f"\nSaved augmented dataset to {args.output_dir}")
    print(dataset_dict)
    print_split_summary(dataset_dict)


if __name__ == "__main__":
    main()
