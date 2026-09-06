#!/usr/bin/env python3
"""Inspect augmentation quality per task category.

For each task category prints:
  - 5 original training examples (from data/arc_1d)
  - 5 random augmented training examples (from the augmented dataset)
  - 5 dev examples
  - 5 test examples

Pipe through `less -R` to page with colour:
    python scripts/visualize_augmentation.py | less -R
    python scripts/visualize_augmentation.py --task 1d_move_1p | less -R
"""

import argparse
import random
from pathlib import Path

import pyarrow.compute as pc
from datasets import load_from_disk

# 256-colour ANSI background codes keyed by ARC colour value (0–9)
_BG = {
    0: "\033[48;5;234m",  # near-black (background)
    1: "\033[48;5;21m",   # blue
    2: "\033[48;5;196m",  # red
    3: "\033[48;5;34m",   # green
    4: "\033[48;5;226m",  # yellow
    5: "\033[48;5;244m",  # grey
    6: "\033[48;5;201m",  # magenta
    7: "\033[48;5;208m",  # orange
    8: "\033[48;5;117m",  # light blue
    9: "\033[48;5;88m",   # maroon
}
_RST = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"


def _cell(c: int) -> str:
    return f"{_BG[c]}  {_RST}"


def _seq(seq: list[int]) -> str:
    return "".join(_cell(c) for c in seq)


def _show_example(ex: dict) -> None:
    base_id = ex["task_id"] // 10000 if ex["task_id"] > 9999 else ex["task_id"]
    print(f"  {_DIM}id={base_id}  len={ex['sequence_length']}{_RST}")
    for i, (si, so) in enumerate(zip(ex["support_inputs"], ex["support_outputs"])):
        print(f"    p{i + 1}: {_seq(si)}  →  {_seq(so)}")
    print(f"      q: {_seq(ex['query_input'])}  →  {_seq(ex['query_output'])}")
    print()


def _section(title: str) -> None:
    print(f"  {_BOLD}{title}{_RST}")
    print(f"  {'·' * 60}")


def _index_by_cat(split) -> dict[str, list[int]]:
    """Build category → row-index mapping via PyArrow (fast, no Python row iteration)."""
    col = split._data["task_category"].combine_chunks()
    unique = pc.unique(col).to_pylist()
    return {
        cat: pc.indices_nonzero(pc.equal(col, cat)).to_pylist()
        for cat in unique
    }


def _sample_examples(split, indices_by_cat: dict, cat: str, n: int, rng: random.Random) -> list[dict]:
    indices = indices_by_cat.get(cat, [])
    chosen = rng.sample(indices, min(n, len(indices)))
    chosen.sort()  # select() requires sorted indices
    return [split[i] for i in chosen]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path("data/arc_1d"))
    parser.add_argument("--aug-dir", type=Path, default=Path("data/arc_1d_all_tasks_augmented"))
    parser.add_argument("--task", metavar="CATEGORY", default=None,
                        help="Show only this task category (e.g. 1d_move_1p).")
    parser.add_argument("-n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    base = load_from_disk(str(args.base_dir))
    aug = load_from_disk(str(args.aug_dir))

    # Index by category using fast columnar reads only
    base_train_idx = _index_by_cat(base["train"])
    aug_train_idx = _index_by_cat(aug["train"])   # reads 681k strings, no row deserialisation
    aug_dev_idx = _index_by_cat(aug["dev"])
    aug_test_idx = _index_by_cat(aug["test"])

    categories = sorted(aug_dev_idx.keys())
    if args.task:
        if args.task not in categories:
            print(f"Unknown task '{args.task}'. Available: {', '.join(categories)}")
            return
        categories = [args.task]

    for cat in categories:
        print(f"\n{'═' * 66}")
        print(f"{_BOLD}  {cat}{_RST}")
        print(f"{'═' * 66}\n")

        n_orig = len(base_train_idx.get(cat, []))
        _section(f"ORIGINAL TRAINING  ({n_orig} tasks)")
        for ex in _sample_examples(base["train"], base_train_idx, cat, args.n, rng):
            _show_example(ex)

        n_aug = len(aug_train_idx.get(cat, []))
        _section(f"AUGMENTED TRAINING  ({n_aug} tasks, {args.n} random samples)")
        for ex in _sample_examples(aug["train"], aug_train_idx, cat, args.n, rng):
            _show_example(ex)

        n_dev = len(aug_dev_idx.get(cat, []))
        _section(f"DEV  ({n_dev} examples)")
        for ex in _sample_examples(aug["dev"], aug_dev_idx, cat, args.n, rng):
            _show_example(ex)

        n_test = len(aug_test_idx.get(cat, []))
        _section(f"TEST  ({n_test} examples)")
        for ex in _sample_examples(aug["test"], aug_test_idx, cat, args.n, rng):
            _show_example(ex)


if __name__ == "__main__":
    main()
