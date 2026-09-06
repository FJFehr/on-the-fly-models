"""Visualise the effect of each data augmentation on a given ARC-1D task category.

Picks 2 random train tasks and renders a side-by-side comparison
(Original left | Augmented right) for each of the two transforms:
colour permutation and shift.

Usage:
    uv run python scripts/visualise_augmentation.py --task-category 1d_move_1p
    uv run python scripts/visualise_augmentation.py --task-category 1d_recolor_cmp \\
        --output-dir /tmp/aug_vis --shift 3
"""

import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from datasets import load_from_disk
from matplotlib import pyplot as plt

from visualisation.core.arc import draw_sequence
from visualisation.core.style import FONT_SIZES, format_task_category

INPUT_DIR = Path("data/arc_1d")

# Ordered labels for the 8 sequences in a task.
SEQ_LABELS = [
    "S1 In",
    "S1 Out",
    "S2 In",
    "S2 Out",
    "S3 In",
    "S3 Out",
    "Query In",
    "Query Out",
]


# ---------------------------------------------------------------------------
# Augmentation helpers (inlined from augment_arc_1d.py)
# ---------------------------------------------------------------------------


def _get_task_colors(task: dict) -> list[int]:
    seen: set[int] = set()
    for seq in task["support_inputs"] + task["support_outputs"]:
        seen.update(seq)
    seen.update(task["query_input"])
    seen.update(task["query_output"])
    return sorted(c for c in seen if c != 0)


def _apply_color_map(task: dict, color_map: dict[int, int]) -> dict:
    def remap(seq: list[int]) -> list[int]:
        return [color_map.get(v, v) for v in seq]

    return {
        **task,
        "support_inputs": [remap(s) for s in task["support_inputs"]],
        "support_outputs": [remap(s) for s in task["support_outputs"]],
        "query_input": remap(task["query_input"]),
        "query_output": remap(task["query_output"]),
    }


def _color_permuted(task: dict, rng: random.Random) -> dict | None:
    """Return one colour-permuted variant, or None if no non-zero colours exist."""
    colors = _get_task_colors(task)
    if not colors:
        return None
    available = list(range(1, 10))
    rng.shuffle(available)
    color_map = {orig: available[i] for i, orig in enumerate(colors)}
    return _apply_color_map(task, color_map)


def _apply_shift(task: dict, shift: int) -> dict:
    """Positive shift prepends zeros; negative appends zeros. Extends sequence_length."""
    if shift == 0:
        return task
    zeros = [0] * abs(shift)
    if shift > 0:

        def do_shift(seq: list[int]) -> list[int]:
            return zeros + list(seq)
    else:

        def do_shift(seq: list[int]) -> list[int]:
            return list(seq) + zeros

    return {
        **task,
        "sequence_length": task["sequence_length"] + abs(shift),
        "support_inputs": [do_shift(s) for s in task["support_inputs"]],
        "support_outputs": [do_shift(s) for s in task["support_outputs"]],
        "query_input": do_shift(task["query_input"]),
        "query_output": do_shift(task["query_output"]),
    }


# ---------------------------------------------------------------------------
# Sequence extraction
# ---------------------------------------------------------------------------


def _task_sequences(task: dict) -> list[list[int]]:
    """Return all 8 sequences in the order matching SEQ_LABELS."""
    return [
        task["support_inputs"][0],
        task["support_outputs"][0],
        task["support_inputs"][1],
        task["support_outputs"][1],
        task["support_inputs"][2],
        task["support_outputs"][2],
        task["query_input"],
        task["query_output"],
    ]


# ---------------------------------------------------------------------------
# Figure rendering
# ---------------------------------------------------------------------------


def _render_comparison(
    orig_tasks: list[dict],
    aug_tasks: list[dict],
    aug_label: str,
    task_category: str,
) -> plt.Figure:
    """Render 2 side-by-side comparisons (Original | Augmented) in one figure."""
    n_examples = len(orig_tasks)
    n_seqs = len(SEQ_LABELS)

    orig_seqs = [_task_sequences(t) for t in orig_tasks]
    aug_seqs = [_task_sequences(t) for t in aug_tasks]

    max_orig = max(len(s) for seqs in orig_seqs for s in seqs)
    max_aug = max(len(s) for seqs in aug_seqs for s in seqs)

    cell_w = 0.38  # inches per token cell
    row_h = 0.52  # inches per sequence row
    spacer_ratio = 0.7  # height of spacer row relative to a normal row

    panel_w_orig = max(max_orig * cell_w, 4.0)
    panel_w_aug = max(max_aug * cell_w, 4.0)

    total_rows = n_examples * n_seqs + (n_examples - 1)
    height_ratios = [1] * n_seqs + [spacer_ratio] + [1] * n_seqs

    fig_w = panel_w_orig + panel_w_aug + 0.5
    fig_h = n_examples * n_seqs * row_h + spacer_ratio * row_h + 1.1

    fig, axes = plt.subplots(
        total_rows,
        2,
        figsize=(fig_w, fig_h),
        gridspec_kw={
            "width_ratios": [panel_w_orig, panel_w_aug],
            "height_ratios": height_ratios,
            "hspace": 0.9,
            "wspace": 0.06,
        },
    )

    for ex_idx in range(n_examples):
        row_start = ex_idx * (n_seqs + 1)
        for seq_idx in range(n_seqs):
            row = row_start + seq_idx
            draw_sequence(axes[row, 0], orig_seqs[ex_idx][seq_idx], SEQ_LABELS[seq_idx])
            draw_sequence(axes[row, 1], aug_seqs[ex_idx][seq_idx], SEQ_LABELS[seq_idx])

    # Hide the spacer row between examples
    if n_examples > 1:
        axes[n_seqs, 0].axis("off")
        axes[n_seqs, 1].axis("off")

    cat_display = format_task_category(task_category)
    fig.suptitle(
        f"{aug_label} — {cat_display}\nOriginal (left)   ·   Augmented (right)",
        fontsize=FONT_SIZES["title"],
        fontweight="bold",
        y=1.0,
    )
    fig.subplots_adjust(top=0.91, bottom=0.01, left=0.01, right=0.99)
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-category",
        required=True,
        help="Task category to visualise (e.g. 1d_move_1p).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=INPUT_DIR,
        help="Base ARC-1D dataset directory (default: data/arc_1d).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--shift",
        type=int,
        default=2,
        help="Shift amount (in tokens) used for the shift examples.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Save PNGs here. Omit to show interactively.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    print(f"Loading {args.data_dir} ...")
    ds = load_from_disk(str(args.data_dir))
    train = ds["train"]

    # Filter to the requested category
    tasks = [t for t in train if t["task_category"] == args.task_category]
    if len(tasks) < 2:
        raise ValueError(
            f"Found only {len(tasks)} tasks for category {args.task_category!r}. Need at least 2."
        )
    chosen = rng.sample(tasks, 2)
    print(f"Selected task IDs: {[t['task_id'] for t in chosen]}")

    figures: list[tuple[str, plt.Figure]] = []

    # --- Colour ---
    aug_colour = [_color_permuted(t, rng) for t in chosen]
    if any(a is None for a in aug_colour):
        print("Warning: one or more tasks have no non-zero colours — skipping colour figure.")
    else:
        fig = _render_comparison(chosen, aug_colour, "Colour permutation", args.task_category)
        figures.append((f"colour_{args.task_category}", fig))

    # --- Shift ---
    aug_shift = [_apply_shift(t, args.shift) for t in chosen]
    fig = _render_comparison(
        chosen,
        aug_shift,
        f"Shift +{args.shift} (prepend zeros)",
        args.task_category,
    )
    figures.append((f"shift_{args.task_category}", fig))

    # Save or show
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in figures:
            path = args.output_dir / f"{name}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {path}")
    else:
        matplotlib.use("TkAgg")
        for _, fig in figures:
            plt.figure(fig.number)
            plt.show()


if __name__ == "__main__":
    main()
