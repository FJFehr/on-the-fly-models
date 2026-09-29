"""Render paper-quality ARC1D task figures: rounded cells, muted rainbow palette.

Companion to plot_tasks.py, which renders the standard high-contrast
ARC-AGI style used for quick inspection / W&B. This script targets the
LaTeX write-up: vector PDF output via visualisation.paper.arc_paper.
"""

import argparse
from pathlib import Path

from datasets import Dataset, DatasetDict
from matplotlib import pyplot as plt

from visualisation.core.style import apply_latex_style
from visualisation.paper.arc_paper import render_task_figure_paper, use_label_fonts
from visualisation.paper.plot_tasks import (
    filter_tasks,
    is_task_dataset,
    load_task_data,
    select_dataset,
)

# Matches experiments/01_multitask_capacity/plot_per_task.py's font
# override, so task-example figures and per-task result figures set the same
# type in the paper.
PAPER_RC_PARAMS = {
    "font.serif": ["Nimbus Roman", "Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,  # embed as TrueType, not the default Type 3
}

# The multitask experiment (experiments/01_multitask_capacity) only
# trains/evaluates on these 14 categories -- 1d_padded_fill and the three
# 1d_recolor_* variants exist in data/arc_1d but aren't part of that story,
# so --first-per-category skips them rather than rendering paper figures
# nobody references.
PAPER_TASK_CATEGORIES = {
    "1d_denoising_1c", "1d_denoising_mc", "1d_fill", "1d_flip", "1d_hollow",
    "1d_mirror", "1d_move_1p", "1d_move_2p", "1d_move_2p_dp", "1d_move_3p",
    "1d_move_dp", "1d_pcopy_1c", "1d_pcopy_mc", "1d_scale_dp",
}

# The 10 chained-skill categories built by data_modules/arc1d_compositional.py
# and saved as data/arc_1d_compositional_holdout's only split, holdout_test
# (experiments/04_compositional_generalization). Each name maps
# to a display title with a "$\circ$" composition mark (visualisation/style.py's
# TASK_CATEGORY_DISPLAY_NAMES), which render_task_figure_paper renders via the
# multi-artist path in visualisation/arc_paper.py -- no special-casing needed
# here beyond pointing at the right dataset/split and category set.
COMPOSITIONAL_DATA_DIR = "data/arc_1d_compositional_holdout"
COMPOSITIONAL_SPLIT = "holdout_test"
COMPOSITIONAL_TASK_CATEGORIES = {
    "1d_comp_denoise1c_shift3", "1d_comp_denoisemc_copy", "1d_comp_denoisemc_denoise1c",
    "1d_comp_denoisemc_mirror", "1d_comp_fill_mirror", "1d_comp_fill_movedynamic",
    "1d_comp_fill_shift3", "1d_comp_hollow_shift3", "1d_comp_movedynamic_hollow",
    "1d_comp_shift3_copy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/arc_1d")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/figures/00_task_examples")
    )
    parser.add_argument("--task-category", default=None)
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--first-per-category", action="store_true")
    parser.add_argument(
        "--compositional",
        action="store_true",
        help=(
            "Render the 10 compositional-generalisation categories instead of the "
            f"14 base ones, from {COMPOSITIONAL_DATA_DIR} (split {COMPOSITIONAL_SPLIT}). "
            "Overrides --data-dir/--split."
        ),
    )
    parser.add_argument(
        "--num-support",
        type=int,
        default=None,
        help="Truncate to this many support pairs (default: all).",
    )
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="Omit the task-name title above the panels.",
    )
    parser.add_argument(
        "--format",
        default="pdf",
        choices=["pdf", "png"],
        help="Output format. pdf is vector and preferred for LaTeX inclusion.",
    )
    return parser.parse_args()


def visualise_task(
    task: dict, save_path: Path, num_support: int | None, show_title: bool = True
) -> None:
    fig = render_task_figure_paper(task, num_support=num_support, show_title=show_title)
    # transparent=True drops the canvas fill entirely (PNG alpha channel /
    # no PDF background rect) rather than baking in a white rectangle --
    # works the same as white once placed on a white LaTeX page, but also
    # composites cleanly anywhere else.
    fig.savefig(save_path, transparent=True)
    plt.close(fig)


def build_output_path(output_dir: Path, task: dict, fmt: str) -> Path:
    category = task["task_category"]
    task_id = task["task_id"]
    return output_dir / f"arc_1d_task_{category}_{task_id}.{fmt}"


def main() -> None:
    apply_latex_style()
    plt.rcParams.update(PAPER_RC_PARAMS)
    # apply_latex_style() sets savefig.bbox="tight", which would crop every
    # figure back down to its drawn content -- undoing PANEL_WIDTH's fixed
    # canvas size for any sequence shorter than the reference length. Put it
    # back to the literal figsize.
    plt.rcParams["savefig.bbox"] = None
    use_label_fonts()
    args = parse_args()
    data_dir = COMPOSITIONAL_DATA_DIR if args.compositional else args.data_dir
    split = COMPOSITIONAL_SPLIT if args.compositional else args.split
    data: Dataset | DatasetDict = load_task_data(data_dir)
    ds = select_dataset(data, split)

    if not is_task_dataset(ds):
        msg = "This script expects the task-level dataset produced by build_arc_1d.py."
        raise ValueError(msg)

    selected_tasks = filter_tasks(
        ds,
        task_category=args.task_category,
        task_id=args.task_id,
        first_per_category=args.first_per_category,
        max_tasks=args.max_tasks,
    )
    # An explicit --task-category always renders whatever was asked for;
    # only the "give me one of everything" mode restricts to the categories
    # the paper actually uses (base 14, or the 10 compositional ones).
    if args.task_category is None:
        allowed = COMPOSITIONAL_TASK_CATEGORIES if args.compositional else PAPER_TASK_CATEGORIES
        selected_tasks = [t for t in selected_tasks if t["task_category"] in allowed]
    if not selected_tasks:
        msg = "No tasks matched the requested filters."
        raise ValueError(msg)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for task in selected_tasks:
        output_path = build_output_path(args.output_dir, task, args.format)
        visualise_task(
            task, output_path, num_support=args.num_support, show_title=not args.no_title
        )
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
