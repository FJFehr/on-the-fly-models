# visualisation/__init__.py
# ---------------------------------------------------------------------------
# Public API for ARC visualisation utilities.
#
# This package centralises all ARC rendering and figure-building code.
# Internal drawing primitives (draw_sequence, draw_masked_sequence, etc.)
# are kept private inside arc.py — only the high-level functions that
# other packages actually need are re-exported here.
#
# Consumers should import from this package, not from the submodule:
#   from visualisation import render_task_figure, figure_to_wandb_image
# ---------------------------------------------------------------------------

from visualisation.arc import (
    figure_to_wandb_image,
    format_task_category,
    render_task_figure,
    render_val_example_figure,
)

__all__ = [
    "figure_to_wandb_image",
    "format_task_category",
    "render_task_figure",
    "render_val_example_figure",
]
