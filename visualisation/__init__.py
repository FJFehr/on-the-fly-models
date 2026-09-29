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

from visualisation.core.arc import (
    figure_to_wandb_image,
    render_task_figure,
    render_task_prediction_figure,
    render_val_example_figure,
)
from visualisation.core.style import (
    FONT_SIZES,
    MODEL_COLORS,
    MODEL_DISPLAY_NAMES,
    TASK_CATEGORY_DISPLAY_NAMES,
    apply_latex_style,
    format_task_category,
    normalize_task_category,
)

__all__ = [
    "FONT_SIZES",
    "MODEL_COLORS",
    "MODEL_DISPLAY_NAMES",
    "TASK_CATEGORY_DISPLAY_NAMES",
    "apply_latex_style",
    "figure_to_wandb_image",
    "format_task_category",
    "normalize_task_category",
    "render_task_figure",
    "render_task_prediction_figure",
    "render_val_example_figure",
]
