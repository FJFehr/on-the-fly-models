"""Regression tests for task-category label normalization."""

from visualisation.core.style import format_task_category, normalize_task_category


def test_normalize_task_category_strips_multiclass_prefix():
    """Multiclass capacity outputs prefix task keys with `mc_`; labels should not."""
    assert normalize_task_category("mc_1d_move_1p") == "1d_move_1p"


def test_format_task_category_uses_normalized_multiclass_key():
    """Display labels should resolve against the canonical key, not the prefixed run name."""
    assert format_task_category("mc_1d_move_1p") == "Move 1 Pixel"
