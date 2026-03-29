"""Tests for task-level ARC visualization helpers."""

import wandb
from visualisation import figure_to_wandb_image, render_task_prediction_figure


def test_render_task_prediction_figure_renders_three_rows_and_four_columns():
    figure = render_task_prediction_figure(
        support_inputs=[[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]],
        support_targets=[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]],
        support_predictions=[[1, 1, 1, 1], [2, 2, 0, 2], [3, 3, 3, 3]],
        query_input=[4, 4, 4, 4],
        query_target=[5, 5, 5, 5],
        query_prediction=[5, 5, 4, 5],
        task_category="1d_move_1p",
        task_id=8,
        query_exact_match=False,
    )

    assert len(figure.axes) == 12
    assert figure.axes[0].get_title() == "Support 1 Input"
    assert figure.axes[4].get_title() == "Support 1 Target"
    assert figure.axes[8].get_title() == "Support 1 Prediction"
    assert isinstance(figure_to_wandb_image(figure), wandb.Image)
