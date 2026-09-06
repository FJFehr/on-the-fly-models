"""Focused tests for seeded capacity plot aggregation."""

from visualisation.plot_capacity import aggregate_seed_metrics, infer_task, split_seed_suffix


def test_split_seed_suffix_extracts_base_name_and_seed():
    """Seeded run directories should parse back to their canonical experiment name."""
    assert split_seed_suffix("mc_1d_move_1p_rnn_seed_43") == ("mc_1d_move_1p_rnn", 43)


def test_infer_task_handles_multiclass_seeded_runs():
    """Task normalization should remove both the `mc_` prefix and seed suffix upstream."""
    assert infer_task("mc_1d_move_1p_rnn", "rnn") == "1d_move_1p"


def test_aggregate_seed_metrics_returns_max_mean_and_std():
    """Aggregated plotting stats should reflect best seed and mean±std across seeds."""
    values = [0.4, 0.8, 1.0]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    aggregated = aggregate_seed_metrics(
        [
            {"task": "1d_move_1p", "model": "rnn", "seed": 42, "val_query_exact_match": 0.4},
            {"task": "1d_move_1p", "model": "rnn", "seed": 43, "val_query_exact_match": 0.8},
            {"task": "1d_move_1p", "model": "rnn", "seed": 44, "val_query_exact_match": 1.0},
        ]
    )

    stats = aggregated[("1d_move_1p", "rnn")]
    assert stats["max"] == 1.0
    assert round(stats["mean"], 4) == round(mean, 4)
    assert round(stats["std"], 4) == round(variance**0.5, 4)
