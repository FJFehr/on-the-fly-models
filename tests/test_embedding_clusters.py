"""Focused tests for embedding cluster-map diagnostics."""

import matplotlib
import numpy as np
import pytest

from visualisation.embedding_clusters import (
    DEFAULT_GROUP_STYLES,
    _legend_bottom_margin,
    compute_linear_probe_accuracy,
    compute_pca_2d,
    compute_tsne_2d,
    compute_umap_2d,
    render_embedding_cluster_figure,
    render_single_projection_figures,
)


def _two_cluster_vectors(n_per_cluster: int = 20, dim: int = 6, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cluster_a = rng.normal(loc=-5.0, scale=0.1, size=(n_per_cluster, dim))
    cluster_b = rng.normal(loc=5.0, scale=0.1, size=(n_per_cluster, dim))
    return np.concatenate([cluster_a, cluster_b], axis=0)


def test_compute_pca_2d_separates_well_separated_clusters():
    """PCA on two far-apart clusters should keep them far apart in 2D too."""
    vectors = _two_cluster_vectors()
    coords = compute_pca_2d(vectors)

    assert coords.shape == (vectors.shape[0], 2)
    cluster_a_mean = coords[:20].mean(axis=0)
    cluster_b_mean = coords[20:].mean(axis=0)
    assert np.linalg.norm(cluster_a_mean - cluster_b_mean) > 5.0


def test_compute_pca_2d_rejects_non_2d_input():
    with pytest.raises(ValueError, match="num_tasks, dim"):
        compute_pca_2d(np.zeros((4, 3, 2)))


def test_compute_tsne_2d_returns_expected_shape():
    vectors = _two_cluster_vectors(n_per_cluster=10)
    coords = compute_tsne_2d(vectors)
    assert coords.shape == (vectors.shape[0], 2)


def test_compute_tsne_2d_rejects_too_few_samples():
    vectors = _two_cluster_vectors(n_per_cluster=1)
    with pytest.raises(ValueError, match="t-SNE needs at least"):
        compute_tsne_2d(vectors)


def test_compute_umap_2d_returns_expected_shape():
    vectors = _two_cluster_vectors(n_per_cluster=10)
    coords = compute_umap_2d(vectors)
    assert coords.shape == (vectors.shape[0], 2)


def test_compute_umap_2d_rejects_too_few_samples():
    vectors = _two_cluster_vectors(n_per_cluster=1)
    with pytest.raises(ValueError, match="UMAP needs at least"):
        compute_umap_2d(vectors)


def test_render_embedding_cluster_figure_renders_all_three_panels():
    vectors = _two_cluster_vectors(n_per_cluster=10)
    task_categories = ["1d_move_1p"] * 10 + ["1d_fill"] * 10

    figure = render_embedding_cluster_figure(vectors, task_categories, title="test")

    assert isinstance(figure, matplotlib.figure.Figure)
    assert len(figure.axes) == 3
    matplotlib.pyplot.close(figure)


def test_render_embedding_cluster_figure_skips_unavailable_projections():
    """Too few samples for t-SNE/UMAP should still yield a PCA-only figure, not an error."""
    vectors = _two_cluster_vectors(n_per_cluster=1)
    task_categories = ["1d_move_1p", "1d_fill"]

    figure = render_embedding_cluster_figure(vectors, task_categories, title="test")

    assert len(figure.axes) == 1
    matplotlib.pyplot.close(figure)


def test_render_embedding_cluster_figure_rejects_mismatched_labels():
    vectors = _two_cluster_vectors(n_per_cluster=10)
    with pytest.raises(ValueError, match="task_categories length must match"):
        render_embedding_cluster_figure(vectors, ["1d_move_1p"], title="test")


def test_render_embedding_cluster_figure_rejects_mismatched_groups():
    vectors = _two_cluster_vectors(n_per_cluster=10)
    task_categories = ["1d_move_1p"] * 10 + ["1d_fill"] * 10
    with pytest.raises(ValueError, match="groups length must match"):
        render_embedding_cluster_figure(
            vectors, task_categories, title="test", groups=["reference"]
        )


def test_render_embedding_cluster_figure_with_groups_adds_a_group_legend():
    """Passing groups should style reference points as fully opaque circles on top and new
    points as translucent X's behind them (DEFAULT_GROUP_STYLES), plus a second legend for
    the group styling."""
    vectors = _two_cluster_vectors(n_per_cluster=10)
    task_categories = ["1d_move_1p"] * 10 + ["1d_fill"] * 10
    groups = ["reference"] * 15 + ["new"] * 5

    figure = render_embedding_cluster_figure(
        vectors, task_categories, title="test", groups=groups
    )

    # Category legend + group legend, both attached to the figure.
    assert len(figure.legends) == 2
    group_legend_labels = {text.get_text() for text in figure.legends[1].get_texts()}
    assert group_legend_labels == {"reference", "new"}
    matplotlib.pyplot.close(figure)


def test_render_embedding_cluster_figure_without_groups_has_a_single_legend():
    vectors = _two_cluster_vectors(n_per_cluster=10)
    task_categories = ["1d_move_1p"] * 10 + ["1d_fill"] * 10

    figure = render_embedding_cluster_figure(vectors, task_categories, title="test")

    assert len(figure.legends) == 1
    matplotlib.pyplot.close(figure)


def test_render_embedding_cluster_figure_default_group_styles_reference_vs_new():
    # Reference (validation) points draw last, fully opaque, on top; new/held-out points sit
    # behind them at a lower (but still clearly visible) alpha.
    assert DEFAULT_GROUP_STYLES["reference"]["marker"] == "o"
    assert DEFAULT_GROUP_STYLES["reference"]["alpha"] == 1.0
    assert DEFAULT_GROUP_STYLES["reference"]["zorder"] > DEFAULT_GROUP_STYLES["new"]["zorder"]
    assert DEFAULT_GROUP_STYLES["new"]["marker"] == "x"
    assert 0.0 < DEFAULT_GROUP_STYLES["new"]["alpha"] < DEFAULT_GROUP_STYLES["reference"]["alpha"]


def test_render_single_projection_figures_returns_one_figure_per_available_projection():
    vectors = _two_cluster_vectors(n_per_cluster=10)
    task_categories = ["1d_move_1p"] * 10 + ["1d_fill"] * 10

    figures = render_single_projection_figures(vectors, task_categories)

    assert set(figures.keys()) == {"PCA", "t-SNE", "UMAP"}
    for figure in figures.values():
        assert isinstance(figure, matplotlib.figure.Figure)
        assert len(figure.axes) == 1
        matplotlib.pyplot.close(figure)


def test_render_single_projection_figures_skips_unavailable_projections():
    vectors = _two_cluster_vectors(n_per_cluster=1)
    task_categories = ["1d_move_1p", "1d_fill"]

    figures = render_single_projection_figures(vectors, task_categories)

    assert set(figures.keys()) == {"PCA"}
    matplotlib.pyplot.close(figures["PCA"])


def test_render_single_projection_figures_rejects_mismatched_groups():
    vectors = _two_cluster_vectors(n_per_cluster=10)
    task_categories = ["1d_move_1p"] * 10 + ["1d_fill"] * 10
    with pytest.raises(ValueError, match="groups length must match"):
        render_single_projection_figures(vectors, task_categories, groups=["reference"])


def test_render_single_projection_figures_each_have_a_category_legend():
    vectors = _two_cluster_vectors(n_per_cluster=10)
    task_categories = ["1d_move_1p"] * 10 + ["1d_fill"] * 10
    groups = ["reference"] * 15 + ["new"] * 5

    figures = render_single_projection_figures(vectors, task_categories, groups=groups)

    for figure in figures.values():
        assert len(figure.legends) == 2  # category legend + group legend
        matplotlib.pyplot.close(figure)


def test_legend_bottom_margin_grows_with_more_legend_rows():
    """A fixed margin cropped multi-row legends off the bottom of the saved image -- the
    reserved margin must scale with how many rows the category legend wraps onto."""
    single_row = _legend_bottom_margin(n_entries=4, ncol=6)
    five_rows = _legend_bottom_margin(n_entries=25, ncol=6)  # e.g. 15 base + 10 composite categories

    assert five_rows > single_row
    assert 0.0 < single_row < 1.0
    assert 0.0 < five_rows <= 1.0


def test_compute_linear_probe_accuracy_separates_well_separated_clusters():
    """A linear probe should trivially classify two far-apart, low-noise clusters."""
    vectors = _two_cluster_vectors(n_per_cluster=20)
    labels = ["1d_move_1p"] * 20 + ["1d_fill"] * 20

    result = compute_linear_probe_accuracy(vectors, labels)

    assert result["mean_accuracy"] > 0.95
    assert result["n_classes"] == 2
    assert len(result["fold_accuracies"]) == result["n_splits"]


def test_compute_linear_probe_accuracy_near_chance_on_random_labels():
    """Randomly shuffled labels should score far below the well-separated case."""
    vectors = _two_cluster_vectors(n_per_cluster=20)
    rng = np.random.default_rng(1)
    labels = rng.permutation(["1d_move_1p"] * 20 + ["1d_fill"] * 20).tolist()

    result = compute_linear_probe_accuracy(vectors, labels)

    assert result["mean_accuracy"] < 0.9


def test_compute_linear_probe_accuracy_rejects_single_class():
    vectors = _two_cluster_vectors(n_per_cluster=10)
    labels = ["1d_move_1p"] * 20
    with pytest.raises(ValueError, match="at least 2 classes"):
        compute_linear_probe_accuracy(vectors, labels)


def test_compute_linear_probe_accuracy_rejects_singleton_class():
    vectors = _two_cluster_vectors(n_per_cluster=10)
    labels = ["1d_move_1p"] * 19 + ["1d_fill"] * 1
    with pytest.raises(ValueError, match="at least 2 examples per class"):
        compute_linear_probe_accuracy(vectors, labels)
