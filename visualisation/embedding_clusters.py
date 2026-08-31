"""Cluster-map diagnostics for the pooled hypernetwork task latent.

Projects per-task pooled embeddings to 2D via PCA, t-SNE, and UMAP, colored by task
category, as a visual check for disentanglement between task categories. Used by
``training.logging.log_embedding_cluster_plots`` as an end-of-run artifact.
"""

from __future__ import annotations

import math

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold, cross_val_score
from umap import UMAP

from visualisation.style import format_task_category

MIN_TSNE_SAMPLES = 4
MIN_UMAP_SAMPLES = 3
MIN_LINEAR_PROBE_SAMPLES_PER_CLASS = 2


def _validate_vectors(vectors: np.ndarray) -> None:
    if vectors.ndim != 2:
        msg = f"vectors must have shape (num_tasks, dim), got {vectors.shape}."
        raise ValueError(msg)


def compute_pca_2d(vectors: np.ndarray) -> np.ndarray:
    """Project vectors to 2D via PCA."""
    _validate_vectors(vectors)
    return PCA(n_components=2, random_state=0).fit_transform(vectors)


def compute_tsne_2d(vectors: np.ndarray, *, perplexity: float | None = None) -> np.ndarray:
    """Project vectors to 2D via t-SNE.

    Raises ValueError when there are too few samples for a meaningful embedding
    (perplexity must stay below the sample count) so callers can skip this panel
    instead of crashing the whole figure.
    """
    _validate_vectors(vectors)
    n_samples = vectors.shape[0]
    if n_samples < MIN_TSNE_SAMPLES:
        msg = f"t-SNE needs at least {MIN_TSNE_SAMPLES} samples, got {n_samples}."
        raise ValueError(msg)
    resolved_perplexity = min(perplexity or 30.0, n_samples - 1)
    return TSNE(
        n_components=2,
        perplexity=resolved_perplexity,
        init="pca",
        random_state=0,
    ).fit_transform(vectors)


def compute_umap_2d(vectors: np.ndarray, *, n_neighbors: int | None = None) -> np.ndarray:
    """Project vectors to 2D via UMAP.

    Raises ValueError when there are too few samples for a meaningful embedding, same
    reasoning as compute_tsne_2d.
    """
    _validate_vectors(vectors)
    n_samples = vectors.shape[0]
    if n_samples < MIN_UMAP_SAMPLES:
        msg = f"UMAP needs at least {MIN_UMAP_SAMPLES} samples, got {n_samples}."
        raise ValueError(msg)
    resolved_n_neighbors = min(n_neighbors or 15, n_samples - 1)
    return UMAP(
        n_components=2,
        n_neighbors=resolved_n_neighbors,
        random_state=0,
    ).fit_transform(vectors)


_PROJECTIONS = (
    ("PCA", compute_pca_2d),
    ("t-SNE", compute_tsne_2d),
    ("UMAP", compute_umap_2d),
)


#: Default per-group scatter styling for the two-group overlay (see render_embedding_cluster_figure's
#: `groups` argument): the reference (validation) points are drawn last, fully opaque, on top;
#: the new/held-out points sit behind them at a higher-than-normal alpha (translucent, but still
#: clearly visible) so you can see where they fall relative to the solid reference clusters
#: without them competing with or obscuring the reference points themselves.
DEFAULT_GROUP_STYLES: dict[str, dict[str, object]] = {
    "reference": {"marker": "o", "alpha": 1.0, "zorder": 3},
    "new": {"marker": "x", "alpha": 0.5, "zorder": 1},
}


def _compute_projection_panels(vectors: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Compute every projection that's viable for this many samples.

    Projections that can't be computed for the given sample count (e.g. too few
    validation tasks for t-SNE/UMAP) are silently skipped rather than failing outright --
    a small validation set still yields a PCA panel.
    """
    panels: list[tuple[str, np.ndarray]] = []
    for name, compute_fn in _PROJECTIONS:
        try:
            coords = compute_fn(vectors)
        except ValueError:
            continue
        panels.append((name, coords))
    if not panels:
        msg = "No projection could be computed for the given vectors."
        raise ValueError(msg)
    return panels


def _validate_cluster_inputs(
    vectors: np.ndarray, task_categories: list[str], groups: list[str] | None
) -> None:
    _validate_vectors(vectors)
    if len(task_categories) != vectors.shape[0]:
        msg = (
            "task_categories length must match vectors.shape[0], got "
            f"{len(task_categories)} and {vectors.shape[0]}."
        )
        raise ValueError(msg)
    if groups is not None and len(groups) != vectors.shape[0]:
        msg = f"groups length must match vectors.shape[0], got {len(groups)} and {vectors.shape[0]}."
        raise ValueError(msg)


def _legend_bottom_margin(n_entries: int, ncol: int) -> float:
    """Fraction of figure height to reserve below the axes for the category legend, scaled
    to how many rows it will actually wrap onto -- a fixed margin cropped multi-row legends
    (e.g. 25 categories at ncol=6 -> 5 rows) off the bottom of the saved/uploaded image."""
    n_rows = max(1, math.ceil(n_entries / ncol))
    return min(0.06 + 0.045 * n_rows, 0.45)


def _draw_scatter_panel(
    axis: plt.Axes,
    coords: np.ndarray,
    display_categories: list[str],
    group_values: list[str],
    unique_categories: list[str],
    unique_groups: list[str],
    color_by_category: dict[str, object],
    resolved_group_styles: dict[str, dict[str, object]],
    groups_enabled: bool,
    handles_by_category: dict[str, object],
    handles_by_group: dict[str, object],
) -> None:
    """Scatter one projection's points onto `axis`, styled by category (colour) and group
    (marker/alpha/z-order), accumulating legend handles into the two dicts passed in."""
    for group in unique_groups:
        style = {"s": 18, "alpha": 0.75} | (resolved_group_styles.get(group, {}) if groups_enabled else {})
        for category in unique_categories:
            mask = np.asarray(
                [dc == category and gv == group for dc, gv in zip(display_categories, group_values, strict=True)]
            )
            points = coords[mask]
            if points.shape[0] == 0:
                continue
            scatter = axis.scatter(
                points[:, 0], points[:, 1], color=color_by_category[category], label=category, **style
            )
            handles_by_category.setdefault(category, scatter)
            if groups_enabled:
                handles_by_group.setdefault(group, axis.scatter([], [], color="grey", label=group, **style))
    axis.set_xlabel("dim 1")
    axis.set_ylabel("dim 2")
    axis.grid(alpha=0.2)


def _add_legends(
    figure: plt.Figure,
    handles_by_category: dict[str, object],
    handles_by_group: dict[str, object],
    *,
    groups_enabled: bool,
    ncol: int,
) -> float:
    """Attach the category legend (and group legend, if applicable) to `figure`; returns the
    bottom margin the caller should reserve via tight_layout's `rect` so the legend doesn't
    get cropped off the saved/uploaded image."""
    category_legend = figure.legend(
        handles_by_category.values(),
        handles_by_category.keys(),
        loc="lower center",
        ncol=ncol,
        bbox_to_anchor=(0.5, -0.02),
        title="task category",
    )
    if groups_enabled:
        figure.add_artist(category_legend)
        figure.legend(
            handles_by_group.values(),
            handles_by_group.keys(),
            loc="upper right",
            bbox_to_anchor=(1.0, 1.02),
            title="group",
        )
    return _legend_bottom_margin(len(handles_by_category), ncol)


def _resolve_common_styling(
    task_categories: list[str], groups: list[str] | None, group_styles: dict[str, dict[str, object]] | None
) -> tuple[list[str], list[str], dict[str, object], dict[str, dict[str, object]], list[str], list[str]]:
    display_categories = [format_task_category(category) for category in task_categories]
    unique_categories = sorted(set(display_categories))
    colormap = matplotlib.colormaps["tab20"]
    color_by_category = {
        category: colormap(index % colormap.N) for index, category in enumerate(unique_categories)
    }
    resolved_group_styles = group_styles or DEFAULT_GROUP_STYLES
    group_values = groups if groups is not None else ["__all__"] * len(task_categories)
    unique_groups = sorted(set(group_values))
    return display_categories, unique_categories, color_by_category, resolved_group_styles, group_values, unique_groups


def render_embedding_cluster_figure(
    vectors: np.ndarray,
    task_categories: list[str],
    *,
    title: str,
    groups: list[str] | None = None,
    group_styles: dict[str, dict[str, object]] | None = None,
) -> plt.Figure:
    """Render PCA / t-SNE / UMAP scatter subplots of vectors, colored by task category, as
    one combined multi-panel figure.

    `vectors`/`task_categories` should already include every group's points (e.g. both
    reference validation embeddings and held-out/new embeddings) -- projections are fit
    once on the combined set, since t-SNE/UMAP don't cleanly support projecting new points
    into a separately-fitted embedding. Pass `groups` (same length as `task_categories`,
    e.g. "reference"/"new") to additionally style points by group -- category still
    controls colour, group controls marker/alpha/z-order (DEFAULT_GROUP_STYLES: reference
    drawn last and fully opaque, new/held-out drawn first at a lower alpha, behind it).
    Omit `groups` (default) for the original single-style behaviour.

    See also `render_single_projection_figures`, which renders each projection as its own
    standalone figure (e.g. for logging PCA/t-SNE/UMAP to W&B separately).
    """
    _validate_cluster_inputs(vectors, task_categories, groups)
    panels = _compute_projection_panels(vectors)
    display_categories, unique_categories, color_by_category, resolved_group_styles, group_values, unique_groups = (
        _resolve_common_styling(task_categories, groups, group_styles)
    )
    groups_enabled = groups is not None

    figure, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5), squeeze=False)
    axes = axes[0]

    handles_by_category: dict[str, object] = {}
    handles_by_group: dict[str, object] = {}
    for axis, (name, coords) in zip(axes, panels, strict=True):
        _draw_scatter_panel(
            axis, coords, display_categories, group_values, unique_categories, unique_groups,
            color_by_category, resolved_group_styles, groups_enabled, handles_by_category, handles_by_group,
        )
        axis.set_title(name)

    figure.suptitle(title)
    ncol = min(len(unique_categories), 6)
    bottom = _add_legends(figure, handles_by_category, handles_by_group, groups_enabled=groups_enabled, ncol=ncol)
    figure.tight_layout(rect=(0, bottom, 1, 1))
    return figure


def render_single_projection_figures(
    vectors: np.ndarray,
    task_categories: list[str],
    *,
    groups: list[str] | None = None,
    group_styles: dict[str, dict[str, object]] | None = None,
) -> dict[str, plt.Figure]:
    """Render each available projection (PCA / t-SNE / UMAP) as its own standalone figure,
    keyed by projection name -- e.g. so each can be logged to W&B as a separate image,
    rather than only ever appearing as one panel within the combined figure (see
    render_embedding_cluster_figure). Same styling/arguments as that function.
    """
    _validate_cluster_inputs(vectors, task_categories, groups)
    panels = _compute_projection_panels(vectors)
    display_categories, unique_categories, color_by_category, resolved_group_styles, group_values, unique_groups = (
        _resolve_common_styling(task_categories, groups, group_styles)
    )
    groups_enabled = groups is not None
    ncol = min(len(unique_categories), 6)

    figures: dict[str, plt.Figure] = {}
    for name, coords in panels:
        figure, axis = plt.subplots(figsize=(7, 5.5))
        handles_by_category: dict[str, object] = {}
        handles_by_group: dict[str, object] = {}
        _draw_scatter_panel(
            axis, coords, display_categories, group_values, unique_categories, unique_groups,
            color_by_category, resolved_group_styles, groups_enabled, handles_by_category, handles_by_group,
        )
        axis.set_title(name)
        bottom = _add_legends(figure, handles_by_category, handles_by_group, groups_enabled=groups_enabled, ncol=ncol)
        figure.tight_layout(rect=(0, bottom, 1, 1))
        figures[name] = figure
    return figures


def compute_linear_probe_accuracy(
    vectors: np.ndarray,
    labels: list[str],
    *,
    max_splits: int = 5,
    random_state: int = 0,
) -> dict[str, object]:
    """Cross-validated multinomial logistic-regression accuracy for task-category
    classification from the pooled embedding -- a quantitative disentanglement probe
    to complement the PCA/t-SNE/UMAP cluster map: if a linear model already separates
    the categories well, that's a strong, cheap signal the representation is
    disentangled; if not, the qualitative clusters may just be a projection artifact.

    Raises ValueError if fewer than 2 classes are present, or any class has fewer than
    MIN_LINEAR_PROBE_SAMPLES_PER_CLASS examples (the minimum needed for stratified
    k-fold cross-validation).
    """
    _validate_vectors(vectors)
    if len(labels) != vectors.shape[0]:
        msg = (
            "labels length must match vectors.shape[0], got "
            f"{len(labels)} and {vectors.shape[0]}."
        )
        raise ValueError(msg)

    unique_labels, counts = np.unique(labels, return_counts=True)
    if len(unique_labels) < 2:
        msg = f"Linear probe needs at least 2 classes, got {len(unique_labels)}."
        raise ValueError(msg)
    if counts.min() < MIN_LINEAR_PROBE_SAMPLES_PER_CLASS:
        msg = (
            "Linear probe needs at least "
            f"{MIN_LINEAR_PROBE_SAMPLES_PER_CLASS} examples per class, got a class "
            f"with only {int(counts.min())}."
        )
        raise ValueError(msg)

    n_splits = min(max_splits, int(counts.min()))
    cross_validator = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    classifier = LogisticRegression(max_iter=1000)
    fold_accuracies = cross_val_score(classifier, vectors, labels, cv=cross_validator)

    return {
        "mean_accuracy": float(fold_accuracies.mean()),
        "fold_accuracies": fold_accuracies.tolist(),
        "n_splits": n_splits,
        "n_classes": int(len(unique_labels)),
    }
