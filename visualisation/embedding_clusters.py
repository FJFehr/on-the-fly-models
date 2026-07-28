"""Cluster-map diagnostics for the pooled hypernetwork task latent.

Projects per-task pooled embeddings to 2D via PCA, t-SNE, and UMAP, colored by task
category, as a visual check for disentanglement between task categories. Used by
``training.logging.log_embedding_cluster_plots`` as an end-of-run artifact.
"""

from __future__ import annotations

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


def render_embedding_cluster_figure(
    vectors: np.ndarray,
    task_categories: list[str],
    *,
    title: str,
) -> plt.Figure:
    """Render PCA / t-SNE / UMAP scatter subplots of vectors, colored by task category.

    Projections that can't be computed for the given sample count (e.g. too few
    validation tasks for t-SNE/UMAP) are silently skipped rather than failing the whole
    figure -- a small validation set still yields a PCA panel.
    """
    _validate_vectors(vectors)
    if len(task_categories) != vectors.shape[0]:
        msg = (
            "task_categories length must match vectors.shape[0], got "
            f"{len(task_categories)} and {vectors.shape[0]}."
        )
        raise ValueError(msg)

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

    display_categories = [format_task_category(category) for category in task_categories]
    unique_categories = sorted(set(display_categories))
    colormap = matplotlib.colormaps["tab20"]
    color_by_category = {
        category: colormap(index % colormap.N) for index, category in enumerate(unique_categories)
    }

    figure, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5), squeeze=False)
    axes = axes[0]

    handles_by_category: dict[str, object] = {}
    for axis, (name, coords) in zip(axes, panels, strict=True):
        for category in unique_categories:
            mask = np.asarray([dc == category for dc in display_categories])
            points = coords[mask]
            if points.shape[0] == 0:
                continue
            scatter = axis.scatter(
                points[:, 0],
                points[:, 1],
                s=18,
                alpha=0.75,
                color=color_by_category[category],
                label=category,
            )
            handles_by_category.setdefault(category, scatter)
        axis.set_title(name)
        axis.set_xlabel("dim 1")
        axis.set_ylabel("dim 2")
        axis.grid(alpha=0.2)

    figure.suptitle(title)
    figure.legend(
        handles_by_category.values(),
        handles_by_category.keys(),
        loc="lower center",
        ncol=min(len(unique_categories), 6),
        bbox_to_anchor=(0.5, -0.05),
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    return figure


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
