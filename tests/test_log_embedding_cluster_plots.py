"""log_embedding_cluster_plots should dump the raw embeddings alongside its figure."""

import numpy as np

from training.logging import log_embedding_cluster_plots


class FakeDatamodule:
    def __init__(self, batches: list[dict]) -> None:
        self._batches = batches

    def val_dataloader(self):
        return self._batches


def test_log_embedding_cluster_plots_dumps_an_npz_alongside_the_figure(
    tmp_path, build_hypernetwork, make_hypernetwork_batch
):
    model = build_hypernetwork(log_embedding_clusters=True)
    # UMAP's default n_neighbors needs more points than a couple of tiny batches --
    # enough samples here for render_single_projection_figures to run to completion.
    categories = ["1d_move_1p", "1d_flip"] * 10
    batches = [
        make_hypernetwork_batch(n=1, task_category=category, task_id=task_id)
        for task_id, category in enumerate(categories)
    ]
    datamodule = FakeDatamodule(batches)

    log_embedding_cluster_plots(model, datamodule, output_path=str(tmp_path))

    npz_path = tmp_path / "embedding_clusters" / "embeddings.npz"
    assert npz_path.exists()

    data = np.load(npz_path, allow_pickle=True)
    assert set(data.files) == {"vectors", "task_categories", "task_ids"}
    assert data["vectors"].shape == (len(batches), model.hypernetwork.pooler.query.shape[-1])
    assert list(data["task_categories"]) == categories
    assert list(data["task_ids"]) == list(range(len(batches)))


def test_log_embedding_cluster_plots_npz_excludes_holdout_records(
    tmp_path, build_hypernetwork, make_hypernetwork_batch
):
    """The dumped .npz is the plain reference set, even when a holdout overlay is requested --
    holdout points get mixed into the *figure* but shouldn't pollute the reusable dump other
    scripts (visualisation.paper.plot_embedding_clusters et al.) read as "this run's
    embeddings"."""
    model = build_hypernetwork(log_embedding_clusters=True)
    reference_batches = [make_hypernetwork_batch(n=1, task_category="1d_move_1p", task_id=1)]
    holdout_batches = [make_hypernetwork_batch(n=1, task_category="1d_comp_fill_mirror", task_id=99)]
    datamodule = FakeDatamodule(reference_batches)

    log_embedding_cluster_plots(
        model,
        datamodule,
        output_path=str(tmp_path),
        holdout_dataloader=holdout_batches,
    )

    data = np.load(tmp_path / "embedding_clusters" / "embeddings.npz", allow_pickle=True)
    assert list(data["task_categories"]) == ["1d_move_1p"]
