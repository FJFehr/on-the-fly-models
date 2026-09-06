"""log_embedding_cluster_plots should dump the raw embeddings alongside its figure."""

import numpy as np
import torch

from models.hypermodel_lightning import HyperModelLightning
from training.logging import log_embedding_cluster_plots


def build_model() -> HyperModelLightning:
    return HyperModelLightning(
        hyper_model={
            "name": "transformer",
            "params": {"hidden_dim": 16, "num_layers": 1, "num_heads": 1, "output_dim": 8},
        },
        target_model={
            "name": "rnn",
            "params": {"hidden_dim": 8, "num_layers": 1, "bidirectional": True},
        },
        input_dim=4,
        task_encoding={"embedding_dim": 8},
        log_embedding_clusters=True,
    )


def make_batch(task_category: str, task_id: int) -> dict:
    return {
        "support_inputs": torch.tensor(
            [[[0, 1, 0, 1], [1, 0, 1, 0], [0, 0, 1, 1]]], dtype=torch.float32
        ),
        "support_outputs": torch.tensor(
            [[[1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 0, 1]]], dtype=torch.float32
        ),
        "query_input": torch.tensor([[1, 0, 0, 1]], dtype=torch.float32),
        "query_output": torch.tensor([[0, 1, 1, 0]], dtype=torch.float32),
        "task_category": [task_category],
        "task_id": torch.tensor([task_id], dtype=torch.long),
    }


class FakeDatamodule:
    def __init__(self, batches: list[dict]) -> None:
        self._batches = batches

    def val_dataloader(self):
        return self._batches


def test_log_embedding_cluster_plots_dumps_an_npz_alongside_the_figure(tmp_path):
    model = build_model()
    # UMAP's default n_neighbors needs more points than a couple of tiny batches --
    # enough samples here for render_single_projection_figures to run to completion.
    categories = ["1d_move_1p", "1d_flip"] * 10
    batches = [make_batch(category, task_id) for task_id, category in enumerate(categories)]
    datamodule = FakeDatamodule(batches)

    log_embedding_cluster_plots(model, datamodule, output_path=str(tmp_path))

    npz_path = tmp_path / "embedding_clusters" / "embeddings.npz"
    assert npz_path.exists()

    data = np.load(npz_path, allow_pickle=True)
    assert set(data.files) == {"vectors", "task_categories", "task_ids"}
    assert data["vectors"].shape == (len(batches), model.hypermodel.hyper_output_dim)
    assert list(data["task_categories"]) == categories
    assert list(data["task_ids"]) == list(range(len(batches)))


def test_log_embedding_cluster_plots_npz_excludes_holdout_records(tmp_path):
    """The dumped .npz is the plain reference set, even when a holdout overlay is requested --
    holdout points get mixed into the *figure* but shouldn't pollute the reusable dump other
    scripts (visualisation.paper.plot_embedding_clusters et al.) read as "this run's embeddings"."""
    model = build_model()
    reference_batches = [make_batch("1d_move_1p", 1)]
    holdout_batches = [make_batch("1d_comp_fill_mirror", 99)]
    datamodule = FakeDatamodule(reference_batches)

    log_embedding_cluster_plots(
        model,
        datamodule,
        output_path=str(tmp_path),
        holdout_dataloader=holdout_batches,
    )

    data = np.load(tmp_path / "embedding_clusters" / "embeddings.npz", allow_pickle=True)
    assert list(data["task_categories"]) == ["1d_move_1p"]
