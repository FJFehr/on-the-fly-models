"""PyTorch Lightning DataModule for padded multiclass 1D-ARC.

This dataset preserves the original ARC token values in 0..9, pads every
sequence to length 33, and excludes the ``1d_padded_fill`` category.

Like the binary baseline, training uses support pairs and validation/test use
held-out query pairs so Milestone 1 stays pair-based rather than task-
conditioned.
"""

import lightning as pl
import torch
from datasets import DatasetDict
from torch.utils.data import DataLoader

from data_modules.arc1d_simple import Arc1dPairDataset, filter_split


def support_tasks_to_pairs(split) -> Arc1dPairDataset:
    """Convert task-level HF dataset to flat support pairs for multiclass training."""
    tasks = list(split)
    samples = []

    for task in tasks:
        for example_index, (inp, out) in enumerate(
            zip(task["support_inputs"], task["support_outputs"], strict=True)
        ):
            samples.append(
                {
                    "inputs": torch.tensor(inp, dtype=torch.float32),
                    "targets": torch.tensor(out, dtype=torch.long),
                    "task_category": task["task_category"],
                    "task_id": task["task_id"],
                    "example_index": example_index,
                    "source": "support",
                }
            )

    return Arc1dPairDataset(samples, tasks=tasks)


def query_tasks_to_pairs(split) -> Arc1dPairDataset:
    """Convert task-level HF dataset to flat query pairs for multiclass evaluation."""
    tasks = list(split)
    samples = []

    for task in tasks:
        samples.append(
            {
                "inputs": torch.tensor(task["query_input"], dtype=torch.float32),
                "targets": torch.tensor(task["query_output"], dtype=torch.long),
                "task_category": task["task_category"],
                "task_id": task["task_id"],
                "example_index": 0,
                "source": "query",
            }
        )

    return Arc1dPairDataset(samples, tasks=tasks)


def dataset_from_source(split, source: str) -> Arc1dPairDataset:
    if source == "train_support_pairs":
        return support_tasks_to_pairs(split)
    if source == "train_query_pairs":
        return query_tasks_to_pairs(split)
    msg = f"Unknown dataset source {source!r}"
    raise ValueError(msg)


class Arc1dPaddedMulticlassDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for padded multiclass 1D-ARC."""

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int = 0,
        task_categories: list[str] | None = None,
        task_ids: list[int] | None = None,
        train_source: str = "train_support_pairs",
        val_source: str = "train_query_pairs",
        test_source: str = "train_query_pairs",
        overfit_single_batch: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.task_categories = task_categories
        self.task_ids = task_ids
        self.train_source = train_source
        self.val_source = val_source
        self.test_source = test_source
        self.overfit_single_batch = overfit_single_batch

    def setup(self, stage=None):
        dataset_dict = DatasetDict.load_from_disk(self.data_dir)

        train_split = filter_split(dataset_dict["train"], self.task_categories, self.task_ids)
        if not train_split:
            msg = "No ARC1D tasks matched the configured category/task filters."
            raise ValueError(msg)

        self.train_dataset = dataset_from_source(train_split, self.train_source)

        if self.val_source == "same_train_pairs":
            self.val_dataset = self.train_dataset
        else:
            self.val_dataset = dataset_from_source(train_split, self.val_source)

        if self.test_source == "same_train_pairs":
            self.test_dataset = self.train_dataset
        else:
            self.test_dataset = dataset_from_source(train_split, self.test_source)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
