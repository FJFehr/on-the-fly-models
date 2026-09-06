"""Meta-learning datamodule for the binary arc_1d_simple track."""

import lightning as pl
import torch
from datasets import DatasetDict
from torch.utils.data import DataLoader, Dataset

from data_modules.task_filtering import filter_split


class Arc1dMetaSimpleTaskDataset(Dataset):
    """Dataset that preserves whole binary ARC tasks for meta-learning."""

    def __init__(self, tasks: list[dict]):
        self.tasks = tasks

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, index: int) -> dict:
        task = self.tasks[index]
        return {
            "support_inputs": torch.tensor(task["support_inputs"], dtype=torch.float32),
            "support_outputs": torch.tensor(task["support_outputs"], dtype=torch.float32),
            "query_input": torch.tensor(task["query_input"], dtype=torch.float32),
            "query_output": torch.tensor(task["query_output"], dtype=torch.float32),
            "task_category": task["task_category"],
            "task_id": task["task_id"],
        }


class Arc1dMetaSimpleDataModule(pl.LightningDataModule):
    """Task-level datamodule for binary meta-learning experiments."""

    SPLIT_ALIASES = {"val": "dev"}

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int = 0,
        task_categories: list[str] | None = None,
        val_task_categories: list[str] | None = None,
        task_ids: list[int] | None = None,
        train_split: str = "train",
        val_split: str = "dev",
        test_split: str = "test",
        **kwargs,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.task_categories = task_categories
        self.val_task_categories = val_task_categories
        self.task_ids = task_ids
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split

    def resolve_split_name(self, dataset_dict: DatasetDict, split_name: str) -> str:
        if split_name in dataset_dict:
            return split_name
        aliased_split_name = self.SPLIT_ALIASES.get(split_name)
        if aliased_split_name is not None and aliased_split_name in dataset_dict:
            return aliased_split_name
        msg = f"Unknown ARC1D split {split_name!r}."
        raise ValueError(msg)

    def build_dataset(
        self, dataset_dict: DatasetDict, split_name: str, task_categories: list[str] | None
    ) -> Arc1dMetaSimpleTaskDataset:
        resolved_split_name = self.resolve_split_name(dataset_dict, split_name)
        filtered_tasks = filter_split(
            dataset_dict[resolved_split_name],
            task_categories,
            self.task_ids,
        )
        if not filtered_tasks:
            msg = f"No ARC1D tasks matched the configured filters in split {split_name!r}."
            raise ValueError(msg)
        return Arc1dMetaSimpleTaskDataset(filtered_tasks)

    def setup(self, stage=None):
        dataset_dict = DatasetDict.load_from_disk(self.data_dir)
        val_cats = self.val_task_categories or self.task_categories
        self.train_dataset = self.build_dataset(
            dataset_dict,
            self.train_split,
            self.task_categories,
        )
        self.val_dataset = self.build_dataset(dataset_dict, self.val_split, val_cats)
        self.test_dataset = self.build_dataset(dataset_dict, self.test_split, val_cats)

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
