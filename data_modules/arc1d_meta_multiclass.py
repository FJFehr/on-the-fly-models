"""Meta-learning variable-length multiclass ARC1D datamodule.

Uses data/arc_1d (unpadded, all 18 categories including 1d_padded_fill).
Sequences are padded dynamically to the longest in each batch.
"""

import lightning as pl
import torch
import torch.nn.functional as F
from datasets import DatasetDict
from torch.utils.data import DataLoader, Dataset

from data_modules.arc1d_simple import filter_split

PAD_IDX = 10  # sentinel outside the 0–9 ARC colour range


class Arc1dMetaPaddingCollator:
    """Collate variable-length Arc1dMetaTask items by padding to the longest sequence in the batch.

    Pads ``support_inputs``, ``support_outputs``, ``query_input``, and ``query_output``
    with ``padding_value`` (default PAD_IDX). Non-tensor fields are gathered into lists.
    """

    def __init__(self, padding_value: int = PAD_IDX):
        self.padding_value = padding_value

    def __call__(self, batch: list[dict]) -> dict:
        max_len = max(item["query_input"].shape[-1] for item in batch)
        sup_inputs, sup_outputs, q_inputs, q_outputs = [], [], [], []
        for item in batch:
            pad = max_len - item["query_input"].shape[-1]
            # support_inputs / support_outputs: (num_support, L)
            sup_inputs.append(F.pad(item["support_inputs"], (0, pad), value=self.padding_value))
            sup_outputs.append(F.pad(item["support_outputs"], (0, pad), value=self.padding_value))
            # query_input / query_output: (L,)
            q_inputs.append(F.pad(item["query_input"], (0, pad), value=self.padding_value))
            q_outputs.append(F.pad(item["query_output"], (0, pad), value=self.padding_value))
        return {
            "support_inputs": torch.stack(sup_inputs),
            "support_outputs": torch.stack(sup_outputs),
            "query_input": torch.stack(q_inputs),
            "query_output": torch.stack(q_outputs),
            "task_category": [item["task_category"] for item in batch],
            "task_id": [item["task_id"] for item in batch],
        }


class Arc1dMetaTaskDataset(Dataset):
    """Dataset that keeps whole ARC tasks intact for meta-learning."""

    def __init__(self, tasks: list[dict]):
        self.tasks = tasks

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, index: int) -> dict:
        task = self.tasks[index]
        return {
            "support_inputs": torch.tensor(task["support_inputs"], dtype=torch.long),
            "support_outputs": torch.tensor(task["support_outputs"], dtype=torch.long),
            "query_input": torch.tensor(task["query_input"], dtype=torch.long),
            "query_output": torch.tensor(task["query_output"], dtype=torch.long),
            "task_category": task["task_category"],
            "task_id": task["task_id"],
        }


class Arc1dMetaMulticlassDataModule(pl.LightningDataModule):
    """Meta-learning datamodule for variable-length multiclass ARC1D runs.

    Uses data/arc_1d which includes all 18 task categories (including 1d_padded_fill)
    with variable-length sequences. The collator pads each batch to its longest sequence.
    """

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
        padding_value: int = PAD_IDX,
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
        self.collator = Arc1dMetaPaddingCollator(padding_value=padding_value)

    def resolve_split_name(self, dataset_dict: DatasetDict, split_name: str) -> str:
        if split_name in dataset_dict:
            return split_name
        aliased_split_name = self.SPLIT_ALIASES.get(split_name)
        if aliased_split_name is not None and aliased_split_name in dataset_dict:
            return aliased_split_name
        msg = f"Unknown ARC1D split {split_name!r}."
        raise ValueError(msg)

    def build_dataset(
        self,
        dataset_dict: DatasetDict,
        split_name: str,
        task_categories: list[str] | None,
    ) -> Arc1dMetaTaskDataset:
        resolved_split_name = self.resolve_split_name(dataset_dict, split_name)
        filtered_tasks = filter_split(
            dataset_dict[resolved_split_name],
            task_categories,
            self.task_ids,
        )
        if not filtered_tasks:
            msg = f"No ARC1D tasks matched the configured filters in split {split_name!r}."
            raise ValueError(msg)
        return Arc1dMetaTaskDataset(filtered_tasks)

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
            collate_fn=self.collator,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collator,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collator,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
