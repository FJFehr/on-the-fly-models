"""Flat supervised datamodule for ARC-1D direct training experiments."""

import lightning as pl
import torch
from datasets import DatasetDict
from torch.utils.data import DataLoader, Dataset

from data_modules.arc1d_simple import filter_split


class Arc1dDirectDataset(Dataset):
    """Flat dataset of (input, output) pairs unpacked from ARC-1D tasks.

    Each item is a single example — either one of the three support pairs
    (when building the training set) or the query pair (for validation/test).
    """

    def __init__(self, items: list[dict], dtype: torch.dtype):
        self.items = items
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        item = self.items[index]
        return {
            "input": torch.tensor(item["input"], dtype=self.dtype),
            "output": torch.tensor(item["output"], dtype=self.dtype),
            "task_category": item["task_category"],
            "task_id": item["task_id"],
        }


class Arc1dDirectDataModule(pl.LightningDataModule):
    """Flat supervised datamodule for direct target-model training on ARC-1D.

    Training set: all 3 support examples per task, pooled across the split.
    Validation/test set: the single query example per task.

    Data sources:
        binary     → data/arc_1d_simple/
        multiclass → data/arc_1d_padded_multiclass/
    """

    SPLIT_ALIASES = {"val": "dev"}

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        prediction_task: str = "binary",
        num_workers: int = 0,
        task_categories: list[str] | None = None,
        val_task_categories: list[str] | None = None,
        task_ids: list[int] | None = None,
        train_split: str = "train",
        val_split: str = "dev",
        test_split: str = "test",
        overfit_single_batch: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.prediction_task = prediction_task
        self.num_workers = num_workers
        self.task_categories = task_categories
        self.val_task_categories = val_task_categories
        self.task_ids = task_ids
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.overfit_single_batch = overfit_single_batch
        # binary tasks store values as float32; multiclass uses class indices (long)
        self.dtype = torch.float32 if prediction_task == "binary" else torch.long

    def resolve_split_name(self, dataset_dict: DatasetDict, split_name: str) -> str:
        if split_name in dataset_dict:
            return split_name
        aliased = self.SPLIT_ALIASES.get(split_name)
        if aliased is not None and aliased in dataset_dict:
            return aliased
        msg = f"Unknown ARC1D split {split_name!r}."
        raise ValueError(msg)

    def build_flat_dataset(
        self,
        dataset_dict: DatasetDict,
        split_name: str,
        task_categories: list[str] | None,
        *,
        use_support: bool,
    ) -> Arc1dDirectDataset:
        """Build a flat list of (input, output) pairs from a split.

        Args:
            dataset_dict: HuggingFace DatasetDict loaded from disk.
            split_name: Name of the split to use (e.g. "train", "dev").
            task_categories: Optional filter; None means all categories.
            use_support: If True, yield all 3 support pairs per task.
                         If False, yield only the query pair per task.
        """
        resolved = self.resolve_split_name(dataset_dict, split_name)
        tasks = filter_split(dataset_dict[resolved], task_categories, self.task_ids)
        if not tasks:
            msg = f"No ARC1D tasks matched the configured filters in split {split_name!r}."
            raise ValueError(msg)

        items = []
        for task in tasks:
            if use_support:
                support_inputs = task["support_inputs"]  # list of 3 sequences
                support_outputs = task["support_outputs"]
                for inp, out in zip(support_inputs, support_outputs, strict=True):
                    items.append(
                        {
                            "input": inp,
                            "output": out,
                            "task_category": task["task_category"],
                            "task_id": task["task_id"],
                        }
                    )
            else:
                items.append(
                    {
                        "input": task["query_input"],
                        "output": task["query_output"],
                        "task_category": task["task_category"],
                        "task_id": task["task_id"],
                    }
                )

        return Arc1dDirectDataset(items, self.dtype)

    def setup(self, stage=None):
        dataset_dict = DatasetDict.load_from_disk(self.data_dir)
        val_cats = self.val_task_categories or self.task_categories
        self.train_dataset = self.build_flat_dataset(
            dataset_dict, self.train_split, self.task_categories, use_support=True
        )
        self.val_dataset = self.build_flat_dataset(
            dataset_dict, self.val_split, val_cats, use_support=False
        )
        self.test_dataset = self.build_flat_dataset(
            dataset_dict, self.test_split, val_cats, use_support=False
        )
        if self.overfit_single_batch:
            self.train_dataset = Arc1dDirectDataset(
                self.train_dataset.items[: self.batch_size], self.dtype
            )
            self.val_dataset = Arc1dDirectDataset(
                self.val_dataset.items[: self.batch_size], self.dtype
            )

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
