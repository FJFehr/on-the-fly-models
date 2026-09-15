"""Meta-learning variable-length multiclass ARC1D datamodule.

Uses data/arc_1d (unpadded, all 18 categories including 1d_padded_fill).
Sequences are padded dynamically to the longest in each batch.
"""

import random

import lightning as pl
import torch
import torch.nn.functional as F
from datasets import DatasetDict
from torch.utils.data import DataLoader, Dataset

from data_modules.task_filtering import filter_split

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
        variants_per_base_task: int | None = None,
        base_tasks_per_category: int | None = None,
        data_seed: int = 42,
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
        self.variants_per_base_task = variants_per_base_task
        self.base_tasks_per_category = base_tasks_per_category
        self.data_seed = data_seed
        self.collator = Arc1dMetaPaddingCollator(padding_value=padding_value)

    def resolve_split_name(self, dataset_dict: DatasetDict, split_name: str) -> str:
        if split_name in dataset_dict:
            return split_name
        aliased_split_name = self.SPLIT_ALIASES.get(split_name)
        if aliased_split_name is not None and aliased_split_name in dataset_dict:
            return aliased_split_name
        msg = f"Unknown ARC1D split {split_name!r}."
        raise ValueError(msg)

    def _stratified_base_tasks_per_category(
        self, tasks: list[dict], base_tasks_per_category: int
    ) -> list[dict]:
        """Keep rows from only up to base_tasks_per_category distinct base tasks per category.

        One level up from _stratified_variants_per_base_task: that method
        thins augmentation depth *within* every base task; this one instead
        drops whole base tasks (all of their rows/variants together), to
        reach data levels below the "every base task's original example"
        floor (variants_per_base_task=1, ~40 base tasks/category).

        Groups by task_category, collects the distinct base-task ids present
        (task_id // 10000) per category in ascending sorted order (the same
        natural, stable base-task ordering _stratified_variants_per_base_task
        already relies on), shuffles that ordering with a local
        data_seed-seeded RNG, and keeps every row belonging to the first
        base_tasks_per_category ids in the shuffled order for that category.
        Nested/cumulative across levels for a fixed data_seed (shuffle once,
        take-prefix), same guarantee as the variants axis, one level up.

        Uses its own fresh random.Random(self.data_seed) instance -- a
        separate object from _stratified_variants_per_base_task's own RNG --
        so the two axes' randomness never shares state and each is
        independent of the other's value when both are set (verified by
        tests/test_arc1d_meta_multiclass.py's own composability test).
        Deterministic given self.data_seed, independent of the training seed,
        same as the variants axis.
        """
        by_category: dict[str, set[int]] = {}
        for task in tasks:
            by_category.setdefault(task["task_category"], set()).add(
                task["task_id"] // 10000
            )

        rng = random.Random(self.data_seed)
        keep: dict[str, set[int]] = {}
        for category in sorted(by_category):
            base_task_ids = sorted(by_category[category])
            rng.shuffle(base_task_ids)
            n = min(base_tasks_per_category, len(base_task_ids))
            keep[category] = set(base_task_ids[:n])

        return [t for t in tasks if (t["task_id"] // 10000) in keep[t["task_category"]]]

    def _stratified_variants_per_base_task(
        self, tasks: list[dict], variants_per_base_task: int
    ) -> list[dict]:
        """Keep up to variants_per_base_task rows per (category, base task) group.

        Augmented rows carry task_id = original_task_id * 10000 + aug_index
        (see scripts/augment_arc_1d.py's augment_task), so aug_index == 0 is
        always the untransformed original example for that base task.
        Each group's selection always includes that original first, then a
        fixed data_seed-ordered sequence of the remaining augmented variants -
        so level K's selection is always level K-1's plus exactly one more
        per base task (nested/cumulative across levels), and level 1 is
        exactly the original, unaugmented example for every base task.

        Deterministic given self.data_seed, independent of the training seed,
        so multiple training seeds at a fixed level see identical training data.
        """
        by_base_task: dict[tuple[str, int], list[dict]] = {}
        for task in tasks:
            key = (task["task_category"], task["task_id"] // 10000)
            by_base_task.setdefault(key, []).append(task)

        rng = random.Random(self.data_seed)
        selected: list[dict] = []
        for key in sorted(by_base_task):
            variants = by_base_task[key]
            original = [t for t in variants if t["task_id"] % 10000 == 0]
            rest = [t for t in variants if t["task_id"] % 10000 != 0]
            rng.shuffle(rest)
            ordered = original + rest
            n = min(variants_per_base_task, len(ordered))
            selected.extend(ordered[:n])
        return selected

    def build_dataset(
        self,
        dataset_dict: DatasetDict,
        split_name: str,
        task_categories: list[str] | None,
        variants_per_base_task: int | None = None,
        base_tasks_per_category: int | None = None,
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
        if base_tasks_per_category is not None:
            filtered_tasks = self._stratified_base_tasks_per_category(
                filtered_tasks, base_tasks_per_category
            )
        if variants_per_base_task is not None:
            filtered_tasks = self._stratified_variants_per_base_task(
                filtered_tasks, variants_per_base_task
            )
        return Arc1dMetaTaskDataset(filtered_tasks)

    def setup(self, stage=None):
        dataset_dict = DatasetDict.load_from_disk(self.data_dir)
        val_cats = self.val_task_categories or self.task_categories
        self.train_dataset = self.build_dataset(
            dataset_dict,
            self.train_split,
            self.task_categories,
            variants_per_base_task=self.variants_per_base_task,
            base_tasks_per_category=self.base_tasks_per_category,
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
