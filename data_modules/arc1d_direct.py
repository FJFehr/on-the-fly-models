"""Flat supervised datamodule for ARC-1D direct training experiments."""

import random

import lightning as pl
import torch
import torch.nn.functional as F
from datasets import DatasetDict
from torch.utils.data import DataLoader, Dataset

from data_modules.task_filtering import filter_split

PAD_IDX = 10  # sentinel outside the 0–9 ARC colour range for multiclass padding


class Arc1dDirectPaddingCollator:
    """Collate variable-length Arc1dDirect items by padding to the longest sequence in the batch.

    Pads ``input`` and ``output`` tensors with ``padding_value`` (default PAD_IDX).
    All other fields (task_category, task_id) are gathered into plain lists.
    """

    def __init__(self, padding_value: int = PAD_IDX):
        self.padding_value = padding_value

    def __call__(self, batch: list[dict]) -> dict:
        max_len = max(item["input"].shape[0] for item in batch)
        inputs, outputs = [], []
        for item in batch:
            pad = max_len - item["input"].shape[0]
            inputs.append(F.pad(item["input"], (0, pad), value=self.padding_value))
            outputs.append(F.pad(item["output"], (0, pad), value=self.padding_value))
        return {
            "input": torch.stack(inputs),
            "output": torch.stack(outputs),
            "task_category": [item["task_category"] for item in batch],
            "task_id": [item["task_id"] for item in batch],
        }


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
        padding_value: int = PAD_IDX,
        variants_per_base_task: int | None = None,
        base_tasks_per_category: int | None = None,
        data_seed: int = 42,
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
        self.variants_per_base_task = variants_per_base_task
        self.base_tasks_per_category = base_tasks_per_category
        self.data_seed = data_seed
        self.collator = Arc1dDirectPaddingCollator(padding_value=padding_value)
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
        tests/test_arc1d_direct.py's own composability test). Deterministic
        given self.data_seed, independent of the training seed, same as the
        variants axis.
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

    def build_flat_dataset(
        self,
        dataset_dict: DatasetDict,
        split_name: str,
        task_categories: list[str] | None,
        *,
        use_support: bool,
        variants_per_base_task: int | None = None,
        base_tasks_per_category: int | None = None,
    ) -> Arc1dDirectDataset:
        """Build a flat list of (input, output) pairs from a split.

        Args:
            dataset_dict: HuggingFace DatasetDict loaded from disk.
            split_name: Name of the split to use (e.g. "train", "dev").
            task_categories: Optional filter; None means all categories.
            use_support: If True, yield all 3 support pairs per task.
                         If False, yield only the query pair per task.
            variants_per_base_task: If set, subsample rows per base task
                (stratified/nested, see _stratified_variants_per_base_task)
                before unpacking into flat (input, output) pairs.
            base_tasks_per_category: If set, subsample whole base tasks per
                category (stratified/nested, see
                _stratified_base_tasks_per_category) before the
                variants_per_base_task step -- an independent, one-level-up
                axis (which base tasks are present at all, not how many
                variants of each).
        """
        resolved = self.resolve_split_name(dataset_dict, split_name)
        tasks = filter_split(dataset_dict[resolved], task_categories, self.task_ids)
        if not tasks:
            msg = f"No ARC1D tasks matched the configured filters in split {split_name!r}."
            raise ValueError(msg)
        if base_tasks_per_category is not None:
            tasks = self._stratified_base_tasks_per_category(tasks, base_tasks_per_category)
        if variants_per_base_task is not None:
            tasks = self._stratified_variants_per_base_task(tasks, variants_per_base_task)

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
            dataset_dict,
            self.train_split,
            self.task_categories,
            use_support=True,
            variants_per_base_task=self.variants_per_base_task,
            base_tasks_per_category=self.base_tasks_per_category,
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
            # persistent_workers avoids respawning the worker every epoch --
            # with a fixed max_steps (not max_epochs) stopping condition,
            # a tiny train_dataset means many epoch boundaries per run, and
            # without this each one paid a full worker-fork cost. Pure
            # wall-clock fix, no effect on results (shuffling happens in the
            # main-process sampler above, not inside workers). Requires
            # num_workers > 0 (PyTorch raises otherwise).
            persistent_workers=self.num_workers > 0,
            collate_fn=self.collator,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=self.collator,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=self.collator,
        )
