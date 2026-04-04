# data_modules/arc1d_simple.py
# ---------------------------------------------------------------------------
# PyTorch Lightning DataModule for the simplified 1D-ARC dataset.
#
# The simplified dataset contains binarised sequences (0/non-zero -> 0/1),
# padded to length 33, restricted to 7 simple ARC categories.
#
# Data splits:
#   - Training uses support examples (3 input/output pairs per task)
#   - Validation and test use query examples (1 input/output pair per task)
#
# This separation mirrors the meta-learning setup: the model trains on
# support examples and is evaluated on its ability to generalise to the
# held-out query example for each task.
# ---------------------------------------------------------------------------

import lightning as pl
import torch
from datasets import DatasetDict
from torch.utils.data import DataLoader, Dataset


class Arc1dPairDataset(Dataset):
    """Dataset of input/output pairs with task metadata."""

    def __init__(self, samples: list[dict], tasks: list[dict] | None = None):
        self.samples = samples
        self.tasks = tasks or []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        return {
            "inputs": sample["inputs"],
            "targets": sample["targets"],
            "task_category": sample["task_category"],
            "task_id": sample["task_id"],
            "example_index": sample["example_index"],
            "source": sample["source"],
        }


def support_tasks_to_pairs(split) -> Arc1dPairDataset:
    """Convert task-level HF dataset to flat support (input, output) tensor pairs.

    Each task has 3 support examples. This function extracts all support
    input/output pairs across all tasks and stacks them into a flat
    TensorDataset suitable for standard batch training.

    Args:
        split: A HuggingFace Dataset split containing task records.
            Each record has "support_inputs" (list of 3 sequences)
            and "support_outputs" (list of 3 sequences).

    """
    tasks = list(split)
    samples = []

    # Iterate over every task in the split and extract each support pair
    for task in tasks:
        for example_index, (inp, out) in enumerate(
            zip(task["support_inputs"], task["support_outputs"], strict=True)
        ):
            samples.append(
                {
                    "inputs": torch.tensor(inp, dtype=torch.float32),
                    "targets": torch.tensor(out, dtype=torch.float32),
                    "task_category": task["task_category"],
                    "task_id": task["task_id"],
                    "example_index": example_index,
                    "source": "support",
                }
            )

    return Arc1dPairDataset(samples, tasks=tasks)


def query_tasks_to_pairs(split) -> Arc1dPairDataset:
    """Convert task-level HF dataset to flat query (input, output) tensor pairs.

    Each task has exactly 1 query example. This function extracts the
    query input/output pair from every task and stacks them into a flat
    TensorDataset for evaluation.

    Args:
        split: A HuggingFace Dataset split containing task records.
            Each record has "query_input" (one sequence) and
            "query_output" (one sequence).

    """
    tasks = list(split)
    samples = []

    # Iterate over every task and extract the single query pair
    for task in tasks:
        samples.append(
            {
                "inputs": torch.tensor(task["query_input"], dtype=torch.float32),
                "targets": torch.tensor(task["query_output"], dtype=torch.float32),
                "task_category": task["task_category"],
                "task_id": task["task_id"],
                "example_index": 0,
                "source": "query",
            }
        )

    return Arc1dPairDataset(samples, tasks=tasks)


def filter_split(split, task_categories: list[str] | None, task_ids: list[int] | None):
    allowed_categories = set(task_categories) if task_categories is not None else None
    allowed_task_ids = set(task_ids) if task_ids is not None else None

    if allowed_categories is None and allowed_task_ids is None:
        return split

    filtered_tasks = []
    for task in split:
        if allowed_categories is not None and task["task_category"] not in allowed_categories:
            continue
        if allowed_task_ids is not None and task["task_id"] not in allowed_task_ids:
            continue
        filtered_tasks.append(task)
    return filtered_tasks


def dataset_from_source(split, source: str) -> Arc1dPairDataset:
    if source == "train_support_pairs":
        return support_tasks_to_pairs(split)
    if source == "train_query_pairs":
        return query_tasks_to_pairs(split)
    msg = f"Unknown dataset source {source!r}"
    raise ValueError(msg)


class Arc1dSimpleDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for the simplified 1D-ARC dataset.

    Loads a pre-built HuggingFace DatasetDict from disk and converts
    it into PyTorch TensorDatasets for training, validation, and testing.

    Training data comes from support examples (3 per task).
    Validation and test data come from query examples (1 per task).

    The constructor accepts explicit keyword arguments. Any extra kwargs
    from the config (e.g. model params, project name) are silently
    ignored via **kwargs, allowing the full config dict to be passed.

    Args:
        data_dir: Path to the saved HuggingFace DatasetDict on disk.
        batch_size: Number of (input, output) pairs per training batch.
        num_workers: Number of parallel data loading workers (0 = main process).
        **kwargs: Extra config keys (ignored, allows passing full config).
    """

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
        """Load the dataset from disk and create train/val/test splits.

        Called by Lightning before training begins. Loads the HuggingFace
        DatasetDict and converts each split into TensorDatasets:
          - Training: support pairs from the train split
          - Validation: query pairs from the train split
          - Test: query pairs from the train split (same as val for now)

        Args:
            stage: Lightning stage ("fit", "validate", "test", or "predict").
                Not used here — we load all splits regardless.
        """
        # Load the pre-built HuggingFace DatasetDict from disk
        dataset_dict = DatasetDict.load_from_disk(self.data_dir)

        # Use only the train split for now
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
        """Create the training DataLoader with shuffling enabled."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """Create the validation DataLoader (no shuffling)."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        """Create the test DataLoader (no shuffling)."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
