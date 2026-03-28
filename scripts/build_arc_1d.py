"""Download the 1D-ARC dataset and convert it to a task-level HuggingFace DatasetDict."""

import argparse
import json
import random
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from datasets import Dataset, DatasetDict

REPO_URL = "https://github.com/khalil-research/1D-ARC.git"
OUTPUT_DIR = Path("data/arc_1d")
SIMPLE_OUTPUT_DIR = Path("data/arc_1d_simple")
MAX_SEQ_LEN = 33
SIMPLE_CATEGORIES = {
    "1d_move_1p",
    "1d_move_2p",
    "1d_move_3p",
    "1d_fill",
    "1d_hollow",
    "1d_denoising_1c",
    "1d_pcopy_1c",
}
SPLIT_RATIOS = {
    "train": 0.8,
    "dev": 0.1,
    "test": 0.1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where the Hugging Face dataset will be saved.",
    )
    parser.add_argument(
        "--holdout-category",
        default=None,
        help="Task category to reserve entirely for the holdout_test split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for deterministic task assignment.",
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Also build a simplified dataset (binary, padded to 33, subset of categories) "
        f"and save to {SIMPLE_OUTPUT_DIR}.",
    )
    return parser.parse_args()


def validate_task_examples(
    task_category: str,
    task_id: int,
    train_examples: list[dict],
    test_examples: list[dict],
) -> int:
    if len(train_examples) != 3:
        msg = (
            f"Expected exactly 3 train examples for {task_category}:{task_id}, "
            f"found {len(train_examples)}"
        )
        raise ValueError(msg)
    if len(test_examples) != 1:
        msg = (
            f"Expected exactly 1 test example for {task_category}:{task_id}, "
            f"found {len(test_examples)}"
        )
        raise ValueError(msg)

    sequence_lengths = set()
    for example in [*train_examples, *test_examples]:
        input_seq = example["input"][0]
        output_seq = example["output"][0]
        sequence_lengths.add(len(input_seq))
        sequence_lengths.add(len(output_seq))

    if len(sequence_lengths) != 1:
        msg = (
            f"Task {task_category}:{task_id} has inconsistent sequence lengths: "
            f"{sorted(sequence_lengths)}"
        )
        raise ValueError(msg)

    return sequence_lengths.pop()


def build_task_record(task_category: str, task_id: int, data: dict) -> dict:
    train_examples = data.get("train", [])
    test_examples = data.get("test", [])
    sequence_length = validate_task_examples(task_category, task_id, train_examples, test_examples)
    test_example = test_examples[0]

    return {
        "task_category": task_category,
        "task_id": task_id,
        "sequence_length": sequence_length,
        "support_inputs": [example["input"][0] for example in train_examples],
        "support_outputs": [example["output"][0] for example in train_examples],
        "query_input": test_example["input"][0],
        "query_output": test_example["output"][0],
    }


def collect_tasks(dataset_dir: Path) -> list[dict]:
    tasks = []

    for task_dir in sorted(dataset_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        task_category = task_dir.name

        for json_file in sorted(task_dir.glob("*.json")):
            task_id = int(json_file.stem.rsplit("_", 1)[-1])
            data = json.loads(json_file.read_text())
            tasks.append(build_task_record(task_category, task_id, data))

    return tasks


def group_tasks_by_category(tasks: list[dict]) -> dict[str, list[dict]]:
    grouped_tasks: dict[str, list[dict]] = defaultdict(list)
    for task in tasks:
        grouped_tasks[task["task_category"]].append(task)
    return grouped_tasks


def allocate_counts(total_count: int, weights: dict[str, float]) -> dict[str, int]:
    raw_counts = {split_name: total_count * weight for split_name, weight in weights.items()}
    counts = {split_name: int(raw_count) for split_name, raw_count in raw_counts.items()}
    remainder = total_count - sum(counts.values())

    ranked_splits = sorted(
        raw_counts,
        key=lambda split_name: (raw_counts[split_name] - counts[split_name], weights[split_name]),
        reverse=True,
    )
    for split_name in ranked_splits[:remainder]:
        counts[split_name] += 1

    return counts


def stratified_task_split(
    tasks: list[dict],
    rng: random.Random,
) -> dict[str, list[dict]]:
    target_counts = allocate_counts(len(tasks), SPLIT_RATIOS)
    assignments = {split_name: [] for split_name in target_counts}
    remaining_counts = target_counts.copy()

    shuffled_tasks = list(tasks)
    rng.shuffle(shuffled_tasks)
    sorted_tasks = sorted(shuffled_tasks, key=lambda task: task["sequence_length"])

    # Build rank-based buckets so each split samples across the full length range.
    # This is simpler than optimizing summary statistics directly and keeps the
    # boxplot quantiles close by construction.
    bucket_count = min(5, len(sorted_tasks))
    bucket_sizes = allocate_counts(
        len(sorted_tasks),
        {str(index): 1.0 / bucket_count for index in range(bucket_count)},
    )

    start_idx = 0
    for bucket_index in range(bucket_count):
        bucket_size = bucket_sizes[str(bucket_index)]
        bucket_tasks = sorted_tasks[start_idx : start_idx + bucket_size]
        start_idx += bucket_size
        rng.shuffle(bucket_tasks)

        bucket_weights = {
            split_name: remaining_counts[split_name] / sum(remaining_counts.values())
            for split_name in remaining_counts
        }
        bucket_counts = allocate_counts(len(bucket_tasks), bucket_weights)

        if len(bucket_tasks) >= 3:
            for split_name in ("dev", "test"):
                if remaining_counts[split_name] > 0 and bucket_counts[split_name] == 0:
                    donor_split = max(
                        (name for name in ("train", "dev", "test") if bucket_counts[name] > 0),
                        key=lambda name: (bucket_counts[name], remaining_counts[name]),
                    )
                    bucket_counts[donor_split] -= 1
                    bucket_counts[split_name] += 1

        bucket_start = 0
        for split_name in ("train", "dev", "test"):
            count = min(bucket_counts[split_name], remaining_counts[split_name])
            selected_tasks = bucket_tasks[bucket_start : bucket_start + count]
            assignments[split_name].extend(selected_tasks)
            remaining_counts[split_name] -= count
            bucket_start += count

        leftover_tasks = bucket_tasks[bucket_start:]
        for task in leftover_tasks:
            split_name = max(remaining_counts, key=remaining_counts.get)
            assignments[split_name].append(task)
            remaining_counts[split_name] -= 1

    if any(remaining_counts.values()):
        msg = f"Task split allocation failed: {remaining_counts}"
        raise ValueError(msg)

    return assignments


def build_dataset_dict(
    tasks: list[dict],
    holdout_category: str | None = None,
    seed: int = 0,
) -> DatasetDict:
    grouped_tasks = group_tasks_by_category(tasks)
    available_categories = sorted(grouped_tasks)
    if holdout_category is not None and holdout_category not in grouped_tasks:
        msg = (
            f"Unknown holdout category {holdout_category!r}. "
            f"Available categories: {', '.join(available_categories)}"
        )
        raise ValueError(msg)

    rng = random.Random(seed)
    split_tasks: dict[str, list[dict]] = {split_name: [] for split_name in SPLIT_RATIOS}
    if holdout_category is not None:
        split_tasks["holdout_test"] = []

    for category in available_categories:
        category_tasks = grouped_tasks[category]
        if category == holdout_category:
            split_tasks["holdout_test"].extend(category_tasks)
            continue

        assignments = stratified_task_split(category_tasks, rng)
        for split_name, split_group_tasks in assignments.items():
            split_tasks[split_name].extend(split_group_tasks)

    return DatasetDict(
        {
            split_name: Dataset.from_list(tasks_for_split)
            for split_name, tasks_for_split in split_tasks.items()
            if tasks_for_split
        }
    )


def binarise_sequence(seq: list[int]) -> list[int]:
    return [1 if v != 0 else 0 for v in seq]


def pad_sequence(seq: list[int], length: int) -> list[int]:
    return seq + [0] * (length - len(seq))


def simplify_task(task: dict) -> dict:
    return {
        "task_category": task["task_category"],
        "task_id": task["task_id"],
        "sequence_length": task["sequence_length"],
        "support_inputs": [
            pad_sequence(binarise_sequence(s), MAX_SEQ_LEN) for s in task["support_inputs"]
        ],
        "support_outputs": [
            pad_sequence(binarise_sequence(s), MAX_SEQ_LEN) for s in task["support_outputs"]
        ],
        "query_input": pad_sequence(binarise_sequence(task["query_input"]), MAX_SEQ_LEN),
        "query_output": pad_sequence(binarise_sequence(task["query_output"]), MAX_SEQ_LEN),
    }


def build_simple_dataset(dataset_dict: DatasetDict) -> DatasetDict:
    splits = {}
    for split_name, dataset in dataset_dict.items():
        tasks = [
            simplify_task(task)
            for task in dataset
            if task["task_category"] in SIMPLE_CATEGORIES
            and task["sequence_length"] <= MAX_SEQ_LEN
        ]
        splits[split_name] = Dataset.from_list(tasks)
    return DatasetDict(splits)


def print_split_summary(dataset_dict: DatasetDict) -> None:
    for split_name, dataset in dataset_dict.items():
        category_counts = Counter(dataset["task_category"])
        print(f"{split_name}: {len(dataset)} tasks")
        for category in sorted(category_counts):
            print(f"  {category}: {category_counts[category]} tasks")


def main() -> None:
    args = parse_args()

    with tempfile.TemporaryDirectory() as tmp_dir:
        repo_dir = Path(tmp_dir) / "1D-ARC"
        print(f"Cloning {REPO_URL} ...")
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, str(repo_dir)],
            check=True,
        )
        tasks = collect_tasks(repo_dir / "dataset")

    num_categories = len({task["task_category"] for task in tasks})
    print(f"Collected {len(tasks)} tasks across {num_categories} task categories")

    dataset_dict = build_dataset_dict(
        tasks,
        holdout_category=args.holdout_category,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(args.output_dir))
    print(f"Saved dataset to {args.output_dir}")
    print(dataset_dict)
    print_split_summary(dataset_dict)

    if args.simple:
        print("\nBuilding simplified dataset (binary, padded to 33) ...")
        simple_dict = build_simple_dataset(dataset_dict)
        SIMPLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        simple_dict.save_to_disk(str(SIMPLE_OUTPUT_DIR))
        print(f"Saved simplified dataset to {SIMPLE_OUTPUT_DIR}")
        print(simple_dict)
        print_split_summary(simple_dict)


if __name__ == "__main__":
    main()
