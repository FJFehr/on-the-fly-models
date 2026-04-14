"""Visualise task-level 1D-ARC examples from a saved Hugging Face dataset."""

import argparse
from pathlib import Path

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from matplotlib import pyplot as plt

from visualisation import render_task_figure
from visualisation.style import apply_latex_style


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="data/arc_1d",
        help="Directory containing the saved ARC task dataset.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to inspect. Use 'all' to concatenate all available splits.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/task_visualisations"),
        help="Directory where rendered task images will be written.",
    )
    parser.add_argument(
        "--task-category",
        default=None,
        help="Optional task category filter.",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="Optional task id filter. If provided, a single matching task is rendered.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Maximum number of tasks to render after filtering.",
    )
    parser.add_argument(
        "--first-per-category",
        action="store_true",
        help="Render only the first task encountered for each category.",
    )
    return parser.parse_args()


def select_dataset(data: Dataset | DatasetDict, split: str) -> Dataset:
    if isinstance(data, Dataset):
        if split != "all":
            print("Loaded a single dataset; ignoring --split.")
        return data

    if split == "all":
        ordered_splits = [
            split_name
            for split_name in ("train", "dev", "test", "holdout_test")
            if split_name in data
        ]
        return concatenate_datasets([data[split_name] for split_name in ordered_splits])

    if split not in data:
        available_splits = ", ".join(data.keys())
        msg = f"Unknown split {split!r}. Available splits: {available_splits}"
        raise ValueError(msg)

    return data[split]


def load_task_data(data_dir: str | Path) -> Dataset | DatasetDict:
    data_path = Path(data_dir)

    # Prefer the explicit DatasetDict loader because the directory can also contain
    # legacy root-level dataset files that confuse the generic loader.
    if (data_path / "dataset_dict.json").exists():
        return DatasetDict.load_from_disk(str(data_path))

    return load_from_disk(str(data_path))


def is_task_dataset(ds: Dataset) -> bool:
    required_columns = {
        "task_category",
        "task_id",
        "support_inputs",
        "support_outputs",
        "query_input",
        "query_output",
    }
    return required_columns.issubset(ds.column_names)


def filter_tasks(
    ds: Dataset,
    task_category: str | None,
    task_id: int | None,
    first_per_category: bool,
    max_tasks: int | None,
) -> list[dict]:
    selected_tasks = []
    seen_categories = set()

    for task in ds:
        if task_category is not None and task["task_category"] != task_category:
            continue
        if task_id is not None and task["task_id"] != task_id:
            continue
        if first_per_category and task["task_category"] in seen_categories:
            continue

        selected_tasks.append(task)
        seen_categories.add(task["task_category"])

        if max_tasks is not None and len(selected_tasks) >= max_tasks:
            break

    return selected_tasks


def visualise_task(task: dict, save_path: Path) -> None:
    fig = render_task_figure(task)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_output_path(output_dir: Path, task: dict) -> Path:
    category = task["task_category"]
    task_id = task["task_id"]
    return output_dir / f"arc_1d_task_{category}_{task_id}.png"


def main() -> None:
    apply_latex_style()
    args = parse_args()
    data = load_task_data(args.data_dir)
    ds = select_dataset(data, args.split)

    if not is_task_dataset(ds):
        msg = "This script expects the task-level dataset produced by build_arc_1d.py."
        raise ValueError(msg)

    selected_tasks = filter_tasks(
        ds,
        task_category=args.task_category,
        task_id=args.task_id,
        first_per_category=args.first_per_category,
        max_tasks=args.max_tasks,
    )
    if not selected_tasks:
        msg = "No tasks matched the requested filters."
        raise ValueError(msg)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for task in selected_tasks:
        output_path = build_output_path(args.output_dir, task)
        visualise_task(task, output_path)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
