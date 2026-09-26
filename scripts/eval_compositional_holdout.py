"""Zero-shot evaluation of a trained hypermodel checkpoint on the held-out compositional set.

Loads a trained notd or frozen_td checkpoint (see
legacy/configs/experiments/arc1d_hypermodel_compositional_generalization/) and runs it, unmodified and
with no fine-tuning, on data/arc_1d_compositional_holdout -- 10 task categories built by chaining
two base rules the model *was* trained on individually (see data_modules/arc1d_compositional.py
and this experiment's README.md for what those categories are and why).

`td` is intentionally not supported here: its num_tasks=18 has no reserved indices for the
composite categories, so a learned one-hot embedding for them is undefined (see README.md).
Composite categories are registered into models.embedding.TASK_CATEGORY_INDEX at
indices 18..18+K-1 for the duration of this script only -- frozen_td's untrained projection
matrix already has those extra rows (num_tasks=28 in frozen_td.yaml), they just need names.

Usage:
    uv run python scripts/eval_compositional_holdout.py \
        --config legacy/configs/experiments/arc1d_hypermodel_compositional_generalization/frozen_td.yaml
    uv run python scripts/eval_compositional_holdout.py \
        --config legacy/configs/experiments/arc1d_hypermodel_compositional_generalization/notd.yaml \
        --checkpoint last --num-qualitative 5

    # Point at a specific seed's checkpoint/output dir via OmegaConf dotlist overrides
    # (same mechanism as train.py), e.g. for a multi-seed rerun:
    uv run python scripts/eval_compositional_holdout.py \
        --config legacy/configs/experiments/arc1d_v2_compositional_generalization/notd.yaml \
        seed=2 experiment_name=v2_compositional_generalization_notd_seed2 \
        output_path=outputs/arc1d_v2_compositional_generalization/v2_compositional_generalization_notd_seed2
"""

import argparse
from collections import defaultdict
from pathlib import Path

import torch
from datasets import DatasetDict
from torch.utils.data import DataLoader

from data_modules import DATA_REGISTRY
from data_modules.arc1d_compositional import COMPOSITE_TASK_SPECS
from data_modules.arc1d_meta_multiclass import Arc1dMetaPaddingCollator, Arc1dMetaTaskDataset
from lightning_modules import MODEL_REGISTRY
from models.embedding import TASK_CATEGORY_INDEX
from training.config import build_runtime_config_dict, load_config
from training.logging import log_embedding_cluster_plots
from training.trainer import load_checkpoint_state, resolve_evaluation_checkpoint_path

BASE_NUM_TASKS = 18  # size of the real ARC-1D TASK_CATEGORY_INDEX registry


def register_composite_task_indices(start_index: int = BASE_NUM_TASKS) -> dict[str, int]:
    """Extend TASK_CATEGORY_INDEX in place with the composite categories, at indices
    start_index..start_index+K-1. Purely additive -- never touches an existing entry."""
    new_entries = {category: start_index + i for i, category in enumerate(COMPOSITE_TASK_SPECS)}
    overlap = set(new_entries) & set(TASK_CATEGORY_INDEX)
    if overlap:
        msg = f"Composite category names collide with existing TASK_CATEGORY_INDEX entries: {overlap}"
        raise ValueError(msg)
    TASK_CATEGORY_INDEX.update(new_entries)
    return new_entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the trained notd.yaml or frozen_td.yaml (not td.yaml -- see module docstring).",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional OmegaConf dotlist overrides such as seed=2 experiment_name=foo_seed2 "
        "output_path=outputs/proj/foo_seed2 -- same mechanism as train.py, needed to point this "
        "script at a specific seed's checkpoint/output dir without a separate physical config "
        "file per seed.",
    )
    parser.add_argument(
        "--checkpoint",
        default="best",
        help="'best', 'last', 'auto', or an explicit checkpoint path (default: best).",
    )
    parser.add_argument(
        "--holdout-data-dir",
        default="data/arc_1d_compositional_holdout",
        help="Directory containing the held-out compositional DatasetDict.",
    )
    parser.add_argument("--holdout-split", default="holdout_test")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write results (default: <run's output_path>/compositional_holdout_eval).",
    )
    parser.add_argument("--num-qualitative", type=int, default=3, help="Rendered examples per category.")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def build_holdout_dataloader(data_dir: str, split: str, batch_size: int, padding_value: int) -> DataLoader:
    dataset_dict = DatasetDict.load_from_disk(data_dir)
    tasks = list(dataset_dict[split])
    return DataLoader(
        Arc1dMetaTaskDataset(tasks),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=Arc1dMetaPaddingCollator(padding_value=padding_value),
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    runtime_cfg = build_runtime_config_dict(cfg)

    num_tasks = cfg.hyper_head.get("num_tasks")
    required_num_tasks = BASE_NUM_TASKS + len(COMPOSITE_TASK_SPECS)
    if num_tasks is not None and num_tasks < required_num_tasks:
        msg = (
            f"{args.config} has hyper_head.num_tasks={num_tasks}, too small to hold the "
            f"{len(COMPOSITE_TASK_SPECS)} composite categories at indices "
            f"{BASE_NUM_TASKS}..{required_num_tasks - 1}. This is expected for td.yaml (num_tasks="
            f"{BASE_NUM_TASKS}) -- td is not supported for held-out eval, use notd or frozen_td "
            "(see this script's module docstring for why)."
        )
        raise ValueError(msg)
    if num_tasks is not None:
        register_composite_task_indices()

    model = MODEL_REGISTRY[cfg.model](**runtime_cfg)
    checkpoint_path = resolve_evaluation_checkpoint_path(cfg.output_path, args.checkpoint)
    load_checkpoint_state(model, checkpoint_path)
    model.to(args.device)
    model.eval()

    datamodule = DATA_REGISTRY[cfg.data](**runtime_cfg)
    datamodule.setup()

    holdout_dataloader = build_holdout_dataloader(
        args.holdout_data_dir,
        args.holdout_split,
        batch_size=cfg.batch_size,
        padding_value=cfg.padding_idx,
    )

    output_dir = Path(args.output_dir or (Path(cfg.output_path) / "compositional_holdout_eval"))
    output_dir.mkdir(parents=True, exist_ok=True)
    qualitative_dir = output_dir / "qualitative"
    qualitative_dir.mkdir(exist_ok=True)

    exact_match_by_category: dict[str, list[bool]] = defaultdict(list)
    accuracy_by_category: dict[str, list[float]] = defaultdict(list)
    rendered_per_category: dict[str, int] = defaultdict(int)

    with torch.no_grad():
        for batch in holdout_dataloader:
            tensor_batch = {
                key: value.to(args.device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            _, predictions, targets = model.predict_batch(tensor_batch)
            records = model.build_task_records(tensor_batch, predictions, targets)
            for record in records:
                category = record["task_category"]
                exact_match_by_category[category].append(record["query_exact_match"])
                accuracy_by_category[category].append(record["query_accuracy"])
                if rendered_per_category[category] < args.num_qualitative:
                    figure = model.build_task_visualization_figure(record)
                    figure.savefig(
                        qualitative_dir / f"{category}_{record['task_id']}.png",
                        dpi=150,
                        bbox_inches="tight",
                    )
                    rendered_per_category[category] += 1

    results_lines = [
        f"Compositional held-out zero-shot eval — config={args.config}, checkpoint={checkpoint_path}",
        "",
        f"{'category':<32}{'n':>6}{'exact_match':>14}{'seq_accuracy':>14}",
    ]
    all_matches: list[bool] = []
    for category in sorted(exact_match_by_category):
        matches = exact_match_by_category[category]
        accuracies = accuracy_by_category[category]
        all_matches.extend(matches)
        mean_match = sum(matches) / len(matches)
        mean_accuracy = sum(accuracies) / len(accuracies)
        results_lines.append(f"{category:<32}{len(matches):>6}{mean_match:>14.3f}{mean_accuracy:>14.3f}")
    overall = sum(all_matches) / len(all_matches) if all_matches else float("nan")
    results_lines.append("")
    results_lines.append(f"{'overall':<32}{len(all_matches):>6}{overall:>14.3f}")

    results_text = "\n".join(results_lines)
    print(results_text)
    (output_dir / "results.txt").write_text(results_text + "\n")

    log_embedding_cluster_plots(
        model=model,
        datamodule=datamodule,
        output_path=str(output_dir),
        holdout_dataloader=holdout_dataloader,
        holdout_label="compositional_holdout",
    )


if __name__ == "__main__":
    main()
