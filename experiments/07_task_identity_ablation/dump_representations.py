"""Experiment 7: dump each in-distribution run's task representations and score how well
they cluster by task category.

For every validation task, four spaces, computed with the model's own submodules:

  support        pooler(encoder(context)), before any latent task identity: for input
                 placement the ID has passed through the encoder, for latent placement it
                 has not been added yet
  support_no_id  pooler(encoder(context)) with no task identity at all: what the encoder
                 makes of the support examples alone (H2: has the ID taught it to align them?)
  task           the representation the weights are generated from, as experiment 2 plots
                 it. With latent placement this contains the one-hot projection, so it
                 clusters by construction; it is not a fair basis for comparing placements.
  weights        the generated target weights (the same space for every arm)

Per space: the shared 5-fold linear probe (visualisation.core.embedding_clusters) and the
silhouette score by category. Also saves each run's task-embedding table for the similarity
heatmaps. Writes outputs/07_task_identity_ablation/representations/<arm>_seed<N>.npz and
cluster_metrics.csv.

Usage:
    PYTHONPATH=. uv run python experiments/07_task_identity_ablation/dump_representations.py
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from common import ARMS, EXPERIMENT, INDIST_RUNS, SEEDS, load_run, parameter_counts
from sklearn.metrics import silhouette_score

from data_modules import DATA_REGISTRY
from visualisation.core.embedding_clusters import compute_linear_probe_accuracy

SPACES = ("support", "support_no_id", "task", "weights")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument(
        "--output-dir", type=Path, default=Path(f"outputs/{EXPERIMENT}/representations")
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def representations(model, dataloader, device: str) -> dict[str, np.ndarray]:
    hypernetwork = model.hypernetwork
    placement = hypernetwork.task_indicator_placement
    spaces = {name: [] for name in (*SPACES, "task_categories", "task_ids")}
    with torch.no_grad():
        for batch in dataloader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
            }
            context, _, _, task_ids = model.prepare_inputs(batch)
            no_id = hypernetwork.task_representation(context, None)
            task = hypernetwork.task_representation(context, task_ids)
            spaces["support"].append(task if placement == "input" else no_id)
            spaces["support_no_id"].append(no_id)
            spaces["task"].append(task)
            spaces["weights"].append(hypernetwork.projection(task).float())
            spaces["task_categories"] += list(batch["task_category"])
            task_id = batch["task_id"]
            spaces["task_ids"] += task_id.tolist() if torch.is_tensor(task_id) else list(task_id)
    out = {name: torch.cat(spaces[name]).cpu().numpy() for name in SPACES}
    out["task_categories"] = np.array(spaces["task_categories"])
    out["task_ids"] = np.array(spaces["task_ids"])
    return out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for arm in args.arms:
        for seed in args.seeds:
            run_dir = INDIST_RUNS[arm].format(seed=seed)
            if not (Path(run_dir) / "best_model.ckpt").exists():
                print(f"SKIP   {arm}_seed{seed} (no checkpoint in {run_dir})")
                continue
            cfg, runtime_cfg, model = load_run(run_dir, "best_model.ckpt")
            model.to(args.device)
            datamodule = DATA_REGISTRY[cfg.data](**runtime_cfg)
            datamodule.setup()
            data = representations(model, datamodule.val_dataloader(), args.device)
            proj = model.hypernetwork.task_indicator_proj
            if proj is not None:
                data["task_embedding_table"] = proj.weight.detach().cpu().numpy().T
            np.savez_compressed(args.output_dir / f"{arm}_seed{seed}.npz", **data)

            labels = list(data["task_categories"])
            row = {"arm": arm, "seed": seed, **parameter_counts(model)}
            for space in SPACES:
                row[f"{space}_linear_probe"] = compute_linear_probe_accuracy(data[space], labels)[
                    "mean_accuracy"
                ]
                row[f"{space}_silhouette"] = float(silhouette_score(data[space], labels))
            rows.append(row)
            print(
                f"DONE   {arm}_seed{seed}: "
                + ", ".join(
                    f"{s} probe {row[f'{s}_linear_probe']:.3f} / sil {row[f'{s}_silhouette']:.3f}"
                    for s in SPACES
                )
            )

    if rows:
        with open(args.output_dir / "cluster_metrics.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
