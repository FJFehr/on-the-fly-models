"""Offline PCA analysis for `1d_move_1p` binary bidirectional 1-layer RNN runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hyper-run",
        type=Path,
        required=True,
        help="Hyper run output directory.",
    )
    parser.add_argument(
        "--direct-run",
        type=Path,
        required=True,
        help="Direct supervised run output directory.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="val",
        help="Dataset split to analyze for hyper-generated vectors.",
    )
    parser.add_argument(
        "--checkpoint",
        choices=["best", "last"],
        default="best",
        help="Which checkpoint selector to load from each run directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where summary, CSVs, and PCA plot will be written.",
    )
    return parser.parse_args()


def _load_project_imports():
    from data_modules import DATA_REGISTRY
    from models import MODEL_REGISTRY
    from training.config import build_runtime_config_dict, load_config
    from visualisation.weight_space import (
        TARGET_TASK_CATEGORY,
        canonicalize_direct_rnn_state_dict,
        compute_pca_projection,
        flatten_param_dict,
        format_weight_space_summary,
        render_weight_space_pca_figure,
        validate_v1_weight_space_configs,
    )

    return {
        "DATA_REGISTRY": DATA_REGISTRY,
        "MODEL_REGISTRY": MODEL_REGISTRY,
        "TARGET_TASK_CATEGORY": TARGET_TASK_CATEGORY,
        "build_runtime_config_dict": build_runtime_config_dict,
        "canonicalize_direct_rnn_state_dict": canonicalize_direct_rnn_state_dict,
        "compute_pca_projection": compute_pca_projection,
        "flatten_param_dict": flatten_param_dict,
        "format_weight_space_summary": format_weight_space_summary,
        "load_config": load_config,
        "render_weight_space_pca_figure": render_weight_space_pca_figure,
        "validate_v1_weight_space_configs": validate_v1_weight_space_configs,
    }


def resolve_checkpoint_path(run_dir: Path, checkpoint_name: str) -> Path:
    if checkpoint_name == "best":
        candidates = sorted(run_dir.glob("best_model*.ckpt"))
    else:
        candidates = sorted(run_dir.glob("last*.ckpt"))

    if not candidates:
        msg = f"Could not find a {checkpoint_name!r} checkpoint in {run_dir}."
        raise FileNotFoundError(msg)
    return candidates[0]


def load_runtime_config(run_dir: Path) -> tuple[object, dict]:
    imports = _load_project_imports()
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        msg = f"Run directory does not contain config.yaml: {run_dir}"
        raise FileNotFoundError(msg)
    cfg = imports["load_config"](str(config_path))
    return cfg, imports["build_runtime_config_dict"](cfg)


def load_model_from_run(run_dir: Path, runtime_cfg: dict, checkpoint_name: str):
    model_registry = _load_project_imports()["MODEL_REGISTRY"]
    model = model_registry[runtime_cfg["model"]](**runtime_cfg)
    checkpoint_path = resolve_checkpoint_path(run_dir, checkpoint_name)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint_path


def collect_hyper_vectors(
    model,
    datamodule,
    split: str,
) -> tuple[torch.Tensor, list[dict[str, object]]]:
    dataloader_getter = getattr(datamodule, f"{split}_dataloader")
    dataloader = dataloader_getter()
    vectors: list[torch.Tensor] = []
    metadata: list[dict[str, object]] = []

    with torch.no_grad():
        for batch in dataloader:
            tensor_batch = {
                key: value if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            task_features, _, _ = model.prepare_inputs(tensor_batch)
            hyper_output = model.hypermodel.hypernetwork(task_features)
            batch_vectors = model.hypermodel.extract_parameter_vectors(hyper_output).cpu()
            task_ids = tensor_batch["task_id"].cpu().tolist()
            task_categories = list(tensor_batch["task_category"])
            for index, vector in enumerate(batch_vectors):
                vectors.append(vector)
                metadata.append(
                    {
                        "task_category": task_categories[index],
                        "task_id": int(task_ids[index]),
                        "split": split,
                    }
                )

    if not vectors:
        msg = f"No tasks were found in split {split!r}."
        raise ValueError(msg)
    return torch.stack(vectors), metadata


def build_direct_reference_vector(direct_model, hyper_model) -> torch.Tensor:
    imports = _load_project_imports()
    param_specs = hyper_model.hypermodel._target_parameter_specs
    direct_params = imports["canonicalize_direct_rnn_state_dict"](direct_model.state_dict())
    return imports["flatten_param_dict"](param_specs, direct_params)


def compute_summary(
    hyper_vectors: torch.Tensor,
    direct_vector: torch.Tensor,
) -> dict[str, object]:
    target_task_category = _load_project_imports()["TARGET_TASK_CATEGORY"]
    hyper_mean = hyper_vectors.mean(dim=0)
    hyper_variance = hyper_vectors.var(dim=0, unbiased=False)
    diff_to_direct = hyper_vectors - direct_vector.unsqueeze(0)
    direct_distance = torch.linalg.vector_norm(hyper_mean - direct_vector).item()
    mean_task_distance = torch.linalg.vector_norm(diff_to_direct, dim=1).mean().item()
    cosine_similarity = torch.nn.functional.cosine_similarity(
        hyper_mean.unsqueeze(0),
        direct_vector.unsqueeze(0),
    ).item()

    return {
        "task_category": target_task_category,
        "num_hyper_task_vectors": int(hyper_vectors.shape[0]),
        "canonical_vector_length": int(hyper_vectors.shape[1]),
        "hyper_mean_vector": hyper_mean.tolist(),
        "hyper_variance_vector": hyper_variance.tolist(),
        "l2_distance_hyper_mean_to_direct": direct_distance,
        "mean_l2_distance_hyper_tasks_to_direct": mean_task_distance,
        "cosine_similarity_hyper_mean_to_direct": cosine_similarity,
    }


def main() -> None:
    args = parse_args()
    imports = _load_project_imports()

    _, hyper_runtime_cfg = load_runtime_config(args.hyper_run)
    _, direct_runtime_cfg = load_runtime_config(args.direct_run)
    imports["validate_v1_weight_space_configs"](hyper_runtime_cfg, direct_runtime_cfg)

    hyper_model, hyper_checkpoint_path = load_model_from_run(
        args.hyper_run,
        hyper_runtime_cfg,
        args.checkpoint,
    )
    direct_model, direct_checkpoint_path = load_model_from_run(
        args.direct_run,
        direct_runtime_cfg,
        args.checkpoint,
    )

    datamodule = imports["DATA_REGISTRY"][hyper_runtime_cfg["data"]](**hyper_runtime_cfg)
    datamodule.setup()
    hyper_vectors, _metadata = collect_hyper_vectors(hyper_model, datamodule, args.split)
    direct_vector = build_direct_reference_vector(direct_model, hyper_model)
    if hyper_vectors.shape[1] != direct_vector.shape[0]:
        msg = (
            "Canonical vector length mismatch: "
            f"hyper={hyper_vectors.shape[1]}, direct={direct_vector.shape[0]}."
        )
        raise ValueError(msg)

    summary = compute_summary(hyper_vectors, direct_vector)
    summary.update(
        {
            "split": args.split,
            "checkpoint": args.checkpoint,
            "hyper_run": str(args.hyper_run),
            "direct_run": str(args.direct_run),
            "hyper_checkpoint": str(hyper_checkpoint_path),
            "direct_checkpoint": str(direct_checkpoint_path),
        }
    )

    pca = imports["compute_pca_projection"](hyper_vectors, direct_vector)
    figure = imports["render_weight_space_pca_figure"](
        hyper_coords=pca["hyper_coords"],
        hyper_mean_coords=pca["hyper_mean_coords"],
        direct_coords=pca["direct_coords"],
        explained_variance_ratio=pca["explained_variance_ratio"],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_text = imports["format_weight_space_summary"](
        hyper_run=summary["hyper_run"],
        direct_run=summary["direct_run"],
        split=summary["split"],
        checkpoint=summary["checkpoint"],
        hyper_checkpoint=summary["hyper_checkpoint"],
        direct_checkpoint=summary["direct_checkpoint"],
        num_hyper_task_vectors=summary["num_hyper_task_vectors"],
        canonical_vector_length=summary["canonical_vector_length"],
        l2_distance_hyper_mean_to_direct=summary["l2_distance_hyper_mean_to_direct"],
        mean_l2_distance_hyper_tasks_to_direct=summary["mean_l2_distance_hyper_tasks_to_direct"],
        cosine_similarity_hyper_mean_to_direct=summary["cosine_similarity_hyper_mean_to_direct"],
    )
    (args.output_dir / "summary.txt").write_text(summary_text)
    figure.savefig(args.output_dir / "pca.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
