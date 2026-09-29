"""`evaluate_on` decides which weights the end-of-training evaluation scores.

- "final": the weights training ended with, also saved to final_model.ckpt.
- "best": the best checkpoint by primary_metric, reloaded before evaluation.

Experiments 1, 5 and 6 use "final" (how their results were produced); 2 uses "best".
"""

import pytest
import torch

from data_modules import DATA_REGISTRY
from lightning_modules import MODEL_REGISTRY
from tests.test_train_lifecycle import write_test_config
from training.config import build_runtime_config_dict, load_config
from training.logging import create_wandb_logger, describe_model
from training.trainer import build_callbacks, build_trainer, run_post_training_artifacts


def train_briefly(tmp_path, monkeypatch, make_arc1d_dataset_dict, evaluate_on):
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.chdir(tmp_path)
    data_dir = make_arc1d_dataset_dict(
        n_base_tasks=2, n_variants_per_base_task=1, categories=["1d_move_1p"]
    )
    config_path = tmp_path / "config.yaml"
    write_test_config(config_path, data_dir=data_dir, output_path=str(tmp_path / "output"))
    cfg = load_config(
        str(config_path),
        ["save_checkpoints=true", f"evaluate_on={evaluate_on}", "max_steps=6"],
    )
    runtime_cfg = build_runtime_config_dict(cfg)
    model = MODEL_REGISTRY[cfg.model](**runtime_cfg)
    datamodule = DATA_REGISTRY[cfg.data](**runtime_cfg)
    logger = create_wandb_logger(cfg, runtime_cfg, describe_model(model), run_name="test")
    callbacks, checkpoint_callback = build_callbacks(cfg, model, wandb_logger=logger)
    trainer = build_trainer(cfg, runtime_cfg, callbacks, logger, model=model)
    trainer.fit(model=model, datamodule=datamodule)
    final_weights = {k: v.clone() for k, v in model.state_dict().items()}
    _, _, checkpoint_path = run_post_training_artifacts(
        cfg, trainer, model, datamodule, checkpoint_callback, wandb_logger=logger
    )
    return cfg, model, final_weights, checkpoint_path


def same_weights(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_final_evaluates_and_saves_the_weights_training_ended_with(
    tmp_path, monkeypatch, make_arc1d_dataset_dict
):
    cfg, model, final_weights, checkpoint_path = train_briefly(
        tmp_path, monkeypatch, make_arc1d_dataset_dict, "final"
    )
    assert checkpoint_path is None
    assert same_weights(model.state_dict(), final_weights)
    saved = torch.load(tmp_path / "output" / "final_model.ckpt", weights_only=False)
    assert same_weights(saved["state_dict"], final_weights)


def test_best_reloads_the_best_checkpoint(tmp_path, monkeypatch, make_arc1d_dataset_dict):
    cfg, model, _, checkpoint_path = train_briefly(
        tmp_path, monkeypatch, make_arc1d_dataset_dict, "best"
    )
    assert checkpoint_path is not None and checkpoint_path.endswith("best_model.ckpt")
    best = torch.load(checkpoint_path, weights_only=False)["state_dict"]
    assert same_weights(model.state_dict(), best)


def test_unknown_evaluate_on_is_rejected(tmp_path, monkeypatch, make_arc1d_dataset_dict):
    with pytest.raises(ValueError, match="evaluate_on"):
        train_briefly(tmp_path, monkeypatch, make_arc1d_dataset_dict, "latest")
