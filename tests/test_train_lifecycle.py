"""Does a run's core artifacts land where the reproduction pipeline expects them?

Nothing in the suite previously exercised train.py's actual lifecycle -- config load,
build model/datamodule from the real registries, fit, write artifacts. This calls that
same sequence directly (not through train.py's own main(), which permanently reassigns
sys.stdout/sys.stderr for log-tee-ing -- calling the underlying functions gets the same
real coverage without risking that leaking into the rest of the pytest session), with a
real but 1-step CPU trainer.fit(), and asserts config.yaml/model.txt/results.txt land at
the exact paths the rest of the repo's tooling (experiments/README.md's workflow,
scripts/run_config.sh) assumes. This is the one behavioral concern this test covers --
not a matrix over every model/data combination, and deliberately not
run_post_training_artifacts's hard-example-export/visualisation-gallery step, which is a
separate concern (does figure rendering work, not does the run save its artifacts).
"""

from pathlib import Path

from omegaconf import OmegaConf

from data_modules import DATA_REGISTRY
from models import MODEL_REGISTRY
from training.config import (
    build_runtime_config_dict,
    ensure_output_path,
    load_config,
    save_resolved_config,
)
from training.logging import (
    create_wandb_logger,
    describe_model,
    write_model_summary,
    write_results_file,
)
from training.trainer import build_callbacks, build_trainer


def write_test_config(config_path: Path, *, data_dir: str, output_path: str) -> None:
    config_path.write_text(
        f"""
seed: 42
project_name: test_lifecycle
experiment_name: smoke
output_path: {output_path}

model: direct_supervised
data: arc_1d_direct
data_dir: {data_dir}
primary_metric: val_query_exact_match
task_categories: ["1d_move_1p"]
train_split: train
val_split: dev
test_split: test

batch_size: 2
num_workers: 0
accelerator: cpu
devices: 1
max_steps: 1
optimizer: Adam
learning_rate: 0.001
save_checkpoints: false

prediction_task: multiclass
num_classes: 10
padding_idx: 10

task_encoding:
  embedding_dim: 4
  value_vocab_size: 11

backbone_model:
  name: transformer
  params:
    hidden_dim: 8
    num_layers: 1
    num_heads: 2
"""
    )


def test_a_real_one_step_run_writes_its_artifacts_to_output_path(
    tmp_path, monkeypatch, make_arc1d_dataset_dict
):
    # No network call / login prompt from create_wandb_logger below -- WandbLogger
    # respects WANDB_MODE transparently, no code-path changes needed to test it for real.
    # create_wandb_logger doesn't pass save_dir, so WandbLogger defaults to "." (CWD) --
    # chdir into tmp_path so its local run files land there instead of the repo root
    # (WANDB_DIR alone doesn't override that default; confirmed empirically).
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.chdir(tmp_path)

    data_dir = make_arc1d_dataset_dict(
        n_base_tasks=2, n_variants_per_base_task=1, categories=["1d_move_1p"]
    )
    output_path = tmp_path / "output"
    config_path = tmp_path / "config.yaml"
    write_test_config(config_path, data_dir=data_dir, output_path=str(output_path))

    cfg = load_config(str(config_path))
    runtime_cfg = build_runtime_config_dict(cfg)

    ensure_output_path(cfg.output_path)
    save_resolved_config(cfg)

    model = MODEL_REGISTRY[cfg.model](**runtime_cfg)
    model_summary = describe_model(model)
    write_model_summary(cfg.output_path, model_summary)

    datamodule = DATA_REGISTRY[cfg.data](**runtime_cfg)

    wandb_logger = create_wandb_logger(
        cfg, runtime_cfg, model_summary, run_name=cfg.experiment_name
    )
    callbacks, _ = build_callbacks(cfg, model, wandb_logger=wandb_logger)
    trainer = build_trainer(cfg, runtime_cfg, callbacks, wandb_logger, model=model)

    trainer.fit(model=model, datamodule=datamodule)
    val_results = trainer.validate(model=model, datamodule=datamodule, verbose=False)
    test_results = trainer.test(model=model, datamodule=datamodule, verbose=False)

    write_results_file(
        output_path=cfg.output_path,
        checkpoint_path=None,
        val_results=val_results,
        test_results=test_results,
    )

    assert trainer.global_step == 1  # the real 1-step fit actually happened

    saved_config = OmegaConf.load(output_path / "config.yaml")
    assert saved_config.experiment_name == "smoke"

    model_summary_text = (output_path / "model.txt").read_text()
    assert "Total parameters" in model_summary_text

    results_text = (output_path / "results.txt").read_text()
    assert "val_query_exact_match" in results_text or "val_" in results_text
    assert "test_" in results_text
