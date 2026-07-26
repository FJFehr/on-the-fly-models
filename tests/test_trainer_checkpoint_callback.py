"""Tests for the save_checkpoints opt-out on the shared checkpoint callback."""

from omegaconf import OmegaConf

from training.trainer import create_checkpoint_callback


def make_cfg(output_path: str, **overrides) -> OmegaConf:
    base = {"primary_metric": "val_loss", "output_path": output_path}
    base.update(overrides)
    return OmegaConf.create(base)


def test_save_checkpoints_defaults_to_true(tmp_path):
    cfg = make_cfg(str(tmp_path))
    callback = create_checkpoint_callback(cfg)
    assert callback.save_last is True
    assert callback.save_top_k == 1


def test_save_checkpoints_false_disables_saving(tmp_path):
    cfg = make_cfg(str(tmp_path), save_checkpoints=False)
    callback = create_checkpoint_callback(cfg)
    assert callback.save_last is False
    assert callback.save_top_k == 0


def test_save_checkpoints_true_is_explicit_noop(tmp_path):
    cfg = make_cfg(str(tmp_path), save_checkpoints=True)
    callback = create_checkpoint_callback(cfg)
    assert callback.save_last is True
    assert callback.save_top_k == 1
