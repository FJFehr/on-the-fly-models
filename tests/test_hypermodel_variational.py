"""Focused tests for the optional beta-VAE bottleneck on the pooled task representation."""

from types import SimpleNamespace

import lightning as pl
import pytest
import torch
from torch.utils.data import DataLoader, Dataset


class _ListDataset(Dataset):
    def __init__(self, items: list[dict]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return self.items[index]


def test_variational_disabled_by_default(build_hypermodel, make_hypermodel_batch):
    """variational defaults to False: no VAE heads are built, no KL is ever stashed."""
    model = build_hypermodel()
    assert model.hypermodel.variational is False
    assert model.hypermodel.vae_mu_head is None
    assert model.hypermodel.vae_logvar_head is None

    model(make_hypermodel_batch())

    assert model.hypermodel._last_kl_loss is None


def test_variational_forward_stashes_kl_loss_and_preserves_pooled_shape(
    build_hypermodel, make_hypermodel_batch
):
    """Enabling the bottleneck keeps hyper_output_dim unchanged and produces a scalar KL."""
    model = build_hypermodel(hyper_head={"variational": True})
    logits, _ = model(make_hypermodel_batch())

    kl = model.hypermodel._last_kl_loss
    assert kl is not None
    assert kl.ndim == 0
    assert kl.item() >= 0.0

    stashed = model.hypermodel._last_task_representation
    assert stashed.shape == (logits.shape[0], model.hypermodel.hyper_output_dim)


def test_variational_bottleneck_is_stochastic_in_train_mode_and_deterministic_in_eval_mode(
    build_hypermodel, make_hypermodel_batch
):
    model = build_hypermodel(hyper_head={"variational": True})
    batch = make_hypermodel_batch()

    model.train()
    model(batch)
    train_sample_a = model.hypermodel._last_task_representation.clone()
    model(batch)
    train_sample_b = model.hypermodel._last_task_representation.clone()
    assert not torch.equal(train_sample_a, train_sample_b)

    model.eval()
    model(batch)
    eval_mean_a = model.hypermodel._last_task_representation.clone()
    model(batch)
    eval_mean_b = model.hypermodel._last_task_representation.clone()
    assert torch.equal(eval_mean_a, eval_mean_b)


def test_variational_kl_loss_is_deterministic_across_train_mode_samples(
    build_hypermodel, make_hypermodel_batch
):
    """KL depends only on mu/logvar, not the stochastic sample -- it must stay constant
    across repeated train-mode forward calls on the same input, even though the sampled
    task representation itself varies each call (see the stochasticity test above)."""
    model = build_hypermodel(hyper_head={"variational": True})
    batch = make_hypermodel_batch()

    model.train()
    model(batch)
    kl_a = model.hypermodel._last_kl_loss.item()
    model(batch)
    kl_b = model.hypermodel._last_kl_loss.item()

    assert kl_a == kl_b


def test_variational_compatible_with_lora_adapter_path(build_hypermodel, make_hypermodel_batch):
    """The VAE bottleneck must not disturb the lora_adapter projection's shape contract."""
    model = build_hypermodel(
        hyper_head={"variational": True, "lora_adapter": True, "lora_adapter_rank": 1}
    )
    logits, targets = model(make_hypermodel_batch())
    assert logits.shape == targets.shape


def test_training_step_backprops_into_variational_heads(build_hypermodel, make_hypermodel_batch):
    """A real training_step with variational=True and kl_beta>0 must reach the VAE heads'
    gradients, and must not raise (the manual-optimization/backward wiring stays intact)."""
    model = build_hypermodel(hyper_head={"variational": True}, kl_beta=1.0)
    dataloader = DataLoader(
        _ListDataset([make_hypermodel_batch(), make_hypermodel_batch()]),
        batch_size=None,
        collate_fn=lambda item: item,
    )
    trainer = pl.Trainer(
        max_steps=1,
        enable_progress_bar=False,
        logger=False,
        enable_checkpointing=False,
        accelerator="cpu",
        devices=1,
    )
    trainer.fit(model, train_dataloaders=dataloader)

    assert model.hypermodel.vae_mu_head.weight.grad is not None
    assert model.hypermodel.vae_mu_head.weight.grad.abs().sum().item() > 0
    assert model.hypermodel.vae_logvar_head.weight.grad is not None
    assert model.hypermodel.vae_logvar_head.weight.grad.abs().sum().item() > 0


def test_current_kl_beta_defaults_to_constant(build_hypermodel):
    """Without kl_beta_anneal='cosine', _current_kl_beta always returns kl_beta unchanged."""
    model = build_hypermodel(hyper_head={"variational": True}, kl_beta=2.0)
    model.trainer = SimpleNamespace(global_step=0)
    assert model._current_kl_beta() == pytest.approx(2.0)

    model.trainer = SimpleNamespace(global_step=10_000)
    assert model._current_kl_beta() == pytest.approx(2.0)


def test_current_kl_beta_cosine_ramp_shape(build_hypermodel):
    """Cosine ramp starts at 0, reaches kl_beta at warmup_steps, and stays there after."""
    model = build_hypermodel(
        hyper_head={"variational": True},
        kl_beta=1.0,
        kl_beta_anneal="cosine",
        kl_beta_warmup_steps=100,
    )

    model.trainer = SimpleNamespace(global_step=0)
    assert model._current_kl_beta() == pytest.approx(0.0)

    model.trainer = SimpleNamespace(global_step=50)
    assert model._current_kl_beta() == pytest.approx(0.5, abs=1e-6)

    model.trainer = SimpleNamespace(global_step=100)
    assert model._current_kl_beta() == pytest.approx(1.0)

    model.trainer = SimpleNamespace(global_step=500)
    assert model._current_kl_beta() == pytest.approx(1.0)


def test_current_kl_beta_cosine_is_monotonically_nondecreasing(build_hypermodel):
    model = build_hypermodel(
        hyper_head={"variational": True},
        kl_beta=5.0,
        kl_beta_anneal="cosine",
        kl_beta_warmup_steps=40,
    )
    values = []
    for step in range(0, 45, 5):
        model.trainer = SimpleNamespace(global_step=step)
        values.append(model._current_kl_beta())

    assert values == sorted(values)
    assert values[0] == pytest.approx(0.0)
    assert values[-1] == pytest.approx(5.0)
