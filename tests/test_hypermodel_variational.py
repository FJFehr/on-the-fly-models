"""Focused tests for the optional beta-VAE bottleneck on the pooled task representation."""

import lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from models.hypermodel_lightning import HyperModelLightning


class _ListDataset(Dataset):
    def __init__(self, items: list[dict]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return self.items[index]


def build_model(
    hyper_head: dict | None = None,
    **kwargs,
) -> HyperModelLightning:
    return HyperModelLightning(
        hyper_model={
            "name": "transformer",
            "params": {
                "hidden_dim": 16,
                "num_layers": 1,
                "num_heads": 1,
                "output_dim": 8,
            },
        },
        target_model={
            "name": "rnn",
            "params": {
                "hidden_dim": 8,
                "num_layers": 1,
                "bidirectional": True,
            },
        },
        hyper_head=hyper_head,
        task_encoding={"embedding_dim": 8},
        input_dim=4,
        **kwargs,
    )


def make_batch() -> dict:
    return {
        "support_inputs": torch.tensor(
            [[[0, 1, 0, 1], [1, 0, 1, 0], [0, 0, 1, 1]]],
            dtype=torch.float32,
        ),
        "support_outputs": torch.tensor(
            [[[1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 0, 1]]],
            dtype=torch.float32,
        ),
        "query_input": torch.tensor([[1, 0, 0, 1]], dtype=torch.float32),
        "query_output": torch.tensor([[0, 1, 1, 0]], dtype=torch.float32),
    }


def test_variational_disabled_by_default():
    """variational defaults to False: no VAE heads are built, no KL is ever stashed."""
    model = build_model()
    assert model.hypermodel.variational is False
    assert model.hypermodel.vae_mu_head is None
    assert model.hypermodel.vae_logvar_head is None

    model(make_batch())

    assert model.hypermodel._last_kl_loss is None


def test_variational_forward_stashes_kl_loss_and_preserves_pooled_shape():
    """Enabling the bottleneck keeps hyper_output_dim unchanged and produces a scalar KL."""
    model = build_model({"variational": True})
    logits, _ = model(make_batch())

    kl = model.hypermodel._last_kl_loss
    assert kl is not None
    assert kl.ndim == 0
    assert kl.item() >= 0.0

    stashed = model.hypermodel._last_task_representation
    assert stashed.shape == (logits.shape[0], model.hypermodel.hyper_output_dim)


def test_variational_bottleneck_is_stochastic_in_train_mode_and_deterministic_in_eval_mode():
    model = build_model({"variational": True})
    batch = make_batch()

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


def test_variational_kl_loss_is_deterministic_across_train_mode_samples():
    """KL depends only on mu/logvar, not the stochastic sample -- it must stay constant
    across repeated train-mode forward calls on the same input, even though the sampled
    task representation itself varies each call (see the stochasticity test above)."""
    model = build_model({"variational": True})
    batch = make_batch()

    model.train()
    model(batch)
    kl_a = model.hypermodel._last_kl_loss.item()
    model(batch)
    kl_b = model.hypermodel._last_kl_loss.item()

    assert kl_a == kl_b


def test_variational_compatible_with_lora_adapter_path():
    """The VAE bottleneck must not disturb the lora_adapter projection's shape contract."""
    model = build_model({"variational": True, "lora_adapter": True, "lora_adapter_rank": 1})
    logits, targets = model(make_batch())
    assert logits.shape == targets.shape


def test_training_step_backprops_into_variational_heads():
    """A real training_step with variational=True and kl_beta>0 must reach the VAE heads'
    gradients, and must not raise (the manual-optimization/backward wiring stays intact)."""
    model = build_model({"variational": True}, kl_beta=1.0)
    dataloader = DataLoader(
        _ListDataset([make_batch(), make_batch()]),
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
