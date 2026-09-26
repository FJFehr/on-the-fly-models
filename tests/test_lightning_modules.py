"""Tests for the direct and hypernetwork Lightning modules."""

import pytest
import torch
from torch.func import functional_call


def make_direct_batch(n: int = 3, length: int = 6, task_category: str = "1d_move_1p") -> dict:
    generator = torch.Generator().manual_seed(0)
    inputs = torch.randint(0, 10, (n, length), generator=generator)
    outputs = torch.randint(0, 10, (n, length), generator=generator)
    inputs[:, -2:] = 10
    outputs[:, -2:] = 10
    return {"input": inputs, "output": outputs, "task_category": [task_category] * n}


# ---------------------------------------------------------------------------
# Direct model
# ---------------------------------------------------------------------------


def test_direct_forward_shapes(build_direct):
    logits, targets = build_direct()(make_direct_batch())
    assert logits.shape == (3, 6, 10)
    assert targets.shape == (3, 6)


def test_direct_loss_ignores_padded_positions(build_direct):
    model = build_direct()
    with torch.no_grad():
        logits, targets = model(make_direct_batch())
    changed = logits.clone()
    changed[:, -2:] = torch.randn_like(changed[:, -2:]) * 100
    torch.testing.assert_close(
        model.compute_loss(changed, targets), model.compute_loss(logits, targets)
    )


def test_direct_padded_positions_count_as_correct(build_direct):
    """How the paper's direct-training token accuracy was computed (DirectLightning.predict)."""
    model = build_direct()
    targets = torch.tensor([[3, 4, 10]])
    logits = torch.nn.functional.one_hot(torch.tensor([[3, 4, 0]]), 10).float()
    predictions, targets = model.predict(logits, targets)
    assert (predictions == targets).all()


def test_direct_task_embedding_changes_output_only_when_enabled(build_direct):
    batch_a = make_direct_batch(task_category="1d_move_1p")
    batch_b = make_direct_batch(task_category="1d_flip")
    for use_task_embedding, should_differ in ((False, False), (True, True)):
        model = build_direct(
            task_encoding={"embedding_dim": 4, "use_task_embedding": use_task_embedding}
        ).eval()
        differs = not torch.equal(model(batch_a)[0], model(batch_b)[0])
        assert differs == should_differ


def test_direct_muon_groups(build_direct):
    model = build_direct()
    names = {id(p): name for name, p in model.named_parameters()}
    adam, muon = model.muon_param_groups()
    muon_names = {names[id(p)] for p in muon["params"]}
    assert "head.weight" not in muon_names
    assert "backbone.input_projection.weight" in muon_names
    assert len(adam["params"]) + len(muon["params"]) == len(list(model.parameters()))


# ---------------------------------------------------------------------------
# Hypernetwork
# ---------------------------------------------------------------------------


def test_hypernetwork_forward_shapes(build_hypernetwork, make_hypernetwork_batch):
    logits, targets = build_hypernetwork()(make_hypernetwork_batch(n=2))
    assert logits.shape == (2, 4, 5, 10)
    assert targets.shape == (2, 4, 5)


def test_target_weights_are_never_trained(build_hypernetwork):
    model = build_hypernetwork()
    assert not any(p.requires_grad for p in model.hypernetwork.target.parameters())


@pytest.mark.parametrize("freeze", [True, False])
def test_task_indicator_gets_gradient_only_when_not_frozen(
    build_hypernetwork, make_hypernetwork_batch, freeze
):
    model = build_hypernetwork(
        hyper_head={"bottleneck_dim": 8, "num_tasks": 18, "freeze_task_indicator": freeze}
    )
    logits, targets = model(make_hypernetwork_batch())
    model.compute_loss(logits, targets).backward()
    grad = model.hypernetwork.task_indicator_proj.weight.grad
    assert (grad is None) == freeze


def test_vmapped_target_matches_a_loop_over_tasks(build_hypernetwork, make_hypernetwork_batch):
    """Each task runs with its own generated weights, exactly as a per-task loop would."""
    model = build_hypernetwork().eval()
    context, target_inputs, _, task_ids = model.prepare_inputs(make_hypernetwork_batch(n=3))
    with torch.no_grad():
        weights = model.hypernetwork.generate_weights(context, task_ids)
        vmapped = model.hypernetwork.run_target(weights, target_inputs)
        split = model.hypernetwork.split_weights(weights)
        looped = torch.stack(
            [
                functional_call(
                    model.hypernetwork.target,
                    {k: v[i] for k, v in split.items()},
                    target_inputs[i],
                )
                for i in range(3)
            ]
        )
    torch.testing.assert_close(vmapped, looped)


def test_hypernetwork_muon_groups(build_hypernetwork):
    model = build_hypernetwork(hyper_head={"bottleneck_dim": 8, "num_tasks": 18})
    names = {id(p): name for name, p in model.named_parameters()}
    _, muon = model.muon_param_groups()
    muon_names = {names[id(p)] for p in muon["params"]}
    assert "hypernetwork.projection.2.weight" in muon_names
    for excluded in ("input_projection", "output_head", "task_indicator_proj", "target."):
        assert not any(excluded in name for name in muon_names), excluded


# ---------------------------------------------------------------------------
# Shared: optimiser, schedule, determinism
# ---------------------------------------------------------------------------


def test_unknown_optimizer_is_rejected(build_direct):
    with pytest.raises(ValueError, match="optimizer"):
        build_direct(optimizer="SGD")


def test_warmup_then_cosine_schedule(build_direct):
    model = build_direct(
        learning_rate=1.0,
        warmup_steps=10,
        lr_scheduler={"name": "CosineAnnealingLR", "params": {"T_max": 110, "eta_min": 0.0}},
    )
    optimizer = model.configure_optimizers()["optimizer"]
    scheduler = model._build_scheduler(optimizer)
    lrs = []
    for _ in range(111):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert lrs[0] == pytest.approx(1e-6)
    assert lrs[10] == pytest.approx(1.0)
    assert lrs[60] == pytest.approx(0.5)  # halfway through the cosine part
    assert lrs[110] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("builder", ["build_direct", "build_hypernetwork"])
def test_same_seed_gives_same_initial_weights(request, builder):
    build = request.getfixturevalue(builder)
    torch.manual_seed(0)
    first = build().state_dict()
    torch.manual_seed(0)
    second = build().state_dict()
    assert all(torch.equal(first[k], second[k]) for k in first)
