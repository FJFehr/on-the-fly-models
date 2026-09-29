"""Tests for experiment 7's task-identity options: placement, trainability and zero vectors."""

import pytest
import torch

ENCODER = {"hidden_dim": 8, "num_layers": 1, "num_heads": 1, "output_dim": 4}


def hyper_head(placement: str = "latent", freeze: bool = True, **extra) -> dict:
    return {
        "bottleneck_dim": 8,
        "num_tasks": 18,
        "freeze_task_indicator": freeze,
        "placement": placement,
        **extra,
    }


def build(build_hypernetwork, seed: int = 0, **head):
    torch.manual_seed(seed)
    return build_hypernetwork(hyper_head=hyper_head(**head), encoder=ENCODER).eval()


@pytest.mark.parametrize("placement", ["latent", "input"])
def test_task_identity_changes_output(build_hypernetwork, make_hypernetwork_batch, placement):
    model = build(build_hypernetwork, placement=placement)
    a = model(make_hypernetwork_batch(task_category="1d_move_1p"))[0]
    b = model(make_hypernetwork_batch(task_category="1d_flip"))[0]
    assert not torch.equal(a, b)


def test_default_placement_is_latent(build_hypernetwork):
    model = build_hypernetwork(hyper_head={"bottleneck_dim": 8, "num_tasks": 18}, encoder=ENCODER)
    assert model.hypernetwork.task_indicator_placement == "latent"


def test_unknown_placement_is_rejected(build_hypernetwork):
    with pytest.raises(ValueError, match="task_indicator_placement"):
        build(build_hypernetwork, placement="middle")


def test_same_seed_gives_same_task_matrix_for_both_placements(build_hypernetwork):
    latent = build(build_hypernetwork, seed=1, placement="latent")
    inputs = build(build_hypernetwork, seed=1, placement="input")
    torch.testing.assert_close(
        latent.hypernetwork.task_indicator_proj.weight,
        inputs.hypernetwork.task_indicator_proj.weight,
    )


@pytest.mark.parametrize("placement", ["latent", "input"])
def test_input_placement_changes_encoder_input_only(
    build_hypernetwork, make_hypernetwork_batch, placement
):
    """Input placement adds the task embedding before the encoder; latent does not."""
    model = build(build_hypernetwork, placement=placement)
    seen = {}
    model.hypernetwork.encoder.register_forward_pre_hook(
        lambda _module, args: seen.setdefault("context", args[0])
    )
    context, _, _, task_ids = model.prepare_inputs(make_hypernetwork_batch())
    with torch.no_grad():
        model.hypernetwork.task_representation(context, task_ids)
    assert torch.equal(seen["context"], context) == (placement == "latent")


@pytest.mark.parametrize("placement", ["latent", "input"])
def test_zero_task_category_matches_no_task_identity(
    build_hypernetwork, make_hypernetwork_batch, placement
):
    """A category in zero_task_categories gets no task embedding, as if task_ids were None."""
    model = build(build_hypernetwork, placement=placement, zero_task_categories=["1d_move_1p"])
    batch = make_hypernetwork_batch(task_category="1d_move_1p")
    context, target_inputs, _, task_ids = model.prepare_inputs(batch)
    assert task_ids.is_floating_point() and not task_ids.any()
    with torch.no_grad():
        zeroed = model.hypernetwork(context, target_inputs, task_ids)
        without = model.hypernetwork(context, target_inputs, None)
    torch.testing.assert_close(zeroed, without)


def test_zero_task_categories_leave_other_categories_one_hot(
    build_hypernetwork, make_hypernetwork_batch
):
    zeroed = build(build_hypernetwork, zero_task_categories=["1d_flip"])
    plain = build(build_hypernetwork)
    batch = make_hypernetwork_batch(task_category="1d_move_1p")
    with torch.no_grad():
        torch.testing.assert_close(zeroed(batch)[0], plain(batch)[0])


def test_multi_hot_task_vector_adds_both_projections(build_hypernetwork):
    model = build(build_hypernetwork)
    projection = model.hypernetwork.task_indicator_proj
    multi_hot = torch.zeros(1, 18)
    multi_hot[0, [2, 5]] = 1.0
    torch.testing.assert_close(
        projection(multi_hot)[0], projection.weight[:, 2] + projection.weight[:, 5]
    )


@pytest.mark.parametrize("freeze", [True, False])
def test_learned_task_indicator_is_trainable(build_hypernetwork, freeze):
    model = build(build_hypernetwork, placement="input", freeze=freeze)
    assert model.hypernetwork.task_indicator_proj.weight.requires_grad == (not freeze)


# ---------------------------------------------------------------------------
# Experiment 7's configs build the model of the config they extend
# ---------------------------------------------------------------------------

EXP07 = "experiments/07_task_identity_ablation/configs"
EXP02_REFERENCE = "experiments/02_hypernetwork_multitask/configs/dim4_frozentd.yaml"
EXP05_REFERENCE = (
    "experiments/05_leave_one_out_task_generalization/configs/mirror_frozentd_seed1.yaml"
)
NEW_ARMS = ("learnedtd_latent", "frozentd_input", "learnedtd_input")
REFERENCE_CONFIGS = {
    **{f"{EXP07}/indist/{arm}.yaml": EXP02_REFERENCE for arm in NEW_ARMS},
    **{
        f"{EXP07}/loo/mirror_{arm}_seed1.yaml": EXP05_REFERENCE
        for arm in ("frozentd_latent", *NEW_ARMS)
    },
}


def build_from_config(path: str):
    from lightning_modules import MODEL_REGISTRY
    from training.config import build_runtime_config_dict, load_config

    cfg = load_config(path)
    torch.manual_seed(0)
    return MODEL_REGISTRY[cfg.model](**build_runtime_config_dict(cfg))


@pytest.mark.parametrize("config, reference", sorted(REFERENCE_CONFIGS.items()))
def test_experiment07_config_matches_its_reference(config, reference, monkeypatch):
    """Same parameters, shapes, initial values and Muon grouping; only the task projection's
    trainability may differ."""
    from pathlib import Path

    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    model, reference_model = build_from_config(config), build_from_config(reference)
    params = dict(model.named_parameters())
    reference_params = dict(reference_model.named_parameters())
    assert list(params) == list(reference_params)
    for name, p in params.items():
        torch.testing.assert_close(p, reference_params[name], msg=name)
        if name != "hypernetwork.task_indicator_proj.weight":
            assert p.requires_grad == reference_params[name].requires_grad, name

    def muon_names(m):
        names = {id(p): n for n, p in m.named_parameters()}
        return [
            names[id(p)] for p in next(g for g in m.muon_param_groups() if g["use_muon"])["params"]
        ]

    assert muon_names(model) == muon_names(reference_model)
