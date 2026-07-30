"""Correctness checks for the Muon/AdamW parameter split.

A misclassified parameter here would silently corrupt training (e.g. running an embedding
table through Newton-Schulz orthogonalization) rather than raise an error, so this is checked
directly rather than only eyeballed once.
"""

import torch.nn as nn

from models.hypermodel_lightning import HyperModelLightning
from models.looped_supervised_lightning import LoopedSupervisedLightning
from models.transformer import RMSNorm


def build_zhu_model(muon_exclude_lora_heads: bool = False) -> HyperModelLightning:
    return HyperModelLightning(
        hyper_model={
            "name": "rope_canon_zhu_transformer",
            "params": {
                "hidden_dim": 16,
                "num_heads": 2,
                "num_layers": 2,
                "output_dim": 16,
                "canon_set": "ABCD",
                "canon_kernel": 3,
                "canon_activation": True,
                "canon_residual": True,
                "canon_causal": False,
                "block_size": 64,
                "use_rmsnorm": True,
                "use_qk_norm": True,
                "use_swiglu": True,
            },
        },
        target_model={
            "name": "rope_canon_looped_transformer",
            "params": {
                "hidden_dim": 16,
                "num_heads": 2,
                "inner_dim": 16,
                "inner_num_heads": 2,
                "n_loops": 2,
                "dropout": 0.0,
                "canon_set": "ABCD",
                "canon_kernel": 3,
                "canon_activation": True,
                "canon_residual": True,
                "canon_causal": False,
                "use_block_skip": False,
                "use_loop_skip": False,
            },
        },
        hyper_head={
            "pooling": "attention",
            "bottleneck_dim": 32,
            "num_tasks": 18,
            "freeze_task_indicator": False,
            "lora_adapter": True,
            "lora_adapter_rank": 2,
        },
        task_encoding={"embedding_dim": 8, "value_vocab_size": 11},
        optimizer="Muon",
        input_dim=4,
        muon_exclude_lora_heads=muon_exclude_lora_heads,
    )


def build_looped_supervised_model() -> LoopedSupervisedLightning:
    return LoopedSupervisedLightning(
        backbone_model={
            "name": "rope_canon_looped_transformer",
            "params": {
                "hidden_dim": 16,
                "num_heads": 2,
                "inner_dim": 16,
                "inner_num_heads": 2,
                "n_loops": 2,
                "dropout": 0.0,
                "canon_set": "ABCD",
                "canon_kernel": 3,
                "canon_activation": True,
                "canon_residual": True,
                "canon_causal": False,
                "use_block_skip": False,
                "use_loop_skip": False,
            },
        },
        task_encoding={"embedding_dim": 8, "value_vocab_size": 11},
        prediction_task="multiclass",
        num_classes=10,
        input_dim=4,
        optimizer="Muon",
    )


def build_zhu_model_variational() -> HyperModelLightning:
    return HyperModelLightning(
        hyper_model={
            "name": "rope_canon_zhu_transformer",
            "params": {
                "hidden_dim": 16,
                "num_heads": 2,
                "num_layers": 2,
                "output_dim": 16,
                "canon_set": "ABCD",
                "canon_kernel": 3,
                "canon_activation": True,
                "canon_residual": True,
                "canon_causal": False,
                "block_size": 64,
                "use_rmsnorm": True,
                "use_qk_norm": True,
                "use_swiglu": True,
            },
        },
        target_model={
            "name": "rope_canon_looped_transformer",
            "params": {
                "hidden_dim": 16,
                "num_heads": 2,
                "inner_dim": 16,
                "inner_num_heads": 2,
                "n_loops": 2,
                "dropout": 0.0,
                "canon_set": "ABCD",
                "canon_kernel": 3,
                "canon_activation": True,
                "canon_residual": True,
                "canon_causal": False,
                "use_block_skip": False,
                "use_loop_skip": False,
            },
        },
        hyper_head={
            "pooling": "attention",
            "bottleneck_dim": 32,
            "variational": True,
        },
        task_encoding={"embedding_dim": 8, "value_vocab_size": 11},
        optimizer="Muon",
        input_dim=4,
    )


def test_muon_param_split_excludes_non_matrix_and_output_layers():
    model = build_zhu_model()
    groups = model._build_muon_param_groups()
    adam_group, muon_group = groups[0], groups[1]
    assert adam_group["use_muon"] is False
    assert muon_group["use_muon"] is True

    muon_ids = {id(p) for p in muon_group["params"]}
    adam_ids = {id(p) for p in adam_group["params"]}
    assert muon_ids.isdisjoint(adam_ids)

    all_trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assert muon_ids | adam_ids == all_trainable

    named = dict(model.named_parameters())

    # Explicitly excluded Linear layers (Keller Jordan's guidance: input embedding and final
    # output layer stay on AdamW) must never end up in the Muon group. The frozen target_model
    # also happens to have its own input_projection/output_head (it mirrors the hypernetwork's
    # architecture as a shape template) -- those are requires_grad=False and correctly absent
    # from both groups entirely, so this only checks the trainable (hypernetwork-side) ones.
    excluded_names = [
        n
        for n in named
        if n.rsplit(".", 2)[-2] in ("input_projection", "output_head", "task_indicator_proj")
        and n.endswith(".weight")
        and named[n].requires_grad
    ]
    assert excluded_names, "expected to find input_projection/output_head/task_indicator_proj"
    for n in excluded_names:
        assert id(named[n]) not in muon_ids, f"{n} should be on AdamW, not Muon"
        assert id(named[n]) in adam_ids

    # Every non-Linear module's parameters (embeddings, norms, Canon depthwise convs) must
    # land on AdamW regardless of ndim, since a naive `ndim >= 2` rule would wrongly route
    # some of these (nn.Embedding weights, Conv1d weights) to Muon.
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            continue
        if isinstance(module, (nn.Embedding, nn.Conv1d, RMSNorm, nn.LayerNorm)):
            for param_name, param in module.named_parameters(prefix=name, recurse=False):
                if param.requires_grad:
                    assert id(param) not in muon_ids, f"{param_name} should be on AdamW"

    # The attention pooler's raw query vector is a bare nn.Parameter, not part of any Linear.
    pool_query_names = [n for n in named if n.endswith("pool_query")]
    assert pool_query_names, "expected to find the attention pooler's pool_query parameter"
    for n in pool_query_names:
        assert id(named[n]) not in muon_ids

    # At least one block-internal attention/MLP Linear weight is Muon-eligible.
    assert any(".blocks." in n and n in named and id(p) in muon_ids for n, p in named.items())

    # At least one hyper_head/LoRA projection Linear weight is Muon-eligible (Fabio's call:
    # these generate the actual target-model weights and are not treated as a classifier-style
    # "final output layer").
    assert any("lora_proj" in n and id(p) in muon_ids for n, p in named.items())


def test_configure_optimizers_builds_muon_optimizer_and_scheduler():
    model = build_zhu_model()
    model.lr_scheduler_cfg = {"name": "CosineAnnealingLR", "params": {"T_max": 100}}
    model.warmup_steps = 0
    result = model.configure_optimizers()
    optimizer = result["optimizer"]
    assert type(optimizer).__name__ == "SingleDeviceMuonWithAuxAdam"
    scheduler = result["lr_scheduler"]["scheduler"]
    assert scheduler.optimizer is optimizer


def test_muon_param_split_excludes_variational_bottleneck_heads():
    """vae_mu_head/vae_logvar_head are nn.Linear but must stay on AdamW, not Muon --
    without this exclusion they'd silently run through Newton-Schulz orthogonalization."""
    model = build_zhu_model_variational()
    assert model.hypermodel.variational is True

    groups = model._build_muon_param_groups()
    adam_group, muon_group = groups[0], groups[1]
    muon_ids = {id(p) for p in muon_group["params"]}
    adam_ids = {id(p) for p in adam_group["params"]}

    named = dict(model.named_parameters())
    vae_head_names = [
        n
        for n in named
        if n.rsplit(".", 2)[-2] in ("vae_mu_head", "vae_logvar_head") and n.endswith(".weight")
    ]
    assert vae_head_names, "expected to find vae_mu_head/vae_logvar_head weights"
    for n in vae_head_names:
        assert id(named[n]) not in muon_ids, f"{n} should be on AdamW, not Muon"
        assert id(named[n]) in adam_ids


def test_muon_param_split_excludes_lora_heads_when_flag_set():
    """muon_exclude_lora_heads=True must route lora_proj_a/lora_proj_b to AdamW.

    lora_proj_a/lora_proj_b are nn.ModuleLists, so a member's named_modules() path ends in
    its list index (e.g. "lora_proj_a.3"), not the attribute name -- this guards against the
    exclusion check only matching name.split(".")[-1], which would silently do nothing here.
    """
    model = build_zhu_model(muon_exclude_lora_heads=True)
    groups = model._build_muon_param_groups()
    adam_group, muon_group = groups[0], groups[1]
    muon_ids = {id(p) for p in muon_group["params"]}
    adam_ids = {id(p) for p in adam_group["params"]}

    named = dict(model.named_parameters())
    # lora_proj_a/lora_proj_b are nn.ModuleLists, so a member's parameter name has an extra
    # list-index segment (e.g. "hypermodel.lora_proj_a.3.weight") -- match by path segment,
    # not by position, since rsplit-by-position would land on the index instead of the name.
    lora_ab_names = [
        n
        for n in named
        if set(n.split(".")) & {"lora_proj_a", "lora_proj_b"} and n.endswith(".weight")
    ]
    assert lora_ab_names, "expected to find lora_proj_a/lora_proj_b weights"
    for n in lora_ab_names:
        assert id(named[n]) not in muon_ids, f"{n} should be on AdamW, not Muon"
        assert id(named[n]) in adam_ids

    # lora_proj_other has no low-rank structure and must remain Muon-eligible either way.
    lora_other_names = [n for n in named if "lora_proj_other" in n.split(".")]
    assert lora_other_names, "expected to find lora_proj_other weight"
    for n in lora_other_names:
        assert id(named[n]) in muon_ids, f"{n} should remain on Muon"


def test_muon_param_split_includes_lora_heads_by_default():
    """Sanity check that the default (muon_exclude_lora_heads=False) is unchanged."""
    model = build_zhu_model()
    groups = model._build_muon_param_groups()
    muon_ids = {id(p) for p in groups[1]["params"]}
    named = dict(model.named_parameters())
    lora_ab_names = [
        n
        for n in named
        if set(n.split(".")) & {"lora_proj_a", "lora_proj_b"} and n.endswith(".weight")
    ]
    assert lora_ab_names, "expected to find lora_proj_a/lora_proj_b weights"
    for n in lora_ab_names:
        assert id(named[n]) in muon_ids, f"{n} should default to Muon-eligible"


def test_looped_supervised_muon_param_split_excludes_head():
    """Mirrors test_muon_param_split_excludes_non_matrix_and_output_layers, but for
    LoopedSupervisedLightning (no hypernetwork) -- previously untested."""
    model = build_looped_supervised_model()
    groups = model._build_muon_param_groups()
    adam_group, muon_group = groups[0], groups[1]
    assert adam_group["use_muon"] is False
    assert muon_group["use_muon"] is True

    muon_ids = {id(p) for p in muon_group["params"]}
    adam_ids = {id(p) for p in adam_group["params"]}
    assert muon_ids.isdisjoint(adam_ids)

    all_trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assert muon_ids | adam_ids == all_trainable

    named = dict(model.named_parameters())

    # The final output Linear ("head") must stay on AdamW, not Muon.
    assert "head.weight" in named, "expected to find the head.weight parameter"
    assert id(named["head.weight"]) not in muon_ids
    assert id(named["head.weight"]) in adam_ids

    # Non-Linear modules (embeddings, norms, Canon depthwise convs) must never land on Muon.
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            continue
        if isinstance(module, (nn.Embedding, nn.Conv1d, RMSNorm, nn.LayerNorm)):
            for param_name, param in module.named_parameters(prefix=name, recurse=False):
                if param.requires_grad:
                    assert id(param) not in muon_ids, f"{param_name} should be on AdamW"

    # At least one backbone attention/MLP Linear weight is Muon-eligible.
    assert any(n.startswith("backbone.") and id(p) in muon_ids for n, p in named.items())


def test_looped_supervised_configure_optimizers_builds_muon_optimizer_and_scheduler():
    model = build_looped_supervised_model()
    model.lr_scheduler_cfg = {"name": "CosineAnnealingLR", "params": {"T_max": 100}}
    model.warmup_steps = 0
    result = model.configure_optimizers()
    optimizer = result["optimizer"]
    assert type(optimizer).__name__ == "SingleDeviceMuonWithAuxAdam"
    scheduler = result["lr_scheduler"]["scheduler"]
    assert scheduler.optimizer is optimizer
