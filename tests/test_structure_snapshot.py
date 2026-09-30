"""Every experiment config builds the same model as before the models/ refactor.

`fixtures/structure_snapshot.json` was recorded from the pre-refactor code (see
fixtures/make_structure_snapshot.py). For every experiment config this checks, after
renaming old parameter names to new ones:
  - the same parameters, with the same shapes and trainability, in the same order
    (the order decides both initialisation and the hypernetwork's weight layout),
  - the same parameters optimised by Muon,
  - the same initial values for a fixed seed.
"""

import json
from pathlib import Path

import pytest
import torch

from lightning_modules import MODEL_REGISTRY
from tests.fixtures.old_checkpoints import new_parameter_name
from training.config import build_runtime_config_dict, load_config

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = json.loads((ROOT / "tests/fixtures/structure_snapshot.json").read_text())
# Experiments added after the snapshot was taken, so they have no pre-refactor entry.
POST_REFACTOR_EXPERIMENTS = ("07_task_identity_ablation", "08_optimiser_tuning")


def build(config_path: str):
    cfg = load_config(str(ROOT / config_path))
    runtime_cfg = build_runtime_config_dict(cfg)
    torch.manual_seed(SNAPSHOT["init_seed"])
    return MODEL_REGISTRY[cfg.model](**runtime_cfg)


@pytest.mark.parametrize("structure_key", sorted(SNAPSHOT["structures"]))
def test_structure_matches_pre_refactor_snapshot(structure_key, monkeypatch):
    # Configs use repo-relative _base_ paths.
    monkeypatch.chdir(ROOT)
    expected = SNAPSHOT["structures"][structure_key]
    # Configs deleted since the snapshot was taken are skipped.
    configs = [
        path
        for path, key in SNAPSHOT["configs"].items()
        if key == structure_key and (ROOT / path).exists()
    ]
    for config_path in configs:
        model = build(config_path)

        parameters = [
            [name, list(p.shape), p.requires_grad] for name, p in model.named_parameters()
        ]
        expected_parameters = [
            [new_parameter_name(n), shape, grad] for n, shape, grad in expected["parameters"]
        ]
        assert parameters == expected_parameters, config_path

        names = {id(p): name for name, p in model.named_parameters()}
        muon_group = next(g for g in model.muon_param_groups() if g["use_muon"])
        assert [names[id(p)] for p in muon_group["params"]] == [
            new_parameter_name(n) for n in expected["muon_parameters"]
        ], config_path

        init_sums = [p.detach().double().sum().item() for p in model.parameters()]
        assert init_sums == pytest.approx(expected["init_sums"], rel=1e-6, abs=1e-6), config_path


def test_every_experiment_config_is_covered():
    """Guards against a config being added or renamed without a snapshot entry.

    Configs may be deleted (the snapshot keeps their entries), but every existing config must
    have been checked against the pre-refactor code. Experiments added after the refactor are
    exempt; tests/test_task_indicator_placement.py checks experiment 7's configs against the
    experiment 2 and 5 configs they extend.
    """
    config_paths = {
        str(path.relative_to(ROOT))
        for path in (ROOT / "experiments").rglob("*.yaml")
        if str(path.relative_to(ROOT)) not in SNAPSHOT["errors"]
        and not any(
            path.is_relative_to(ROOT / "experiments" / e) for e in POST_REFACTOR_EXPERIMENTS
        )
    }
    assert config_paths <= set(SNAPSHOT["configs"])
