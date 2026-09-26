"""Record the model structure every experiment config builds, as a refactor reference.

Run once against the pre-refactor code (commit 5c5745d) to produce
`structure_snapshot.json`. `tests/test_structure_snapshot.py` then checks that the
refactored code builds the same models: same parameter shapes in the same order, the
same Muon/Adam optimiser grouping, and the same initial weights for a fixed seed.

Only builds models. No data is loaded and nothing is trained.

    PYTHONPATH=. python tests/fixtures/make_structure_snapshot.py
"""

import hashlib
import json
from pathlib import Path

import torch

from models import MODEL_REGISTRY
from training.config import build_runtime_config_dict, load_config

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = Path(__file__).with_name("structure_snapshot.json")
INIT_SEED = 0


def describe(model) -> dict:
    """Parameter list, optimiser grouping and an init checksum for one built model."""
    names = {id(p): name for name, p in model.named_parameters()}
    muon_group = next(g for g in model._build_muon_param_groups() if g["use_muon"])
    return {
        "parameters": [
            [name, list(p.shape), p.requires_grad] for name, p in model.named_parameters()
        ],
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "num_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "muon_parameters": [names[id(p)] for p in muon_group["params"]],
        # Sum of each parameter at init, in creation order. Matches only if the refactored
        # code creates the same parameters in the same order from the same RNG state.
        "init_sums": [round(p.detach().double().sum().item(), 8) for p in model.parameters()],
    }


def main() -> None:
    structures: dict[str, dict] = {}
    configs: dict[str, str] = {}
    errors: dict[str, str] = {}

    for path in sorted((ROOT / "experiments").rglob("*.yaml")):
        rel = str(path.relative_to(ROOT))
        try:
            cfg = load_config(rel)
            runtime_cfg = build_runtime_config_dict(cfg)
            runtime_cfg.pop("output_path", None)
            torch.manual_seed(INIT_SEED)
            model = MODEL_REGISTRY[cfg.model](**runtime_cfg)
        except Exception as exc:  # base.yaml files with unresolved placeholders, etc.
            errors[rel] = f"{type(exc).__name__}: {exc}"[:300]
            continue

        entry = {"model": cfg.model, **describe(model)}
        key = hashlib.sha1(json.dumps(entry, sort_keys=True).encode()).hexdigest()[:12]
        structures.setdefault(key, entry)
        configs[rel] = key

    OUT_PATH.write_text(
        json.dumps(
            {
                "init_seed": INIT_SEED,
                "torch_version": torch.__version__,
                "configs": configs,
                "structures": structures,
                "errors": errors,
            },
            indent=1,
        )
    )
    print(f"{len(configs)} configs, {len(structures)} distinct structures, {len(errors)} errors")
    for rel, msg in errors.items():
        print(f"  skipped {rel}: {msg}")


if __name__ == "__main__":
    main()
