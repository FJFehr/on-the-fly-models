"""Generate extended ablation config directories from the base 72 yamls.

Creates three new experiment variants alongside the existing
configs/experiments/arc1d_recursion_ablation/:

  arc1d_recursion_ablation_8k        — same model, 2× training steps
  arc1d_recursion_ablation_large     — wider/deeper model, same steps
  arc1d_recursion_ablation_large_8k  — wider/deeper model, 2× steps

Each variant inherits from the original base_ablation.yaml and applies
per-condition parameter overrides.  Seeds are handled at launch time via
CLI overrides (seed=N project_name=...) — no per-seed yaml files needed.

Scaling rules
-------------
  8k:       A/B max_steps 4000→8000; C/D max_steps 1000→2000
  large:    hidden_dim 256→512, num_heads 4→8
            A/C: num_layers 2→4 (standard depth scaling)
            B/D: n_loops 2→4, num_layers stays 1 (recursive depth scaling)
  large_8k: both combined
"""

import copy
import sys
from pathlib import Path

import yaml

SRC = Path("configs/experiments/arc1d_recursion_ablation")

# ---------------------------------------------------------------------------
# Variant definitions: per-condition parameter overrides
# ---------------------------------------------------------------------------
LARGE_PARAMS_AC = {"hidden_dim": 512, "num_layers": 4, "num_heads": 8}
LARGE_PARAMS_BD = {"hidden_dim": 512, "num_layers": 1, "num_heads": 8, "n_loops": 4}

VARIANTS: dict[str, dict] = {
    "8k": {
        "project_name": "arc1d_recursion_ablation_8k",
        "A": {"max_steps": 8000},
        "B": {"max_steps": 8000},
        "C": {"max_steps": 2000},
        "D": {"max_steps": 2000},
    },
    "large": {
        "project_name": "arc1d_recursion_ablation_large",
        "A": {"backbone_model": {"params": copy.deepcopy(LARGE_PARAMS_AC)}},
        "B": {"backbone_model": {"params": copy.deepcopy(LARGE_PARAMS_BD)}},
        "C": {"backbone_model": {"params": copy.deepcopy(LARGE_PARAMS_AC)}},
        "D": {"backbone_model": {"params": copy.deepcopy(LARGE_PARAMS_BD)}},
    },
    "large_8k": {
        "project_name": "arc1d_recursion_ablation_large_8k",
        "A": {"max_steps": 8000, "backbone_model": {"params": copy.deepcopy(LARGE_PARAMS_AC)}},
        "B": {"max_steps": 8000, "backbone_model": {"params": copy.deepcopy(LARGE_PARAMS_BD)}},
        "C": {"max_steps": 2000, "backbone_model": {"params": copy.deepcopy(LARGE_PARAMS_AC)}},
        "D": {"max_steps": 2000, "backbone_model": {"params": copy.deepcopy(LARGE_PARAMS_BD)}},
    },
}


def deep_merge(base: dict, override: dict) -> dict:
    """Return base with override applied recursively (does not mutate inputs)."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def condition_from_filename(name: str) -> str:
    """Return 'A', 'B', 'C', or 'D' from a yaml filename like 'A_transformer.yaml'."""
    cond = name[0].upper()
    if cond not in ("A", "B", "C", "D"):
        raise ValueError(f"Cannot determine condition from filename {name!r}")
    return cond


def generate(variant_name: str, variant_cfg: dict, dry_run: bool = False) -> int:
    dst = Path(f"configs/experiments/arc1d_recursion_ablation_{variant_name}")
    project_name = variant_cfg["project_name"]
    n_written = 0

    if not dry_run:
        dst.mkdir(parents=True, exist_ok=True)

    # --- per-task yamls ---
    # load_config resolves only ONE level of _base_, so each yaml points directly
    # to the original base_ablation.yaml (not a chained variant base).
    # project_name is set inline so individual configs work without CLI overrides.
    ORIGINAL_BASE = "configs/experiments/arc1d_recursion_ablation/base_ablation.yaml"

    SKIP_DIRS = {"overfit"}
    task_dirs = sorted(d for d in SRC.iterdir() if d.is_dir() and d.name not in SKIP_DIRS)
    for task_dir in task_dirs:
        out_task_dir = dst / task_dir.name
        if not dry_run:
            out_task_dir.mkdir(parents=True, exist_ok=True)

        for src_yaml_path in sorted(task_dir.glob("*.yaml")):
            cond = condition_from_filename(src_yaml_path.name)

            with src_yaml_path.open() as f:
                cfg = yaml.safe_load(f)

            # Always point to the original base (single-level resolution)
            cfg["_base_"] = ORIGINAL_BASE
            # Set project_name inline for standalone use; run script overrides it
            cfg["project_name"] = project_name

            # Apply condition-specific overrides
            if cond in variant_cfg:
                cfg = deep_merge(cfg, variant_cfg[cond])

            out_path = out_task_dir / src_yaml_path.name
            if not dry_run:
                out_path.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
            print(f"  {'(dry) ' if dry_run else ''}write {out_path}")
            n_written += 1

    return n_written


def main() -> None:
    dry_run = "--dry-run" in sys.argv

    if dry_run:
        print("DRY RUN — no files will be written\n")

    total = 0
    for variant_name, variant_cfg in VARIANTS.items():
        print(f"Variant: {variant_name}")
        n = generate(variant_name, variant_cfg, dry_run=dry_run)
        print(f"  {n} files {'would be ' if dry_run else ''}written\n")
        total += n

    print(f"Total: {total} files {'would be ' if dry_run else ''}written")
    print(f"Variants: {list(VARIANTS)}")


if __name__ == "__main__":
    main()
