"""Generate ordering-ablation configs for arc1d_v2_backbone_capacity, dim=10 only.

Follow-up to the T1-T5 backbone progression: dim=10 was the width that
actually showed separation between steps (dim=16 was flat throughout), with
almost the entire jump concentrated at T3 (+ Canon). This asks whether
*when* each component is introduced changes the shape of that progression,
plus whether more training-time supervision depth (N_supervision=4) helps
the earlier stages catch up faster. Three questions, eight new configs
(everything else reuses the existing T1/T2/T4/T5 dim=10 results directly -
see docs comment in each step below for which original step it matches):

Experiment 1 - RoPE earlier. N_supervision and the RoPE-vs-sin-PE choice
are orthogonal axes, so "RoPE before N_sup" and "RoPE before Canon" turn
out to converge onto the same two new intermediate states:
  R2: + RoPE only (canon_set="", n_loops=1), N_sup=1 - "RoPE before N_sup"'s
      2nd step
  R3: + RoPE only (canon_set="", n_loops=1), N_sup=2 - the state both
      "RoPE before N_sup" (after adding N_sup) and "RoPE before Canon"
      (after adding RoPE to T2) land on
  "RoPE before N_sup" = T1 -> R2 -> R3 -> T4 -> T5
  "RoPE before Canon" = T1 -> T2 -> R3 -> T4 -> T5

Experiment 2 - RoPE, then looped, then N_sup, Canon last (explicit user
ordering):
  L3: + looped (canon_set="", n_loops=4), N_sup=1 - RoPE+looped, no Canon
  L4: + N_sup=2, same architecture as L3
  Full sequence: T1 -> R2 -> L3 -> L4 -> T5 (T5 = full config, Canon
  re-added last, identical to the existing T5 endpoint)

Experiment 3 - N_supervision=4 on the original T2-T5 sequence (T1 stays
N_sup=1, unaffected by this axis). Compute-matched per the round's
convention: max_steps halved again (2000, vs T2-T5's 4000 at N_sup=2) so
total optimizer updates stay at 8000 throughout, same as every other cell
in this story.
  T2n4, T3n4, T4n4, T5n4 - same architecture as T2/T3/T4/T5, N_sup=4.

8 new configs x 15 tasks x 3 seeds = 360 jobs. dim=10 only, per Fabio's
decision - dim=16 was flat at every step in the original run and is
unlikely to show anything these orderings would change.
"""

import copy
from pathlib import Path

import yaml

SRC_TASKS = Path("configs/experiments/arc1d_recursion_ablation")
DST = Path("configs/experiments/arc1d_v2_backbone_capacity")
BASE_CFG = "configs/experiments/arc1d_recursion_ablation/base_ablation.yaml"
PROJECT = "arc1d_v2_backbone_capacity"

SKIP_DIRS = {"overfit", "1d_padded_fill", "1d_recolor_cnt", "1d_recolor_oe"}

DIM = 10
HEADS = 1

MUON_PARAMS = {
    "optimizer": "Muon",
    "muon_lr": 0.005,
    "muon_momentum": 0.95,
}

CANON_PARAMS = {
    "canon_set": "ABCD",
    "canon_kernel": 5,
    "canon_activation": True,
    "canon_residual": True,
    "canon_causal": False,
}

NO_CANON_PARAMS = {**CANON_PARAMS, "canon_set": ""}


def _rope_backbone(canon_params: dict, n_loops: int) -> dict:
    return {
        "name": "rope_canon_looped_transformer",
        "params": {
            "hidden_dim": DIM,
            "num_heads": HEADS,
            "inner_dim": DIM,
            "inner_num_heads": HEADS,
            "n_loops": n_loops,
            "dropout": 0.1,
            **canon_params,
            "use_block_skip": False,
            "use_loop_skip": False,
        },
    }


def r2() -> dict:
    """+ RoPE only (no Canon), N_sup=1. 'RoPE before N_sup''s 2nd step."""
    return {
        "model": "direct_supervised",
        **MUON_PARAMS,
        "learning_rate": 0.0005,
        "max_steps": 8000,
        "task_encoding": {"embedding_dim": DIM, "value_vocab_size": 11, "use_sinusoidal_pe": False},
        "backbone_model": _rope_backbone(NO_CANON_PARAMS, n_loops=1),
    }


def r3() -> dict:
    """+ RoPE only (no Canon), N_sup=2. Where both 'RoPE before N_sup' and
    'RoPE before Canon' converge."""
    cfg = r2()
    cfg["model"] = "looped_supervised"
    cfg["N_supervision"] = 2
    cfg["max_steps"] = 4000
    return cfg


def l3() -> dict:
    """+ looped (still no Canon), N_sup=1. RoPE+looped without Canon."""
    cfg = r2()
    cfg["backbone_model"] = _rope_backbone(NO_CANON_PARAMS, n_loops=4)
    return cfg


def l4() -> dict:
    """+ N_sup=2, same RoPE+looped-no-Canon architecture as l3."""
    cfg = l3()
    cfg["model"] = "looped_supervised"
    cfg["N_supervision"] = 2
    cfg["max_steps"] = 4000
    return cfg


def _nsup4(canon_params: dict, n_loops: int, use_sinusoidal_pe: bool) -> dict:
    return {
        "model": "looped_supervised",
        **MUON_PARAMS,
        "learning_rate": 0.0005,
        "N_supervision": 4,
        "max_steps": 2000,
        "task_encoding": {
            "embedding_dim": DIM,
            "value_vocab_size": 11,
            "use_sinusoidal_pe": use_sinusoidal_pe,
        },
        "backbone_model": _rope_backbone(canon_params, n_loops) if not use_sinusoidal_pe else {
            "name": "canon_transformer" if canon_params.get("canon_set") else "transformer",
            "params": {
                "hidden_dim": DIM, "num_layers": 3, "num_heads": HEADS, "dropout": 0.1,
                **({} if not canon_params.get("canon_set") else CANON_PARAMS),
            },
        },
    }


def t2n4() -> dict:
    """N_sup=4 version of T2 (vanilla arch, sin PE)."""
    return _nsup4(canon_params={}, n_loops=1, use_sinusoidal_pe=True)


def t3n4() -> dict:
    """N_sup=4 version of T3 (+ Canon, sin PE)."""
    return _nsup4(canon_params=CANON_PARAMS, n_loops=1, use_sinusoidal_pe=True)


def t4n4() -> dict:
    """N_sup=4 version of T4 (+ RoPE, flat)."""
    return _nsup4(canon_params=CANON_PARAMS, n_loops=1, use_sinusoidal_pe=False)


def t5n4() -> dict:
    """N_sup=4 version of T5 (+ looped)."""
    return _nsup4(canon_params=CANON_PARAMS, n_loops=4, use_sinusoidal_pe=False)


STEP_BUILDERS = {
    "R2": r2,
    "R3": r3,
    "L3": l3,
    "L4": l4,
    "T2n4": t2n4,
    "T3n4": t3n4,
    "T4n4": t4n4,
    "T5n4": t5n4,
}

STEP_SUFFIX = {
    "R2": "rope_only",
    "R3": "rope_nsup",
    "L3": "rope_looped_n1",
    "L4": "rope_looped_nsup",
    "T2n4": "nsup4",
    "T3n4": "canon_nsup4",
    "T4n4": "rope_nsup4",
    "T5n4": "looped_nsup4",
}

DST.mkdir(parents=True, exist_ok=True)

n_written = 0
for task_dir in sorted(d for d in SRC_TASKS.iterdir() if d.is_dir()):
    if task_dir.name in SKIP_DIRS:
        continue
    task = task_dir.name
    out_task_dir = DST / task
    out_task_dir.mkdir(exist_ok=True)

    src_a = task_dir / "A_transformer.yaml"
    with open(src_a) as f:
        src_cfg = yaml.safe_load(f)
    task_categories = src_cfg.get("task_categories", [task])

    for step, builder in STEP_BUILDERS.items():
        suffix = STEP_SUFFIX[step]
        exp_name = f"v2_{step}_dim{DIM}_{task}_{suffix}"
        cfg = {
            "_base_": BASE_CFG,
            "experiment_name": exp_name,
            "project_name": PROJECT,
            "task_categories": task_categories,
            "save_checkpoints": False,
            **copy.deepcopy(builder()),
        }
        out_path = out_task_dir / f"{step}_dim{DIM}_{suffix}.yaml"
        with open(out_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        n_written += 1

print(f"Written {n_written} configs to {DST}/ (x3 seeds at launch = {n_written * 3} jobs)")
