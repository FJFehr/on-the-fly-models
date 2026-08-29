"""Experiment 1: minimal model size, 14-task set.

Sweeps hidden_dim in {6, 4} below RC1's dim=10 (5,130-param backbone), with
task_encoding.embedding_dim FIXED at 10 throughout for comparability - the
RoPECanonLoopedTransformer's input_projection layer already decouples
embedding_dim from hidden_dim, so this needs no code changes.

Actual total parameters (embedder + backbone + head), measured by
instantiating DirectSupervisedLightning directly:

    hidden_dim=10 (RC1, existing): 5,400
    hidden_dim=6:                  2,444
    hidden_dim=4:                  1,398

Everything else matches RC1 exactly: RoPE + Canon (set ABCD, kernel 5),
Muon, direct_supervised, flat (n_loops=1), N_sup=1, max_steps=8000.

14-task set (1d_recolor_cmp excluded per the Phase 1 scope decision, see
docs/arc1d_story/06_phase1_findings.md).

2 sizes x 14 tasks x 3 seeds = 84 jobs.
"""

import copy
from pathlib import Path

import yaml

SRC_TASKS = Path("configs/experiments/arc1d_recursion_ablation")
DST = Path("configs/experiments/arc1d_v2_minimal_size")
BASE_CFG = "configs/experiments/arc1d_recursion_ablation/base_ablation.yaml"
PROJECT = "arc1d_v2_minimal_size"

SKIP_DIRS = {"overfit", "1d_padded_fill", "1d_recolor_cnt", "1d_recolor_oe", "1d_recolor_cmp"}

EMBEDDING_DIM = 10
HEADS = 1
SIZES = [6, 4]

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


def rc1_at(hidden_dim: int) -> dict:
    """RC1's architecture (RoPE + Canon, N_sup=1, flat), with hidden_dim
    swept and embedding_dim held fixed at 10."""
    return {
        "model": "direct_supervised",
        **MUON_PARAMS,
        "learning_rate": 0.0005,
        "max_steps": 8000,
        "task_encoding": {
            "embedding_dim": EMBEDDING_DIM,
            "value_vocab_size": 11,
            "use_sinusoidal_pe": False,
        },
        "backbone_model": {
            "name": "rope_canon_looped_transformer",
            "params": {
                "hidden_dim": hidden_dim,
                "num_heads": HEADS,
                "inner_dim": hidden_dim,
                "inner_num_heads": HEADS,
                "n_loops": 1,
                "dropout": 0.1,
                **CANON_PARAMS,
                "use_block_skip": False,
                "use_loop_skip": False,
            },
        },
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

    for dim in SIZES:
        exp_name = f"v2_minsize_dim{dim}_{task}"
        cfg = {
            "_base_": BASE_CFG,
            "experiment_name": exp_name,
            "project_name": PROJECT,
            "task_categories": task_categories,
            "save_checkpoints": False,
            **copy.deepcopy(rc1_at(dim)),
        }
        out_path = out_task_dir / f"dim{dim}.yaml"
        with open(out_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        n_written += 1

print(f"Written {n_written} configs to {DST}/ (x3 seeds at launch = {n_written * 3} jobs)")
