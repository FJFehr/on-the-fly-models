"""Generate the flip+recolor hypernetwork capacity sweep: n_loops x N_supervision x skip.

Targets the four phase-1 hypermodel_looped tasks that don't hit ~100% val exact match
(1d_flip, 1d_recolor_cmp/cnt/oe) -- every other task category already works, so this is
scoped to just these four.

Verified (not assumed, see git history / README) that the recolor tasks have no
support/query data bug: for each recolor rule, the 3 support pairs always fully and
correctly determine the query (200/200 real task instances checked per category). Output
colours genuinely vary across task instances in data/arc_1d_looped_augmented -- that is the
correct, intended shape of the task (colour identity is part of what must be inferred from
support each time, same as every other task category in this repo), not a bug to route
around. So this sweep stays on the real data throughout and instead asks whether more
model capacity lets the hypernetwork actually do that few-shot colour-binding + structural
reasoning:

  n_loops        in {4, 8, 16}          -- internal loop depth
  N_supervision  in {2, 4}              -- external supervision loop
  skip           in {none, block+loop}  -- capacity story's best-performing skip
                                            combination (matches T7/L4), vs no skip at all

3 x 2 x 2 = 12 configs per task x 4 tasks = 48 total, all on data/arc_1d_looped_augmented
(no canonical-colour variant -- that diagnostic was tried and deliberately dropped in favour
of directly testing whether capacity closes the gap on the true task). The
n_loops=4/N_supervision=2/no-skip cell in each task dir is exactly the existing phase-1
config, included in the naming scheme for completeness, not a new result.
"""

from pathlib import Path

import yaml

DST = Path("configs/experiments/arc1d_hypermodel_looped_recolor")
BASE_CFG = "configs/experiments/arc1d_hypermodel_looped/base_hypermodel_looped.yaml"
PROJECT = "arc1d_hypermodel_looped_recolor"
DATA_DIR = "data/arc_1d_looped_augmented"

TASKS = ["1d_flip", "1d_recolor_cmp", "1d_recolor_cnt", "1d_recolor_oe"]

LOOPS = [4, 8, 16]
NSUPS = [2, 4]
SKIP_VARIANTS = {
    "noskip": {"use_block_skip": False, "use_loop_skip": False},
    "skip": {"use_block_skip": True, "use_loop_skip": True},
}

BASE_TARGET_MODEL_PARAMS = {
    "hidden_dim": 16,
    "num_heads": 2,
    "inner_dim": 16,
    "inner_num_heads": 2,
    "dropout": 0.1,
    "canon_set": "ABCD",
    "canon_kernel": 5,
    "canon_activation": True,
    "canon_residual": True,
    "canon_causal": False,
}

DST.mkdir(parents=True, exist_ok=True)

n_written = 0
for task in TASKS:
    short_task = task.removeprefix("1d_")  # e.g. "recolor_cmp", "flip"
    out_task_dir = DST / task
    out_task_dir.mkdir(exist_ok=True)

    for n_loops in LOOPS:
        for nsup in NSUPS:
            for skip_key, skip_params in SKIP_VARIANTS.items():
                suffix = f"n{nsup}_loop{n_loops}_{skip_key}"
                target_params = dict(BASE_TARGET_MODEL_PARAMS)
                target_params["n_loops"] = n_loops
                target_params.update(skip_params)

                cfg = {
                    "_base_": BASE_CFG,
                    "task": short_task,
                    "experiment_name": f"looped_hyper_{short_task}_{suffix}",
                    "project_name": PROJECT,
                    "data_dir": DATA_DIR,
                    "task_categories": [task],
                    "val_task_categories": [task],
                    "N_supervision": nsup,
                    "target_model": {
                        "name": "rope_canon_looped_transformer",
                        "params": target_params,
                    },
                }
                out_path = out_task_dir / f"{suffix}.yaml"
                with open(out_path, "w") as f:
                    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
                n_written += 1

print(f"Written {n_written} configs to {DST}/")
