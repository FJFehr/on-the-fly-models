"""Generate the layers x clip x task-descriptor grid for
arc1d_hypermodel_looped_rope_canon_grid.

See base.yaml in that directory for the full rationale: isolating gradient_clip_val alone
(5 vs 10) recovered most of the gap between the followup experiment's strong rank=8 result
and later experiments' weaker ones, but hyper_model.num_layers (2 in the followup experiment,
4 ever since) was never isolated, so layers and clip may interact. This also crosses in
hyper_head.num_tasks (task-descriptor presence), since the mechanism_ablation experiment
found removing it cost ~0.12 exact match and that effect might also depend on layers/clip.

hyper_model.params.num_layers in {2, 4, 8}
gradient_clip_val            in {5, 10, 20}
hyper_head.num_tasks         in {18 (with descriptor), null (without)}

3 x 3 x 2 = 18 configs.
"""

from pathlib import Path

import yaml

DST = Path("configs/experiments/arc1d_hypermodel_looped_rope_canon_grid")
BASE_CFG = "configs/experiments/arc1d_hypermodel_looped_rope_canon_grid/base.yaml"
PROJECT = "arc1d_hypermodel_looped_rope_canon_grid"

LAYERS = [2, 4, 8]
CLIPS = [5, 10, 20]
TASK_DESCRIPTOR = {"td": 18, "notd": None}

HYPER_MODEL_PARAMS = {
    "hidden_dim": 64,
    "num_heads": 4,
    "output_dim": 64,
    "canon_set": "ABCD",
    "canon_kernel": 5,
    "canon_activation": True,
    "canon_residual": True,
    "canon_causal": False,
    "block_size": 2048,
}

n_written = 0
for num_layers in LAYERS:
    for clip in CLIPS:
        for td_key, num_tasks in TASK_DESCRIPTOR.items():
            suffix = f"l{num_layers}_clip{clip}_{td_key}"
            hyper_model_params = dict(HYPER_MODEL_PARAMS)
            hyper_model_params["num_layers"] = num_layers

            cfg = {
                "_base_": BASE_CFG,
                "experiment_name": f"looped_hyper_rope_canon_grid_{suffix}",
                "gradient_clip_val": float(clip),
                "hyper_model": {
                    "name": "rope_canon_transformer",
                    "params": hyper_model_params,
                },
                "hyper_head": {
                    "pooling": "attention",
                    "bottleneck_dim": 128,
                    "num_tasks": num_tasks,
                    "lora_adapter": True,
                    "lora_adapter_rank": 8,
                    "lora_adapter_zero_backbone": False,
                },
            }
            out_path = DST / f"cell_{suffix}.yaml"
            with open(out_path, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            n_written += 1

print(f"Written {n_written} configs to {DST}/")
