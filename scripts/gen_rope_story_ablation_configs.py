"""Generate configs for arc1d_rope_story_ablation.

Five conditions used to build the 8-step "story" heatmap in
scripts/plot_story_ablation.py, bridging the plain transformer
(arc1d_recursion_ablation) and the full Canon+skip model:

  SC1: Canon ABCD on the plain N_sup transformer (dim=512, no RoPE)  — feeds story step S3
  SC2: + RoPE, flat (n_loops=1, dim=16, Canon carries over from SC1) — feeds story step S4
  SC3: SC2 + block skip, still n_loops=1 (no looping, no wide middle)
       — diagnostic: isolates block skip's effect from looping/widening,
         since S5/S6 (looped/wide middle) showed no improvement over S4.
         Not part of the plotted 8-step narrative.
  SC4: uniform dim=16, n_loops=4, + block skip, no wide middle
       — S7 (skip_abcd_hw) without the wide middle: is looping-with-skip
         enough, or does skip need the wide middle specifically?
  SC5: uniform dim=16, n_loops=4, + block skip + loop skip, no wide middle
       — S8 (loop_skip_f4) without the wide middle, same question for the
         full skip combo. Neither SC4 nor SC5 is part of the plotted
         8-step narrative — both are diagnostics for the "does the
         sandwich (wide middle) actually do anything" question.
  SC6/SC7: uniform dim=20, n_loops=4, no skip / + block skip
       — parameter-matched to S6/S7 (wide outer=8,inner=32 = 16,672 backbone
         params): uniform dim=20 = 17,660 params, same budget. Tests whether
         it's raw capacity or the outer/inner bottleneck structure itself
         that lets skip connections help (SC4 showed skip hurts at dim=16
         uniform; this checks if simply adding params at uniform width,
         with no bottleneck, is enough instead).
  SC8/SC9: uniform dim=36, n_loops=4, no skip / + block skip
       — parameter-matched to arc1d_rope_dim_ablation cond J/K (wide
         outer=8,inner=64 = 55,552 backbone params): uniform dim=36 =
         53,100 params, same budget at a larger scale. Same capacity-vs-
         bottleneck question as SC6/SC7, one size up. (Cond K combines
         block+loop skip, not block skip alone, so the SC9 vs K comparison
         isn't perfectly matched on skip type — closest available reference.)

Neither SC6/SC7 nor SC8/SC9 is part of the plotted 8-step narrative.

  S3: Flat 3L transformer with RoPE  (n_loops=1, dim=16, no Canon)   — legacy, reference only
  S4: Looped middle with RoPE        (n_loops=4, dim=16, no Canon)   — legacy, reference only
  S5: Wide middle with RoPE          (outer=8, inner=32, n_loops=4, no Canon) — legacy, reference only

S3/S4/S5 were the original "no Canon" baselines for the RoPE progression. They are
kept as-is (not deleted, not regenerated with changes) for full-reproducibility
reference, but scripts/plot_story_ablation.py no longer plots them — SC1/SC2 now
carry the story's Canon/RoPE steps instead, so Canon appears earlier (step 3) and
the heatmap increases monotonically. See configs/experiments/arc1d_rope_story_ablation/README.md.

1d_padded_fill is excluded (not included in the story narrative).
"""

from pathlib import Path

import yaml

SRC_TASKS = Path("configs/experiments/arc1d_recursion_ablation")
DST = Path("configs/experiments/arc1d_rope_story_ablation")
BASE_CFG_STORY = "configs/experiments/arc1d_rope_story_ablation/base_story.yaml"
BASE_CFG_RECURSION = "configs/experiments/arc1d_recursion_ablation/base_ablation.yaml"
PROJECT = "arc1d_rope_story_ablation"
SKIP_DIRS = {"overfit", "1d_padded_fill"}

_ROPE_COMMON = {
    "dropout": 0.1,
    "canon_set": "",
    "canon_kernel": 5,
    "canon_activation": True,
    "canon_residual": True,
    "canon_causal": False,
    "use_block_skip": False,
    "use_loop_skip": False,
}

CONDITIONS = {
    # S3: flat 3-layer RoPE transformer (n_loops=1 = pre + 1×middle + post = 3 blocks)
    "S3": {
        "task_encoding": {
            "embedding_dim": 16,
            "value_vocab_size": 11,
            "use_sinusoidal_pe": False,
        },
        "backbone_model": {
            "name": "rope_canon_sandwich_transformer",
            "params": {
                **_ROPE_COMMON,
                "hidden_dim": 16,
                "num_heads": 2,
                "inner_dim": 16,
                "inner_num_heads": 2,
                "n_loops": 1,
            },
        },
    },
    # S4: looped middle, same dim (n_loops=4 = 6 effective blocks, weight-shared middle)
    "S4": {
        "task_encoding": {
            "embedding_dim": 16,
            "value_vocab_size": 11,
            "use_sinusoidal_pe": False,
        },
        "backbone_model": {
            "name": "rope_canon_sandwich_transformer",
            "params": {
                **_ROPE_COMMON,
                "hidden_dim": 16,
                "num_heads": 2,
                "inner_dim": 16,
                "inner_num_heads": 2,
                "n_loops": 4,
            },
        },
    },
    # S5: wide middle (outer=8, inner=32, n_loops=4) — concentrates capacity, no Canon yet
    "S5": {
        "backbone_model": {
            "name": "rope_canon_sandwich_transformer",
            "params": {
                **_ROPE_COMMON,
                "hidden_dim": 8,
                "num_heads": 1,
                "inner_dim": 32,
                "inner_num_heads": 4,
                "n_loops": 4,
            },
        },
    },
    # SC1: Canon ABCD added directly onto the S2 architecture (dim=512, no RoPE yet).
    # Mirrors arc1d_recursion_ablation_large_8k cond C exactly, but with backbone
    # canon_transformer instead of transformer (drop-in superset interface).
    "SC1": {
        "model": "looped_supervised",
        "N_supervision": 4,
        "learning_rate": 0.00025,
        "max_steps": 2000,
        "backbone_model": {
            "name": "canon_transformer",
            "params": {
                "hidden_dim": 512,
                "num_layers": 4,
                "num_heads": 8,
                "dropout": 0.1,
                "canon_set": "ABCD",
                "canon_kernel": 5,
                "canon_activation": True,
                "canon_residual": True,
                "canon_causal": False,
            },
        },
    },
    # SC2: + RoPE, flat (n_loops=1, dim=16). Identical to S3 but with Canon ABCD
    # carried over from SC1 instead of turned off.
    "SC2": {
        "task_encoding": {
            "embedding_dim": 16,
            "value_vocab_size": 11,
            "use_sinusoidal_pe": False,
        },
        "backbone_model": {
            "name": "rope_canon_sandwich_transformer",
            "params": {
                **_ROPE_COMMON,
                "canon_set": "ABCD",
                "hidden_dim": 16,
                "num_heads": 2,
                "inner_dim": 16,
                "inner_num_heads": 2,
                "n_loops": 1,
            },
        },
    },
    # SC3: SC2 + block skip, still n_loops=1. Diagnostic — isolates block skip's
    # effect from looping/wide-middle (S5/S6 showed no improvement over S4).
    "SC3": {
        "task_encoding": {
            "embedding_dim": 16,
            "value_vocab_size": 11,
            "use_sinusoidal_pe": False,
        },
        "backbone_model": {
            "name": "rope_canon_sandwich_transformer",
            "params": {
                **_ROPE_COMMON,
                "canon_set": "ABCD",
                "hidden_dim": 16,
                "num_heads": 2,
                "inner_dim": 16,
                "inner_num_heads": 2,
                "n_loops": 1,
                "use_block_skip": True,
            },
        },
    },
    # SC4: uniform dim=16, n_loops=4, + block skip, no wide middle.
    # S7 (skip_abcd_hw) without the wide middle.
    "SC4": {
        "task_encoding": {
            "embedding_dim": 16,
            "value_vocab_size": 11,
            "use_sinusoidal_pe": False,
        },
        "backbone_model": {
            "name": "rope_canon_sandwich_transformer",
            "params": {
                **_ROPE_COMMON,
                "canon_set": "ABCD",
                "hidden_dim": 16,
                "num_heads": 2,
                "inner_dim": 16,
                "inner_num_heads": 2,
                "n_loops": 4,
                "use_block_skip": True,
            },
        },
    },
    # SC5: uniform dim=16, n_loops=4, + block skip + loop skip, no wide middle.
    # S8 (loop_skip_f4) without the wide middle.
    "SC5": {
        "task_encoding": {
            "embedding_dim": 16,
            "value_vocab_size": 11,
            "use_sinusoidal_pe": False,
        },
        "backbone_model": {
            "name": "rope_canon_sandwich_transformer",
            "params": {
                **_ROPE_COMMON,
                "canon_set": "ABCD",
                "hidden_dim": 16,
                "num_heads": 2,
                "inner_dim": 16,
                "inner_num_heads": 2,
                "n_loops": 4,
                "use_block_skip": True,
                "use_loop_skip": True,
            },
        },
    },
    # SC6: uniform dim=20, n_loops=4, no skip. Param-matched control for SC7
    # and for S6 (wide outer=8/inner=32, 16,672 params vs this 17,660 params).
    "SC6": {
        "task_encoding": {
            "embedding_dim": 20,
            "value_vocab_size": 11,
            "use_sinusoidal_pe": False,
        },
        "backbone_model": {
            "name": "rope_canon_sandwich_transformer",
            "params": {
                **_ROPE_COMMON,
                "canon_set": "ABCD",
                "hidden_dim": 20,
                "num_heads": 2,
                "inner_dim": 20,
                "inner_num_heads": 2,
                "n_loops": 4,
            },
        },
    },
    # SC7: SC6 + block skip. Param-matched to S7 (skip_abcd_hw, wide outer=8/inner=32).
    "SC7": {
        "task_encoding": {
            "embedding_dim": 20,
            "value_vocab_size": 11,
            "use_sinusoidal_pe": False,
        },
        "backbone_model": {
            "name": "rope_canon_sandwich_transformer",
            "params": {
                **_ROPE_COMMON,
                "canon_set": "ABCD",
                "hidden_dim": 20,
                "num_heads": 2,
                "inner_dim": 20,
                "inner_num_heads": 2,
                "n_loops": 4,
                "use_block_skip": True,
            },
        },
    },
    # SC8: uniform dim=36, n_loops=4, no skip. Param-matched control for SC9 and
    # for arc1d_rope_dim_ablation cond J (wide outer=8/inner=64, 55,552 params
    # vs this 53,100 params).
    "SC8": {
        "task_encoding": {
            "embedding_dim": 36,
            "value_vocab_size": 11,
            "use_sinusoidal_pe": False,
        },
        "backbone_model": {
            "name": "rope_canon_sandwich_transformer",
            "params": {
                **_ROPE_COMMON,
                "canon_set": "ABCD",
                "hidden_dim": 36,
                "num_heads": 2,
                "inner_dim": 36,
                "inner_num_heads": 2,
                "n_loops": 4,
            },
        },
    },
    # SC9: SC8 + block skip. Param-matched to arc1d_rope_dim_ablation cond K
    # (wide outer=8/inner=64 + block+loop skip) — note K has loop skip too,
    # so this comparison isn't perfectly matched on skip type.
    "SC9": {
        "task_encoding": {
            "embedding_dim": 36,
            "value_vocab_size": 11,
            "use_sinusoidal_pe": False,
        },
        "backbone_model": {
            "name": "rope_canon_sandwich_transformer",
            "params": {
                **_ROPE_COMMON,
                "canon_set": "ABCD",
                "hidden_dim": 36,
                "num_heads": 2,
                "inner_dim": 36,
                "inner_num_heads": 2,
                "n_loops": 4,
                "use_block_skip": True,
            },
        },
    },
}

COND_BASE = {
    "S3": BASE_CFG_STORY,
    "S4": BASE_CFG_STORY,
    "S5": BASE_CFG_STORY,
    "SC1": BASE_CFG_RECURSION,
    "SC2": BASE_CFG_STORY,
    "SC3": BASE_CFG_STORY,
    "SC4": BASE_CFG_STORY,
    "SC5": BASE_CFG_STORY,
    "SC6": BASE_CFG_STORY,
    "SC7": BASE_CFG_STORY,
    "SC8": BASE_CFG_STORY,
    "SC9": BASE_CFG_STORY,
}

COND_SUFFIX = {
    "S3": "story_rope_flat",
    "S4": "story_rope_loop",
    "S5": "story_wide_nc",
    "SC1": "story_canon_plain",
    "SC2": "story_canon_rope_flat",
    "SC3": "story_flat_block_skip",
    "SC4": "story_uniform_block_skip",
    "SC5": "story_uniform_block_loop_skip",
    "SC6": "story_dim20_base",
    "SC7": "story_dim20_block_skip",
    "SC8": "story_dim36_base",
    "SC9": "story_dim36_block_skip",
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

    for cond, overrides in CONDITIONS.items():
        exp_name = f"ablation_{cond}_{task}_{COND_SUFFIX[cond]}"
        cfg = {
            "_base_": COND_BASE[cond],
            "experiment_name": exp_name,
            "project_name": PROJECT,
            "task_categories": task_categories,
            **overrides,
        }
        out_path = out_task_dir / f"{cond}_{COND_SUFFIX[cond]}.yaml"
        with open(out_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        n_written += 1

print(f"Written {n_written} configs to {DST}/")
