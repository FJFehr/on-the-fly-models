"""Generate the 10 arc1d_v2_generalization leaf configs (5 held-out categories x
{notd, frozentd}) -- the v2-scale (dim=4, 10,156-param matched-scale hypernetwork) rerun of
arc1d_hypermodel_looped_rope_canon_generalization's leave-one-task-out grid. Same held-out
categories and siblings as that experiment; task_categories/val_task_categories mechanism the
same but trimmed to 14 categories (1d_recolor_cmp dropped, matching the main v2 story's
convention). Only `notd`/`frozentd` built -- plain `td` dropped per project precedent
(historically bimodal/unstable, see arc1d_v2_hypernetwork_multitask's README). Architecture is
the same matched-scale recipe as arc1d_v2_compositional_generalization. See that experiment's
README.md for the full held-out-category rationale, and
arc1d_v2_hypernetwork_multitask/shrink_matched_dim4.yaml for the architecture this reuses.

Usage:
    uv run python scripts/gen_v2_generalization_configs.py
"""

from pathlib import Path

OUT_DIR = Path("configs/experiments/arc1d_v2_generalization")

ALL_CATEGORIES = [
    "1d_denoising_1c",
    "1d_denoising_mc",
    "1d_fill",
    "1d_flip",
    "1d_hollow",
    "1d_mirror",
    "1d_move_1p",
    "1d_move_2p",
    "1d_move_2p_dp",
    "1d_move_3p",
    "1d_move_dp",
    "1d_pcopy_1c",
    "1d_pcopy_mc",
    "1d_scale_dp",
]

# (held-out category, short name, sibling note) -- identical set to
# arc1d_hypermodel_looped_rope_canon_generalization's README table.
HELD_OUT = [
    ("1d_move_2p", "move2p", "1d_move_1p, 1d_move_3p, 1d_move_dp, 1d_move_2p_dp"),
    ("1d_denoising_mc", "denoisingmc", "1d_denoising_1c"),
    ("1d_flip", "flip", "1d_mirror (different mechanism -- weaker pairing)"),
    ("1d_pcopy_mc", "pcopymc", "1d_pcopy_1c"),
    ("1d_hollow", "hollow", "1d_fill"),
]

VARIANTS = {
    "notd": {
        "num_tasks": "null",
        "freeze_task_indicator": "false",
        "desc": "No task-identity signal at all -- the model must infer the task purely from the 3-shot support examples.",
    },
    "frozentd": {
        "num_tasks": "18",
        "freeze_task_indicator": "true",
        "desc": "One-hot task-identity embedding present but frozen at random init for every category, trained or not, so the held-out category's column is drawn from the same distribution the rest of the network learned to interpret, not the one column that never moved while its 17 siblings did.",
    },
}


def yaml_list(items: list[str]) -> str:
    return "\n".join(f"  - {item}" for item in items)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for held_out, short, siblings in HELD_OUT:
        train_categories = [c for c in ALL_CATEGORIES if c != held_out]
        for variant, cfg in VARIANTS.items():
            path = OUT_DIR / f"arm_{short}_{variant}.yaml"
            content = f"""# Held out: {held_out} (sibling(s) remaining in training: {siblings})
# Variant: {variant} -- {cfg["desc"]}
_base_: configs/experiments/arc1d_v2_generalization/base.yaml
experiment_name: v2_generalization_{short}_{variant}

task_categories:
{yaml_list(train_categories)}
val_task_categories:
{yaml_list(ALL_CATEGORIES)}

hyper_head:
  num_tasks: {cfg["num_tasks"]}
  freeze_task_indicator: {cfg["freeze_task_indicator"]}
"""
            path.write_text(content)
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
