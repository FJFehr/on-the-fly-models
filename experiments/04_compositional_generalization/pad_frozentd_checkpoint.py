"""Pad a frozen (never-trained) task_indicator_proj from num_tasks=18 to 28 columns.

Experiment 2's dim=4 frozen_td checkpoints were trained with hyper_head.num_tasks=18 (no
reason for experiment 2 itself to reserve indices for compositional-holdout categories it
never evaluates on). scripts/eval_compositional_holdout.py needs num_tasks=28 (10 reserved
slots, indices 18-27, one per composite category -- see that script's own module docstring)
to build a model whose task_indicator_proj has rows to register those categories into.

Since freeze_task_indicator=true means this whole hypermodel.task_indicator_proj --
nn.Linear(num_tasks, hyper_output_dim, bias=False) -- is random-init and NEVER
gradient-updated (models/hypermodel.py:280-289: requires_grad_(False) right after
construction), padding it with 10 freshly-initialized columns is statistically identical to
having trained with num_tasks=28 from the start: both the real 18 columns and the 10 padded
ones are equally "untrained random values drawn from nn.Linear's default init," never touched
by an optimizer step either way. Not a hack -- a mathematically-justified extension of a layer
that was never learned in the first place. Build a fresh nn.Linear at the TARGET width (28) so
the padding columns come from the exact default-init distribution a genuine num_tasks=28 build
would have used, then keep only its columns [old_num_tasks:] as the padding, concatenated
after the real (frozen) columns [:old_num_tasks] from the source checkpoint.

The padded checkpoint loads cleanly into a model built from experiments/
04_compositional_generalization/configs/frozen_td.yaml (already num_tasks=28, otherwise
architecturally identical to experiment 2's own frozen_td recipe) via an ordinary strict
load_state_dict -- shapes match once padded, no other change needed.

Usage
-----
    CKPT=outputs/02_hypernetwork_multitask/hyper_multitask_dim4_frozentd_seed1/best_model.ckpt
    uv run python experiments/04_compositional_generalization/pad_frozentd_checkpoint.py \\
        --checkpoint "$CKPT" --num-tasks 28 --seed 1 --out /tmp/frozentd_seed1_padded.ckpt
"""

import argparse
from pathlib import Path

import torch
from torch import nn

KEY = "hypermodel.task_indicator_proj.weight"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True, help="Source checkpoint, e.g. exp 2's own.")
    parser.add_argument("--num-tasks", type=int, required=True, help="Target width (28 here).")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for the padding columns.")
    parser.add_argument("--out", required=True, help="Where to write the padded checkpoint.")
    return parser.parse_args()


def pad_task_indicator(state_dict: dict, num_tasks: int, seed: int) -> dict:
    old_weight = state_dict[KEY]
    hyper_output_dim, old_num_tasks = old_weight.shape
    if num_tasks <= old_num_tasks:
        msg = f"--num-tasks ({num_tasks}) must exceed the checkpoint's own ({old_num_tasks})"
        raise ValueError(msg)

    torch.manual_seed(seed)
    reference = nn.Linear(num_tasks, hyper_output_dim, bias=False)
    new_weight = torch.cat([old_weight, reference.weight.detach()[:, old_num_tasks:]], dim=1)
    assert new_weight.shape == (hyper_output_dim, num_tasks)

    state_dict = dict(state_dict)
    state_dict[KEY] = new_weight
    print(f"Padded {KEY}: {old_num_tasks} -> {num_tasks} columns")
    return state_dict


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    ckpt["state_dict"] = pad_task_indicator(ckpt["state_dict"], args.num_tasks, args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
