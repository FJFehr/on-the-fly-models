"""Map pre-refactor (commit 5c5745d) parameter names to the current ones.

Used by the structure snapshot test, and to convert a pre-refactor checkpoint so it can
be evaluated with the current code:

    python tests/fixtures/old_checkpoints.py OLD.ckpt NEW.ckpt
"""

import sys

import torch

PREFIXES = [
    ("shared_task_token_embedder.", "embedder."),
    ("hypermodel.hypernetwork.", "hypernetwork.encoder."),
    ("hypermodel.target_model.", "hypernetwork.target."),
    ("hypermodel.hyper_pooling.pool.pool_query", "hypernetwork.pooler.query"),
    ("hypermodel.task_indicator_proj.", "hypernetwork.task_indicator_proj."),
    ("hypermodel.hyper_projection.", "hypernetwork.projection."),
]
# The old target/backbone was a "looped" transformer with n_loops=1: three named blocks.
BLOCKS = [
    (".pre_layer.", ".blocks.0."),
    (".middle_layer.", ".blocks.1."),
    (".post_layer.", ".blocks.2."),
]


def new_parameter_name(old_name: str) -> str:
    name = old_name
    for old, new in PREFIXES:
        if name.startswith(old):
            name = new + name[len(old) :]
            break
    for old, new in BLOCKS:
        name = name.replace(old, new)
    return name


def convert_checkpoint(old_path: str, new_path: str) -> None:
    checkpoint = torch.load(old_path, map_location="cpu", weights_only=False)
    checkpoint["state_dict"] = {
        new_parameter_name(name): value for name, value in checkpoint["state_dict"].items()
    }
    # Optimiser state is keyed by position, not name, and is not needed for evaluation.
    checkpoint.pop("optimizer_states", None)
    checkpoint.pop("lr_schedulers", None)
    torch.save(checkpoint, new_path)


if __name__ == "__main__":
    convert_checkpoint(sys.argv[1], sys.argv[2])
