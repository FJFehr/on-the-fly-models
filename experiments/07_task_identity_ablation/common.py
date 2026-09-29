"""Shared helpers for experiment 7's evaluation and analysis scripts.

Everything experiment-specific lives here rather than in the shared code: the five arms and
where their runs are, the composite -> component category mapping, a hypernetwork that takes a
ready-made task vector per category, and checkpoint loading.
"""

from pathlib import Path

import torch

from lightning_modules.hypernetwork import HypernetworkLightning
from models.embedding import TASK_CATEGORY_INDEX
from training.config import build_runtime_config_dict, load_config, locate_saved_run
from training.trainer import load_checkpoint_state

EXPERIMENT = "07_task_identity_ablation"
SEEDS = (1, 2, 3, 4, 5)
NUM_TASKS = 18

# Arm -> in-distribution run folder pattern. notd and frozentd_latent are experiment 2's dim-4
# runs (the same recipe and seeds); the three new arms are this experiment's.
EXP02 = "outputs/02_hypernetwork_multitask"
INDIST_RUNS = {
    "notd": f"{EXP02}/hyper_multitask_dim4_notd_seed{{seed}}",
    "frozentd_latent": f"{EXP02}/hyper_multitask_dim4_frozentd_seed{{seed}}",
    "learnedtd_latent": f"outputs/{EXPERIMENT}/indist_learnedtd_latent_seed{{seed}}",
    "frozentd_input": f"outputs/{EXPERIMENT}/indist_frozentd_input_seed{{seed}}",
    "learnedtd_input": f"outputs/{EXPERIMENT}/indist_learnedtd_input_seed{{seed}}",
}
ARMS = tuple(INDIST_RUNS)
TD_ARMS = ARMS[1:]
ARM_LABELS = {
    "notd": "w/o Task ID",
    "frozentd_latent": "Frozen, latent",
    "learnedtd_latent": "Learned, latent",
    "frozentd_input": "Frozen, input",
    "learnedtd_input": "Learned, input",
}

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


def short_name(category: str) -> str:
    """Mirrors gen_loo_configs.py (and experiment 5's gen_configs.py)."""
    return category.removeprefix("1d_").replace("_", "")


def loo_results_path(arm: str, held_out: str, seed: int) -> Path:
    """results.txt of one leave-one-out run, every arm scored with a zero held-out task
    vector: notd from experiment 5 (no task vector to zero), the td arms from this
    experiment."""
    short = short_name(held_out)
    if arm == "notd":
        return Path(
            f"outputs/05_leave_one_out_task_generalization/{short}_notd_seed{seed}/results.txt"
        )
    return Path(f"outputs/{EXPERIMENT}/loo_{short}_{arm}_seed{seed}/results.txt")


def parse_results(results_path: Path) -> dict[str, float]:
    """Every '<metric>: <number>' line of a results.txt."""
    scores = {}
    for line in results_path.read_text().splitlines():
        key, _, value = line.strip().partition(":")
        try:
            scores[key] = float(value)
        except ValueError:
            continue
    return scores


def read_parameter_counts(run_dir: str | Path) -> dict[str, int]:
    """Trainable and total parameters from a run's model.txt (older runs say "params")."""
    counts = {}
    for line in (Path(run_dir) / "model.txt").read_text().splitlines():
        label, _, value = line.partition(":")
        if label in ("Total trainable parameters", "Total trainable params"):
            counts["params_trainable"] = int(value.strip().replace(",", ""))
        elif label in ("Total parameters", "Total params"):
            counts["params_total"] = int(value.strip().replace(",", ""))
    return counts


# Each composite category of experiment 4 (data_modules/arc1d_compositional.py) and the two
# base categories whose rules it chains. shift = move by 3 (SHIFT_AMOUNT); move_dynamic =
# move to the pivot dot; copy stamps the shape in the marker dot's colour (pcopy_mc).
COMPOSITE_COMPONENTS = {
    "1d_comp_denoise1c_shift3": ("1d_denoising_1c", "1d_move_3p"),
    "1d_comp_fill_mirror": ("1d_fill", "1d_mirror"),
    "1d_comp_fill_shift3": ("1d_fill", "1d_move_3p"),
    "1d_comp_fill_movedynamic": ("1d_fill", "1d_move_dp"),
    "1d_comp_hollow_shift3": ("1d_hollow", "1d_move_3p"),
    "1d_comp_denoisemc_copy": ("1d_denoising_mc", "1d_pcopy_mc"),
    "1d_comp_denoisemc_denoise1c": ("1d_denoising_mc", "1d_denoising_1c"),
    "1d_comp_movedynamic_hollow": ("1d_move_dp", "1d_hollow"),
    "1d_comp_shift3_copy": ("1d_move_3p", "1d_pcopy_mc"),
    "1d_comp_denoisemc_mirror": ("1d_denoising_mc", "1d_mirror"),
}


def composite_task_vector(category: str, mode: str = "multihot") -> torch.Tensor:
    """Task vector (NUM_TASKS,) for a composite: its two components, summed or averaged."""
    vector = torch.zeros(NUM_TASKS)
    for component in COMPOSITE_COMPONENTS[category]:
        vector[TASK_CATEGORY_INDEX[component]] = 1.0
    return vector / 2 if mode == "mean" else vector


class TaskVectorHypernetwork(HypernetworkLightning):
    """HypernetworkLightning that looks up a fixed task vector for listed categories.

    `category_vectors` maps a category to a float (NUM_TASKS,) vector; other categories get
    their one-hot as usual. Used for composite categories, which have no index of their own.
    """

    def __init__(self, category_vectors: dict[str, torch.Tensor], **config):
        super().__init__(**config)
        self.category_vectors = category_vectors

    def prepare_inputs(self, batch: dict):
        categories = batch["task_category"]
        if self.hypernetwork.task_indicator_proj is None:
            return super().prepare_inputs(batch)
        # The parent looks every category up in TASK_CATEGORY_INDEX; give it a base category
        # for the composites and replace its task ids with the vectors below.
        placeholder = next(iter(TASK_CATEGORY_INDEX))
        context, target_inputs, targets, _ = super().prepare_inputs(
            {
                **batch,
                "task_category": [
                    placeholder if c in self.category_vectors else c for c in categories
                ],
            }
        )
        rows = []
        for category in categories:
            if category in self.category_vectors:
                rows.append(self.category_vectors[category])
            else:
                rows.append(
                    torch.nn.functional.one_hot(
                        torch.tensor(TASK_CATEGORY_INDEX[category]), NUM_TASKS
                    ).float()
                )
        return context, target_inputs, targets, torch.stack(rows).to(self.device)


def load_run(
    run_dir: str | Path,
    checkpoint: str,
    overrides: list[str] | None = None,
    category_vectors: dict[str, torch.Tensor] | None = None,
):
    """Build a saved run's model from its config.yaml and load `checkpoint` (a file name in
    the run folder, e.g. best_model.ckpt). Returns (cfg, runtime_cfg, model)."""
    config_path = str(Path(run_dir) / "config.yaml")
    cfg = load_config(config_path, overrides or [])
    locate_saved_run(cfg, config_path)
    runtime_cfg = build_runtime_config_dict(cfg)
    if category_vectors is None:
        model = HypernetworkLightning(**runtime_cfg)
    else:
        model = TaskVectorHypernetwork(category_vectors=category_vectors, **runtime_cfg)
    load_checkpoint_state(model, str(Path(run_dir) / checkpoint))
    return cfg, runtime_cfg, model.eval()


def parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    """Trainable and total parameters, as reported in every results table."""
    return {
        "params_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "params_total": sum(p.numel() for p in model.parameters()),
    }
