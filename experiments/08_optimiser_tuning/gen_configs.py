"""Generate experiment 8's leaf configs: the Muon grid (arm A, 60 configs) and the plain-AdamW
grid (arm B, 15 configs). Seeds come from run.sh (SEEDS_OVERRIDE), so none is written here.

Every leaf inherits configs/base.yaml (experiment 2's dim-4 notd hypernetwork, eta_min 0) and
sets only the optimiser settings being tuned. Rerunning reproduces the committed files exactly.

Usage:
    uv run python experiments/08_optimiser_tuning/gen_configs.py
"""

from pathlib import Path

BASE_CFG = "experiments/08_optimiser_tuning/configs/base.yaml"
CONFIG_DIR = Path("experiments/08_optimiser_tuning/configs")

# Values are written as strings so file names and YAML stay exactly as listed.
MUON_LRS = ["0.002", "0.005", "0.01", "0.02", "0.04"]
MUON_ADAM_LRS = ["1e-4", "3e-4", "5e-4", "1e-3"]  # the Adam group inside Muon
ADAMW_LRS = ["1e-4", "3e-4", "1e-3", "3e-3", "1e-2"]
WEIGHT_DECAYS = ["0.0", "0.01", "0.1"]


def muon_name(muon_lr: str, adam_lr: str, wd: str) -> str:
    return f"muon_m{muon_lr}_a{adam_lr}_wd{wd}"


def adamw_name(lr: str, wd: str) -> str:
    return f"adamw_lr{lr}_wd{wd}"


def write(path: Path, name: str, settings: dict[str, str]) -> None:
    lines = [f"_base_: {BASE_CFG}", f"experiment_name: {name}"]
    lines += [f"{key}: {value}" for key, value in settings.items()]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    n_written = 0
    muon_dir = CONFIG_DIR / "muon"
    muon_dir.mkdir(parents=True, exist_ok=True)
    for muon_lr in MUON_LRS:
        for adam_lr in MUON_ADAM_LRS:
            for wd in WEIGHT_DECAYS:
                name = muon_name(muon_lr, adam_lr, wd)
                settings = {
                    "optimizer": "Muon",
                    "muon_lr": muon_lr,
                    "learning_rate": adam_lr,
                    "weight_decay": wd,
                }
                write(muon_dir / f"{name}.yaml", name, settings)
                n_written += 1

    adamw_dir = CONFIG_DIR / "adamw"
    adamw_dir.mkdir(parents=True, exist_ok=True)
    for lr in ADAMW_LRS:
        for wd in WEIGHT_DECAYS:
            name = adamw_name(lr, wd)
            settings = {"optimizer": "AdamW", "learning_rate": lr, "weight_decay": wd}
            write(adamw_dir / f"{name}.yaml", name, settings)
            n_written += 1

    print(f"Wrote {n_written} leaf configs under {CONFIG_DIR}")


if __name__ == "__main__":
    main()
