"""Plot the loop-skip/no-skip diagnostics (T4-T7, L1-L6) from arc1d_uniform_ablation.

Not part of the T1-T7 story plot (see plot_uniform_ablation.py) -- this covers the
side-diagnostics that ask two separate questions:

  1. "Just loop more, no skip mechanisms at all" -- how far does looping alone
     scale? (T4 n_loops=1 -> T5 n=4 -> L1 n=8 -> L5 n=16 -> L6 n=32, all no-skip)
  2. Does per-loop h0 injection ("loop skip") need more iterations to pay off?
     (L2 n=4 loop-skip-only, L3 n=8 loop-skip-only, vs T7/L4 block+loop skip)

Usage:
    python scripts/plot_loop_diagnostics.py [outputs/arc1d_uniform_ablation]
"""

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

# code -> (n_loops, use_block_skip, use_loop_skip), from gen_uniform_ablation_configs.py
CODE_CONFIG = {
    "T4": (1, False, False),
    "T5": (4, False, False),
    "T6": (4, True, False),
    "T7": (4, True, True),
    "L1": (8, False, False),
    "L2": (4, False, True),
    "L3": (8, False, True),
    "L4": (8, True, True),
    "L5": (16, False, False),
    "L6": (32, False, False),
}

NO_SKIP_SWEEP = ["T4", "T5", "L1", "L5", "L6"]  # n_loops = 1, 4, 8, 16, 32

SUFFIXES = {
    "T4": "rope_flat",
    "T5": "looped",
    "T6": "block_skip",
    "T7": "loop_skip",
    "L1": "n8_base",
    "L2": "n4_loop_skip_only",
    "L3": "n8_loop_skip_only",
    "L4": "n8_block_loop_skip",
    "L5": "n16_base",
    "L6": "n32_base",
}

VAL_MARKER = "val_query_exact_match: "
TEST_MARKER = "test_query_exact_match: "
SKIP_TASKS = {"1d_padded_fill"}


def load_results(results_dir: Path) -> dict:
    """code -> {"val": [...], "test": [...]} pooled across all tasks and seeds."""
    data: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    if not results_dir.exists():
        return data
    for exp_dir in sorted(results_dir.iterdir()):
        rf = exp_dir / "results.txt"
        if not rf.exists():
            continue
        name = exp_dir.name
        if not name.startswith("ablation_"):
            continue
        rest = name[len("ablation_"):]
        code = rest.split("_", 1)[0]
        if code not in CODE_CONFIG:
            continue
        if f"dim16_" not in rest and f"dim32_" not in rest:
            continue
        dim = 16 if "dim16_" in rest else 32
        if any(f"_{t}_" in f"_{rest}_" for t in SKIP_TASKS):
            continue

        text = rf.read_text()
        vm = next((l for l in text.splitlines() if l.startswith(VAL_MARKER)), None)
        tm = next((l for l in text.splitlines() if l.startswith(TEST_MARKER)), None)
        if vm is None or tm is None:
            continue
        data[dim][code]["val"].append(float(vm[len(VAL_MARKER):]))
        data[dim][code]["test"].append(float(tm[len(TEST_MARKER):]))
    return data


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    n = len(values)
    mu = sum(values) / n
    var = sum((v - mu) ** 2 for v in values) / n
    return mu, var**0.5


def print_table(data: dict) -> None:
    print(f"{'code':<4} {'n_loops':>7} {'block':>6} {'loop':>5} "
          f"{'dim':>4} {'n':>5} {'val':>14} {'test':>14}")
    for code in list(CODE_CONFIG):
        n_loops, block_skip, loop_skip = CODE_CONFIG[code]
        for dim in sorted(data):
            vals = data[dim].get(code)
            if not vals or not vals["val"]:
                continue
            n = len(vals["val"])
            vmu, vsd = mean_std(vals["val"])
            tmu, tsd = mean_std(vals["test"])
            print(
                f"{code:<4} {n_loops:>7} {str(block_skip):>6} {str(loop_skip):>5} "
                f"{dim:>4} {n:>5} {vmu:>6.1%} +/- {vsd:>4.1%}  {tmu:>6.1%} +/- {tsd:>4.1%}"
            )


def plot_no_skip_sweep(data: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, metric, title in zip(axes, ("val", "test"), ("Validation", "Test")):
        for dim, marker in ((16, "o"), (32, "s")):
            xs, ys, yerrs = [], [], []
            for code in NO_SKIP_SWEEP:
                vals = data.get(dim, {}).get(code, {}).get(metric, [])
                if not vals:
                    continue
                n_loops = CODE_CONFIG[code][0]
                mu, sd = mean_std(vals)
                xs.append(n_loops)
                ys.append(mu)
                yerrs.append(sd)
            if xs:
                ax.errorbar(xs, ys, yerr=yerrs, marker=marker, capsize=3, label=f"dim={dim}")
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 4, 8, 16, 32])
        ax.set_xticklabels(["1", "4", "8", "16", "32"])
        ax.set_xlabel("n_loops (no skip)")
        ax.set_ylim(0, 1)
        ax.set_title(f"{title} Exact Match")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Exact Match (mean +/- std across tasks x seeds)")
    axes[0].legend()
    fig.suptitle("Loop-only scaling (no skip mechanisms): T4 -> T5 -> L1 -> L5 -> L6")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def main() -> None:
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/arc1d_uniform_ablation")
    data = load_results(results_dir)

    print_table(data)
    print()

    out_dir = Path("outputs/uniform_ablation")
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_no_skip_sweep(data, out_dir / "loop_diagnostics_no_skip_sweep.png")


if __name__ == "__main__":
    main()
