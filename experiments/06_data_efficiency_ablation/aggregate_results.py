"""Aggregate experiment 6's raw results.txt files into tidy CSVs.

Scans all three arms' outputs/ trees (hypernetwork: 06_data_efficiency_ablation_hypernetwork, joint:
06_data_efficiency_ablation_joint, individual: 06_data_efficiency_ablation_individual), parses each run's
results.txt (a flat "key: value" dump written by train.py) together with its
directory name (which encodes condition/level/category/seed), and writes:

  outputs/results/06_data_efficiency_ablation/results_hypernetwork.csv
  outputs/results/06_data_efficiency_ablation/results_joint.csv
  outputs/results/06_data_efficiency_ablation/results_individual.csv
  outputs/results/06_data_efficiency_ablation/results_combined.csv
      (all three arms aligned on the levels they share: 1, 2, 3, full, and
      (2026-09-15) t1/t3/t5/t10/t20 -- individual's per-category rows
      macro-averaged to one row/level/seed first, since the other two arms
      are already single joint models)

Ignores the 18 old-naming hypernetwork orphans and the 135 (already-deleted)
stale individual dirs from the pre-2026-09-10 scaffolding automatically --
this only recognises the current experiment_name patterns, nothing else.

Usage:
    python experiments/06_data_efficiency_ablation/aggregate_results.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "outputs" / "results" / "06_data_efficiency_ablation"

# Nested/cumulative levels -> a numeric sort key and total-rows-per-category label.
# t{N} (added 2026-09-15) is a genuinely different axis from v{N} -- base task
# *count* at fixed variants_per_base_task=1, not augmentation depth at fixed
# ~40 base tasks -- so its sort keys sit strictly below v1's own (1), and its
# label keeps the "t" prefix rather than being stripped to a bare number: a
# bare "3" would collide with v3's own stripped label while meaning a
# 40x-different row count (t3 = 3 rows/category, v3 = 120).
LEVEL_ORDER = {
    "t1": 0.01, "t3": 0.03, "t5": 0.05, "t10": 0.10, "t20": 0.20,
    "v1": 1, "v2": 2, "v3": 3, "v4": 4, "v5": 5, "v20": 20, "full": 10_000,
}
LEVEL_LABEL = {
    "t1": "t1", "t3": "t3", "t5": "t5", "t10": "t10", "t20": "t20",
    "v1": "1", "v2": "2", "v3": "3", "v4": "4", "v5": "5", "v20": "20", "full": "full",
}

CATEGORIES = [
    "1d_denoising_1c", "1d_denoising_mc", "1d_fill", "1d_flip", "1d_hollow",
    "1d_mirror", "1d_move_1p", "1d_move_2p", "1d_move_2p_dp", "1d_move_3p",
    "1d_move_dp", "1d_pcopy_1c", "1d_pcopy_mc", "1d_scale_dp",
]


def parse_results_txt(path: Path) -> dict[str, float]:
    """Parse a results.txt's flat "key: value" lines into floats (skips non-numeric)."""
    out: dict[str, float] = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        try:
            out[key] = float(value)
        except ValueError:
            continue  # e.g. "checkpoint: none"
    return out


def collect(project_dir: Path, name_re: re.Pattern, level_key: str = "level") -> list[dict]:
    rows = []
    if not project_dir.is_dir():
        return rows
    for run_dir in sorted(project_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        m = name_re.match(run_dir.name)
        if not m:
            continue  # old-naming orphan or unrelated dir -- skip
        results_file = run_dir / "results.txt"
        if not results_file.is_file():
            continue
        metrics = parse_results_txt(results_file)
        row = m.groupdict()
        row[level_key + "_n"] = LEVEL_ORDER[row[level_key]]
        row[level_key] = LEVEL_LABEL[row[level_key]]
        row["seed"] = int(row["seed"])
        row.update(metrics)
        row["run_dir"] = run_dir.name
        rows.append(row)
    return rows


def per_task_long(df: pd.DataFrame, id_cols: list[str]) -> pd.DataFrame:
    """Melt the val_query_exact_match_by_task_<category> columns into a long frame."""
    task_cols = [c for c in df.columns if c.startswith("val_query_exact_match_by_task_")]
    if not task_cols:
        return pd.DataFrame()
    long = df.melt(
        id_vars=id_cols, value_vars=task_cols,
        var_name="category", value_name="val_query_exact_match_by_task",
    )
    long["category"] = long["category"].str.replace(
        "val_query_exact_match_by_task_", "", regex=False
    )
    return long


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- hypernetwork: lowdata_{frozentd,notd}_{level}_seed{n} ---
    hyper_re = re.compile(r"^lowdata_(?P<condition>frozentd|notd)_(?P<level>v\d+|t\d+|full)_seed(?P<seed>\d+)$")
    hyper_rows = collect(REPO_ROOT / "outputs" / "06_data_efficiency_ablation_hypernetwork", hyper_re)
    hyper_df = pd.DataFrame(hyper_rows).sort_values(["condition", "level_n", "seed"])
    hyper_df.to_csv(OUT_DIR / "results_hypernetwork.csv", index=False)

    # --- joint: lowdata_joint_{td,notd}_{level}_seed{n} ---
    joint_re = re.compile(r"^lowdata_joint_(?P<condition>td|notd)_(?P<level>v\d+|t\d+|full)_seed(?P<seed>\d+)$")
    joint_rows = collect(REPO_ROOT / "outputs" / "06_data_efficiency_ablation_joint", joint_re)
    joint_df = pd.DataFrame(joint_rows).sort_values(["condition", "level_n", "seed"])
    joint_df.to_csv(OUT_DIR / "results_joint.csv", index=False)

    # --- individual: lowdata_baseline_{category}_{level}_seed{n} ---
    cat_alt = "|".join(re.escape(c) for c in CATEGORIES)
    indiv_re = re.compile(rf"^lowdata_baseline_(?P<category>{cat_alt})_(?P<level>v\d+|t\d+|full)_seed(?P<seed>\d+)$")
    indiv_rows = collect(REPO_ROOT / "outputs" / "06_data_efficiency_ablation_individual", indiv_re)
    indiv_df = pd.DataFrame(indiv_rows).sort_values(["category", "level_n", "seed"])
    indiv_df.to_csv(OUT_DIR / "results_individual.csv", index=False)

    print(f"hypernetwork: {len(hyper_df)} rows (expect 120)")
    print(f"joint:        {len(joint_df)} rows (expect 120)")
    print(f"individual:   {len(indiv_df)} rows (expect 630)")

    # --- combined: aligned on the levels all three arms share (1, 2, 3, full,
    # plus the new sub-40 t1/t3/t5/t10/t20 family -- the sharpest 3-arm
    # comparison this whole axis exists for) ---
    shared_levels = {"1", "2", "3", "full", "t1", "t3", "t5", "t10", "t20"}
    combined = []

    for _, r in hyper_df[hyper_df["level"].isin(shared_levels)].iterrows():
        combined.append({
            "arm": "hypernetwork", "condition": r["condition"], "level": r["level"],
            "level_n": r["level_n"], "seed": r["seed"],
            "test_query_exact_match": r["test_query_exact_match"],
            "val_query_exact_match": r["val_query_exact_match"],
        })
    for _, r in joint_df[joint_df["level"].isin(shared_levels)].iterrows():
        combined.append({
            "arm": "joint", "condition": r["condition"], "level": r["level"],
            "level_n": r["level_n"], "seed": r["seed"],
            "test_query_exact_match": r["test_query_exact_match"],
            "val_query_exact_match": r["val_query_exact_match"],
        })
    # individual has no condition axis and one row per category -- macro-average
    # across categories first, so it's one row per (level, seed), comparable to
    # the other two arms' single joint model per (condition, level, seed).
    indiv_macro = (
        indiv_df[indiv_df["level"].isin(shared_levels)]
        .groupby(["level", "level_n", "seed"], as_index=False)[
            ["test_query_exact_match", "val_query_exact_match"]
        ]
        .mean()
    )
    for _, r in indiv_macro.iterrows():
        combined.append({
            "arm": "individual", "condition": "n/a", "level": r["level"],
            "level_n": r["level_n"], "seed": r["seed"],
            "test_query_exact_match": r["test_query_exact_match"],
            "val_query_exact_match": r["val_query_exact_match"],
        })

    combined_df = pd.DataFrame(combined).sort_values(["arm", "condition", "level_n", "seed"])
    combined_df.to_csv(OUT_DIR / "results_combined.csv", index=False)
    print(f"combined:     {len(combined_df)} rows (levels 1, 2, 3, full, t1, t3, t5, t10, t20)")

    # --- per-task long frames (hypernetwork/joint only -- individual is already single-task) ---
    hyper_long = per_task_long(hyper_df, ["condition", "level", "level_n", "seed"])
    hyper_long.to_csv(OUT_DIR / "results_hypernetwork_per_task.csv", index=False)
    joint_long = per_task_long(joint_df, ["condition", "level", "level_n", "seed"])
    joint_long.to_csv(OUT_DIR / "results_joint_per_task.csv", index=False)

    print(f"\nWrote CSVs to {OUT_DIR.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
