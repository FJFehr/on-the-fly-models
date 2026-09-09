"""Per-composition breakdown for the paper's compositional-generalization table.

Reads results_holdout.csv (written by plot_compositional.py --outputs-dir, itself sourced
from run.sh's eval output) and writes results_holdout_paper_table.csv: one row per composite
category plus an "Overall" row, matching the paper table's exact aggregation convention --
token accuracy macro-averaged over categories and seeds (every cell has equal n=40, so the
per-category-per-seed mean, then simple mean across the 5 seeds, then simple mean across the
10 categories, is exactly the doubly-macro-averaged number); exact match pooled over all 2,000
held-out instances (same equal-n property makes this the plain mean of the 50 per-category-
per-seed cells too, not a separately weighted pool).

Usage
-----
    uv run python experiments/04_compositional_generalization/report_holdout_breakdown.py
"""

import csv
import statistics
from collections import defaultdict
from pathlib import Path

SRC = Path("outputs/results/04_compositional_generalization/results_holdout.csv")
OUT = Path("outputs/results/04_compositional_generalization/results_holdout_paper_table.csv")

# category token -> human label, for building "$B \circ A$" (apply A, then B) from a
# "<A>_<B>" category name -- matches data_modules/arc1d_compositional.py's naming and the
# paper's own composition order.
COMPONENT_LABEL = {
    "denoise1c": "Denoise 1C", "denoisemc": "Denoise MC", "copy": "Copy",
    "mirror": "Mirror", "fill": "Fill", "movedynamic": "Move Dynamic",
    "shift3": "Shift 3", "hollow": "Hollow",
}
# Row order matches the paper table.
ORDER = [
    "1d_comp_denoisemc_mirror", "1d_comp_denoisemc_copy", "1d_comp_denoisemc_denoise1c",
    "1d_comp_fill_mirror", "1d_comp_denoise1c_shift3", "1d_comp_hollow_shift3",
    "1d_comp_movedynamic_hollow", "1d_comp_fill_shift3", "1d_comp_fill_movedynamic",
    "1d_comp_shift3_copy",
]


def display(category: str) -> str:
    stem = category[len("1d_comp_") :]
    first, second = stem.split("_", 1)
    return f"{COMPONENT_LABEL[second]} o {COMPONENT_LABEL[first]}"


def main() -> None:
    rows = defaultdict(lambda: defaultdict(list))
    with open(SRC, newline="") as f:
        for r in csv.DictReader(f):
            rows[r["category"]][r["condition"]].append(
                (float(r["seq_accuracy"]), float(r["exact_match"]))
            )

    per_category = {}
    for cat in ORDER:
        td_seq = statistics.mean(v[0] for v in rows[cat]["frozen_td"]) * 100
        td_em = statistics.mean(v[1] for v in rows[cat]["frozen_td"]) * 100
        notd_seq = statistics.mean(v[0] for v in rows[cat]["notd"]) * 100
        notd_em = statistics.mean(v[1] for v in rows[cat]["notd"]) * 100
        per_category[cat] = (td_seq, td_em, notd_seq, notd_em)

    out_rows = [
        {
            "composition": display(cat),
            "category": cat,
            "task_id_token_acc": round(td_seq, 2),
            "task_id_exact_match": round(td_em, 2),
            "notd_token_acc": round(notd_seq, 2),
            "notd_exact_match": round(notd_em, 2),
            "token_acc_diff_notd_minus_taskid": round(notd_seq - td_seq, 2),
        }
        for cat, (td_seq, td_em, notd_seq, notd_em) in per_category.items()
    ]

    td_seq_all = statistics.mean(v[0] for v in per_category.values())
    td_em_all = statistics.mean(v[1] for v in per_category.values())
    notd_seq_all = statistics.mean(v[2] for v in per_category.values())
    notd_em_all = statistics.mean(v[3] for v in per_category.values())
    out_rows.append(
        {
            "composition": "Overall",
            "category": "overall",
            "task_id_token_acc": round(td_seq_all, 2),
            "task_id_exact_match": round(td_em_all, 2),
            "notd_token_acc": round(notd_seq_all, 2),
            "notd_exact_match": round(notd_em_all, 2),
            "token_acc_diff_notd_minus_taskid": round(notd_seq_all - td_seq_all, 2),
        }
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote {len(out_rows)} rows to {OUT}")

    cols = ("composition", "td_tok", "td_em", "notd_tok", "notd_em", "diff")
    widths = (30, 7, 6, 9, 8, 7)
    print("\n" + " ".join(f"{c:>{w}s}" for c, w in zip(cols, widths, strict=True)))
    for row in out_rows:
        print(
            f"{row['composition']:30s} {row['task_id_token_acc']:6.1f}% "
            f"{row['task_id_exact_match']:5.2f}% {row['notd_token_acc']:8.1f}% "
            f"{row['notd_exact_match']:7.2f}% {row['token_acc_diff_notd_minus_taskid']:+6.1f}"
        )


if __name__ == "__main__":
    main()
