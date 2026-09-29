"""Generate the synthetic compositional ARC-1D held-out dataset and save it as a HuggingFace
DatasetDict, mirroring scripts/build_arc_1d.py's CLI conventions.

Unlike build_arc_1d.py, there's no external benchmark to download here -- the data comes from
data_modules/arc1d_compositional.py's generators (see that module's docstring, and
experiments/04_compositional_generalization/README.md, for why this generator exists and what
real-benchmark conventions it matches). The output is a single
`holdout_test` split, since this dataset is purely a zero-shot generalisation eval set --
these 10 categories are never trained on.
"""

import argparse
from pathlib import Path

from datasets import Dataset, DatasetDict

from data_modules.arc1d_compositional import COMPOSITE_TASK_SPECS, generate_dataset

OUTPUT_DIR = Path("data/arc_1d_compositional_holdout")
N_PER_CATEGORY = 40
SEED = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--n-per-category", type=int, default=N_PER_CATEGORY)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tasks = generate_dataset(seed=args.seed, n_per_category=args.n_per_category)
    dataset_dict = DatasetDict({"holdout_test": Dataset.from_list(tasks)})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(args.output_dir))

    print(f"Saved {len(tasks)} tasks to {args.output_dir}")
    print(dataset_dict)
    print(f"{len(COMPOSITE_TASK_SPECS)} categories x {args.n_per_category} instances each:")
    for category in COMPOSITE_TASK_SPECS:
        print(f"  {category}")


if __name__ == "__main__":
    main()
