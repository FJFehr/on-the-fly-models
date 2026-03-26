"""Download the 1D-ARC dataset and convert it to a HuggingFace Dataset."""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from datasets import Dataset

REPO_URL = "https://github.com/khalil-research/1D-ARC.git"
OUTPUT_DIR = Path("data/arc_1d")


def main():
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo_dir = Path(tmp_dir) / "1D-ARC"
        print(f"Cloning {REPO_URL} ...")
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, str(repo_dir)],
            check=True,
        )

        dataset_dir = repo_dir / "dataset"
        rows = []

        for task_dir in sorted(dataset_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            task_category = task_dir.name

            for json_file in sorted(task_dir.glob("*.json")):
                # Extract task_id from filename like "1d_move_1p_42.json"
                task_id = int(json_file.stem.rsplit("_", 1)[-1])
                data = json.loads(json_file.read_text())

                for split in ("train", "test"):
                    for example_idx, example in enumerate(data.get(split, [])):
                        rows.append(
                            {
                                "task_category": task_category,
                                "task_id": task_id,
                                "split": split,
                                "example_idx": example_idx,
                                "input": example["input"][0],
                                "output": example["output"][0],
                            }
                        )

    print(
        f"Collected {len(rows)} examples across {len(set(r['task_category'] for r in rows))} task categories"
    )

    ds = Dataset.from_list(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(OUTPUT_DIR))
    print(f"Saved dataset to {OUTPUT_DIR}")
    print(ds)


if __name__ == "__main__":
    main()
