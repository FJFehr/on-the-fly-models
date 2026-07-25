"""Shared utilities for ARC1D data modules."""

import hashlib
import json
import os

FILTER_CACHE_DIR = "data/.filter_index_cache"


def _filter_cache_path(split, task_categories, task_ids) -> str | None:
    """Return a stable cache path for this (dataset file, filters) pair, or None if uncacheable."""
    cache_files = getattr(split, "cache_files", None)
    if not cache_files:
        return None
    payload = {
        "cache_files": sorted(f["filename"] for f in cache_files),
        "task_categories": sorted(task_categories) if task_categories is not None else None,
        "task_ids": sorted(task_ids) if task_ids is not None else None,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return os.path.join(FILTER_CACHE_DIR, f"{digest}.json")


def filter_split(split, task_categories: list[str] | None, task_ids: list[int] | None):
    """Return the subset of split matching task_categories/task_ids.

    Scanning a large ARC1D split (hundreds of thousands of rows) row-by-row in
    Python is expensive, and every caller (every training job, plus a second
    setup() pass for hard-example export within the same job) re-pays that
    cost from scratch. The matching row indices are cached to disk keyed by
    the underlying arrow file paths + filter params, so repeat calls -
    whether later in the same process or from a different process entirely -
    load the index list directly instead of rescanning.
    """
    allowed_categories = set(task_categories) if task_categories is not None else None
    allowed_task_ids = set(task_ids) if task_ids is not None else None

    if allowed_categories is None and allowed_task_ids is None:
        return split

    cache_path = _filter_cache_path(split, task_categories, task_ids)
    if cache_path is not None and os.path.exists(cache_path):
        with open(cache_path) as f:
            indices = json.load(f)
        return split.select(indices)

    indices = [
        i
        for i, task in enumerate(split)
        if (allowed_categories is None or task["task_category"] in allowed_categories)
        and (allowed_task_ids is None or task["task_id"] in allowed_task_ids)
    ]

    if cache_path is not None:
        os.makedirs(FILTER_CACHE_DIR, exist_ok=True)
        tmp_path = f"{cache_path}.tmp{os.getpid()}"
        with open(tmp_path, "w") as f:
            json.dump(indices, f)
        os.replace(tmp_path, cache_path)  # atomic, safe if racing with other processes

    return split.select(indices)
