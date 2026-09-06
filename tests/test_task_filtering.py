"""Contract tests for filter_split, the task-category/id filter both live datamodules share.

Its cache directory (FILTER_CACHE_DIR) is a hardcoded module-level constant, not a
parameter -- every test here monkeypatches it to a tmp_path so nothing writes into the
repo's real data/.filter_index_cache.
"""

from datasets import Dataset

from data_modules import task_filtering
from data_modules.task_filtering import filter_split


def make_split(tmp_path, n_categories: int = 2, n_per_category: int = 3):
    """A small on-disk (not purely in-memory) dataset -- filter_split's caching only
    activates when split.cache_files is populated, which requires a real save/load
    round trip, not Dataset.from_list() alone."""
    rows = [
        {"task_category": f"cat{c}", "task_id": c * 100 + i}
        for c in range(n_categories)
        for i in range(n_per_category)
    ]
    dataset = Dataset.from_list(rows)
    dataset_path = tmp_path / "dataset"
    dataset.save_to_disk(str(dataset_path))
    return Dataset.load_from_disk(str(dataset_path))


def test_filter_split_returns_matching_subset_by_category_and_id(tmp_path, monkeypatch):
    monkeypatch.setattr(task_filtering, "FILTER_CACHE_DIR", str(tmp_path / "cache"))
    split = make_split(tmp_path)

    filtered = filter_split(split, task_categories=["cat0"], task_ids=None)

    assert len(filtered) == 3
    assert all(row["task_category"] == "cat0" for row in filtered)


def test_filter_split_returns_the_same_object_when_no_filters_given(tmp_path, monkeypatch):
    """Both filters None is the documented fast-path: no scan, no cache write, the input
    split itself comes back unchanged (identity, not just an equal copy)."""
    monkeypatch.setattr(task_filtering, "FILTER_CACHE_DIR", str(tmp_path / "cache"))
    split = make_split(tmp_path)

    result = filter_split(split, task_categories=None, task_ids=None)

    assert result is split
    assert not (tmp_path / "cache").exists()


def test_filter_split_reuses_the_cached_index_list_on_a_second_call(tmp_path, monkeypatch):
    """The second call with identical filter params must read the cache file the first
    call wrote, rather than rescanning -- proven by poisoning the cache file's contents
    on disk and observing the (wrong, poisoned) result come back unchanged."""
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(task_filtering, "FILTER_CACHE_DIR", str(cache_dir))
    split = make_split(tmp_path)

    filter_split(split, task_categories=["cat1"], task_ids=None)
    cache_files = list(cache_dir.glob("*.json"))
    assert len(cache_files) == 1

    # Poison the cache: if the second call rescanned instead of reading this file, the
    # poisoned (empty) result below would never appear.
    cache_files[0].write_text("[]")
    poisoned = filter_split(split, task_categories=["cat1"], task_ids=None)

    assert len(poisoned) == 0
