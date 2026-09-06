"""Verify there is no train/eval content overlap in the built ARC1D datasets.

Marked `slow` (excluded from the default `pytest` run, see pyproject.toml) --
fingerprinting the full ~700K-row train split takes about a minute, and
needs the dataset actually built on disk (`skipif`-guarded, so a fresh
checkout with no `data/` yet just skips rather than failing). Run
deliberately, before a large rerun:

    pytest -m slow tests/test_data_leakage.py

scripts/augment_arc_1d.py already drops augmented train examples that
exactly match a dev/test example at build time
(`filter_held_out_contamination`). This is an independent regression guard
for that property: it re-derives the same exact-content fingerprint
directly from the final, on-disk dataset splits, rather than trusting that
the build-time filter ran (or ran correctly). It also checks the
compositional holdout set (built by a wholly separate generator,
data_modules/arc1d_compositional.py, with disjoint category names) against
train, for the same reason.

A fingerprint is the exact (support_inputs, support_outputs, query_input,
query_output) content of an example, ignoring task_id -- identical to
scripts/augment_arc_1d.py::_example_fingerprint. Two examples with the same
fingerprint are, for training purposes, the same example: a model that sees
one at train time has seen the other's content at eval time too.
"""

import pathlib

import pytest
from datasets import load_from_disk

DATA_DIR = pathlib.Path("data/arc_1d_looped_augmented")
HOLDOUT_DIR = pathlib.Path("data/arc_1d_compositional_holdout")

pytestmark = pytest.mark.slow


def _example_fingerprint(task: dict) -> tuple:
    """Hashable fingerprint of an example's full content (ignores task_id).

    Kept identical to scripts/augment_arc_1d.py::_example_fingerprint --
    two examples must be judged the same way in both places, or this test
    would not actually be checking what the build-time filter checks.
    """
    return (
        tuple(tuple(s) for s in task["support_inputs"]),
        tuple(tuple(s) for s in task["support_outputs"]),
        tuple(task["query_input"]),
        tuple(task["query_output"]),
    )


def _fingerprints(dataset) -> set[tuple]:
    return {_example_fingerprint(task) for task in dataset}


@pytest.fixture(scope="module")
def dataset():
    if not DATA_DIR.is_dir():
        pytest.skip(f"{DATA_DIR} not built -- run scripts/augment_arc_1d.py first")
    return load_from_disk(str(DATA_DIR))


@pytest.fixture(scope="module")
def train_fingerprints(dataset):
    return _fingerprints(dataset["train"])


def test_train_has_no_overlap_with_dev(dataset, train_fingerprints):
    assert not (train_fingerprints & _fingerprints(dataset["dev"]))


def test_train_has_no_overlap_with_test(dataset, train_fingerprints):
    assert not (train_fingerprints & _fingerprints(dataset["test"]))


def test_train_has_no_overlap_with_compositional_holdout(train_fingerprints):
    if not HOLDOUT_DIR.is_dir():
        pytest.skip(f"{HOLDOUT_DIR} not built -- run scripts/build_arc1d_compositional.py first")
    holdout = load_from_disk(str(HOLDOUT_DIR))
    assert not (train_fingerprints & _fingerprints(holdout["holdout_test"]))
