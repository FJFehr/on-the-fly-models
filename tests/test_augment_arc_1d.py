"""Tests for ARC 1D data augmentation primitives.

Each test covers one augmentation function with a hand-crafted minimal task
so the expected behaviour is fully auditable without running the pipeline.
"""

import random
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "augment_arc_1d.py"
MODULE_SPEC = spec_from_file_location("augment_arc_1d", MODULE_PATH)
assert MODULE_SPEC is not None
assert MODULE_SPEC.loader is not None
augment_arc_1d = module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(augment_arc_1d)

apply_color_map = augment_arc_1d.apply_color_map
apply_shift = augment_arc_1d.apply_shift
augment_task = augment_arc_1d.augment_task
generate_color_permutations = augment_arc_1d.generate_color_permutations
get_task_colors = augment_arc_1d.get_task_colors


def make_task(
    support_inputs: list[list[int]],
    support_outputs: list[list[int]],
    query_input: list[int],
    query_output: list[int],
    seq_len: int | None = None,
    task_category: str = "1d_move_1p",
) -> dict:
    if seq_len is None:
        seq_len = len(query_input)
    return {
        "task_category": task_category,
        "task_id": 42,
        "sequence_length": seq_len,
        "support_inputs": support_inputs,
        "support_outputs": support_outputs,
        "query_input": query_input,
        "query_output": query_output,
    }


# ---------------------------------------------------------------------------
# get_task_colors
# ---------------------------------------------------------------------------


def test_get_task_colors_basic():
    """Returns sorted non-zero colors; deduplicates across all sequences."""
    task = make_task(
        [[0, 1, 3], [0, 2, 0]],
        [[1, 0, 0], [0, 3, 0]],
        [0, 1, 0],
        [3, 0, 0],
    )
    assert get_task_colors(task) == [1, 2, 3]


def test_get_task_colors_excludes_background():
    """Color 0 is never included in the returned list."""
    task = make_task([[0, 0]], [[0, 0]], [0], [0])
    assert get_task_colors(task) == []


# ---------------------------------------------------------------------------
# apply_color_map
# ---------------------------------------------------------------------------


def test_apply_color_map_consistent():
    """Same mapping applies identically to all sequences; rule relationship preserved."""
    task = make_task([[1, 0, 2]], [[2, 0, 1]], [1, 2, 0], [2, 1, 0])
    result = apply_color_map(task, {1: 5, 2: 7})
    assert result["support_inputs"] == [[5, 0, 7]]
    assert result["support_outputs"] == [[7, 0, 5]]
    assert result["query_input"] == [5, 7, 0]
    assert result["query_output"] == [7, 5, 0]


def test_apply_color_map_preserves_background():
    """Color 0 is preserved when not in the mapping dict (uses dict.get fallback)."""
    task = make_task([[0, 1]], [[1, 0]], [0, 0], [0, 0])
    result = apply_color_map(task, {1: 3})
    assert result["support_inputs"] == [[0, 3]]
    assert result["support_outputs"] == [[3, 0]]
    assert result["query_input"] == [0, 0]
    assert result["query_output"] == [0, 0]


# ---------------------------------------------------------------------------
# generate_color_permutations
# ---------------------------------------------------------------------------


def test_generate_color_permutations_count():
    """Returns exactly n distinct variants when the colour space is large enough."""
    task = make_task([[1, 2, 3]], [[2, 3, 1]], [3, 0, 0], [1, 0, 0])
    variants = generate_color_permutations(task, 10, random.Random(0))
    assert len(variants) == 10


def test_generate_color_permutations_uniqueness():
    """No two returned variants have the same colour assignment."""
    task = make_task([[1, 2, 3]], [[2, 3, 1]], [3, 0, 0], [1, 0, 0])
    variants = generate_color_permutations(task, 10, random.Random(0))
    fingerprints = [tuple(v["support_inputs"][0]) for v in variants]
    assert len(fingerprints) == len(set(fingerprints))


def test_generate_color_permutations_excludes_original_mapping():
    """Additional colour permutations should not include the unchanged task."""
    task = make_task([[1, 2, 3]], [[2, 3, 1]], [3, 0, 0], [1, 0, 0])
    variants = generate_color_permutations(task, 20, random.Random(0))
    assert all(v["support_inputs"] != task["support_inputs"] for v in variants)


def test_generate_color_permutations_fewer_when_space_exhausted():
    """Returns fewer than n when only one non-zero colour is present (9 max mappings)."""
    # Task with only colour 1; 8 non-identity injective maps to {1..9}
    task = make_task([[1, 0, 1]], [[0, 1, 0]], [1, 0], [0, 1])
    variants = generate_color_permutations(task, 20, random.Random(0))
    assert len(variants) <= 8


# ---------------------------------------------------------------------------
# apply_shift
# ---------------------------------------------------------------------------


def test_apply_shift_right_prepends_zeros():
    """Positive shift prepends zeros to all sequences; no content is discarded."""
    task = make_task([[1, 2, 3]], [[0, 1, 2]], [3, 0, 0], [0, 3, 0], seq_len=3)
    result = apply_shift(task, 2)
    assert result["support_inputs"] == [[0, 0, 1, 2, 3]]
    assert result["support_outputs"] == [[0, 0, 0, 1, 2]]
    assert result["query_input"] == [0, 0, 3, 0, 0]
    assert result["query_output"] == [0, 0, 0, 3, 0]


def test_apply_shift_left_appends_zeros():
    """Negative shift appends zeros to all sequences; no content is discarded."""
    task = make_task([[1, 2, 3]], [[2, 3, 0]], [1, 2, 0], [2, 0, 0], seq_len=3)
    result = apply_shift(task, -2)
    assert result["support_inputs"] == [[1, 2, 3, 0, 0]]
    assert result["support_outputs"] == [[2, 3, 0, 0, 0]]
    assert result["query_input"] == [1, 2, 0, 0, 0]
    assert result["query_output"] == [2, 0, 0, 0, 0]


def test_apply_shift_zero_is_identity():
    """k=0 leaves all sequences and sequence_length unchanged."""
    task = make_task([[1, 2, 3]], [[3, 2, 1]], [2, 0, 0], [0, 2, 0], seq_len=3)
    assert apply_shift(task, 0) == task


def test_apply_shift_extends_sequence_length():
    """sequence_length increases by abs(shift) for both directions."""
    task = make_task([[1, 0]], [[0, 1]], [1, 0], [0, 1], seq_len=2)
    for shift in [1, 2, -1, -2]:
        result = apply_shift(task, shift)
        assert result["sequence_length"] == 2 + abs(shift), f"shift={shift}"


# ---------------------------------------------------------------------------
# augment_task (full pipeline)
# ---------------------------------------------------------------------------


def test_augment_task_count():
    """Full pipeline returns exactly (1+n_perms) × (1+len(shifts)) × n_mirror variants."""
    task = make_task([[1, 2]], [[2, 1]], [1, 0], [2, 0], seq_len=2)
    n_perms = 5
    shifts = [1, -1]
    variants_no_mirror = augment_task(task, n_perms, shifts, rng=random.Random(0), mirror=False)
    expected_no_mirror = (1 + n_perms) * (1 + len(shifts))
    assert len(variants_no_mirror) == expected_no_mirror

    variants_mirror = augment_task(task, n_perms, shifts, rng=random.Random(0), mirror=True)
    assert len(variants_mirror) == expected_no_mirror * 2


def test_augment_task_ids_unique():
    """All returned task_id values are distinct."""
    task = make_task([[1, 2]], [[2, 1]], [1, 0], [2, 0], seq_len=2)
    variants = augment_task(task, 5, [1, -1], rng=random.Random(0))
    ids = [v["task_id"] for v in variants]
    assert len(ids) == len(set(ids))


def test_augment_task_ids_traceable():
    """task_ids encode the original: original_task_id * 10000 + aug_index."""
    task = make_task([[1, 2]], [[2, 1]], [1, 0], [2, 0], seq_len=2)
    orig_id = task["task_id"]
    variants = augment_task(task, 5, [1, -1], rng=random.Random(0))
    for i, v in enumerate(variants):
        assert v["task_id"] == orig_id * 10000 + i


# ---------------------------------------------------------------------------
# fixed_colour (1d_mirror pivot colour 9)
# ---------------------------------------------------------------------------


def _all_values(task: dict) -> list[int]:
    """Flat list of all token values across every sequence in the task."""
    seqs = (
        task["support_inputs"]
        + task["support_outputs"]
        + [task["query_input"], task["query_output"]]
    )
    return [v for seq in seqs for v in seq]


def test_generate_color_permutations_fixed_colour_stays_9():
    """With fixed_colour=9, every 9 in the original task remains 9 in all variants."""
    task = make_task([[1, 9, 2]], [[9, 2, 1]], [9, 1, 0], [9, 2, 0])
    variants = generate_color_permutations(task, 10, random.Random(0), fixed_colour=9)
    for v in variants:
        assert v["support_inputs"][0][1] == 9
        assert v["support_outputs"][0][0] == 9
        assert v["query_input"][0] == 9
        assert v["query_output"][0] == 9


def test_generate_color_permutations_fixed_colour_not_a_target():
    """With fixed_colour=9, no other colour is remapped to 9."""
    task = make_task([[1, 2, 3]], [[2, 3, 1]], [3, 0, 0], [1, 0, 0])
    variants = generate_color_permutations(task, 20, random.Random(0), fixed_colour=9)
    for v in variants:
        assert 9 not in _all_values(v), "a colour was incorrectly remapped to 9"


def test_augment_task_mirror_category_preserves_colour_9():
    """For 1d_mirror tasks, colour 9 appears (unchanged) in every augmented variant."""
    task = make_task(
        [[1, 9, 0], [2, 9, 0]],
        [[0, 9, 1], [0, 9, 2]],
        [3, 9, 0],
        [0, 9, 3],
        task_category="1d_mirror",
    )
    variants = augment_task(task, 10, [1], rng=random.Random(0))
    for v in variants:
        assert 9 in set(_all_values(v)), "colour 9 disappeared from a 1d_mirror variant"


# ---------------------------------------------------------------------------
# 1d_recolor_cnt: global augmentation preserves run-length → colour mapping
# ---------------------------------------------------------------------------


def _extract_recolor_cnt_mapping(task: dict) -> dict[int, int] | None:
    """Extract the (run_size → output_colour) mapping from a recolor_cnt-style task.

    Returns None if the mapping is inconsistent across any pair or the query.
    Each contiguous run of non-zero input values must map to a single output colour,
    and that colour must be the same for runs of the same size across all sequences.
    """
    mapping: dict[int, int] = {}
    all_pairs = list(zip(task["support_inputs"], task["support_outputs"])) + [
        (task["query_input"], task["query_output"])
    ]
    for inp, out in all_pairs:
        i = 0
        while i < len(inp):
            if inp[i] == 0:
                i += 1
                continue
            run_start = i
            run_colour = inp[i]
            while i < len(inp) and inp[i] == run_colour:
                i += 1
            run_len = i - run_start
            out_colours = set(out[run_start:i])
            if len(out_colours) != 1:
                return None
            out_colour = out_colours.pop()
            if run_len in mapping and mapping[run_len] != out_colour:
                return None
            mapping[run_len] = out_colour
    return mapping


def test_recolor_cnt_global_augmentation_preserves_mapping():
    """Global augmentation of a recolor_cnt task keeps the run-size → colour mapping
    globally consistent across all support pairs and the query in every variant."""
    # Rule: size 1 → colour 5, size 2 → colour 7, size 3 → colour 6
    task = make_task(
        support_inputs=[[4, 0, 4, 4, 0, 4, 4, 4], [4, 4, 4, 0, 4, 0, 4, 4], [4, 4, 0, 4, 4, 4, 0, 4]],
        support_outputs=[[5, 0, 7, 7, 0, 6, 6, 6], [6, 6, 6, 0, 5, 0, 7, 7], [7, 7, 0, 6, 6, 6, 0, 5]],
        query_input=[4, 0, 4, 4, 4, 0, 4, 4],
        query_output=[5, 0, 6, 6, 6, 0, 7, 7],
        task_category="1d_recolor_cnt",
    )
    variants = augment_task(task, 20, [1, -1], rng=random.Random(0), mirror=False, per_pair=False)
    for i, v in enumerate(variants):
        mapping = _extract_recolor_cnt_mapping(v)
        assert mapping is not None, f"variant {i} has inconsistent run-size → colour mapping"
        assert len(mapping) == 3, f"variant {i} lost run-size entries: {mapping}"


def test_recolor_cnt_augmentation_variant_count():
    """With n=199 colour perms, 4 shifts and no mirror, augment_task yields exactly 1000 variants
    — matching the arc_1d_all_tasks_augmented build parameters (200 colour × 5 shifts)."""
    task = make_task(
        support_inputs=[[4, 0, 4, 4, 0, 4, 4, 4], [4, 4, 4, 0, 4, 0, 4, 4], [4, 4, 0, 4, 4, 4, 0, 4]],
        support_outputs=[[5, 0, 7, 7, 0, 6, 6, 6], [6, 6, 6, 0, 5, 0, 7, 7], [7, 7, 0, 6, 6, 6, 0, 5]],
        query_input=[4, 0, 4, 4, 4, 0, 4, 4],
        query_output=[5, 0, 6, 6, 6, 0, 7, 7],
        task_category="1d_recolor_cnt",
    )
    variants = augment_task(
        task,
        n_color_perms=199,
        shifts=[1, 2, -1, -2],
        rng=random.Random(0),
        mirror=False,
        per_pair=False,
    )
    assert len(variants) == 1000  # 200 colour × 5 shifts × 1 (no mirror)
