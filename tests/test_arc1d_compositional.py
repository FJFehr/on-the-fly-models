"""Correctness tests for the synthetic compositional ARC-1D task generators.

Stage-2 transforms are verified mathematically against hand-built `ObjectState`s (so the
expected result is derived from the transform's own definition, not hardcoded arrays that
would just duplicate the implementation). Stage-1 constructors and full composite pipelines
are verified structurally/statistically across many seeds, since their exact layout is
randomised.
"""

import random

import pytest

from data_modules.arc1d_compositional import (
    _RetryGeneration,
    BACKGROUND,
    COPY_TEMPLATE_LENGTHS,
    MAX_SEQ_LEN,
    MIN_SEQ_LEN,
    MIRROR_PIVOT_COLOUR,
    MOVE_POINTER_COLOUR,
    RESERVED_MARKER_COLOURS,
    SHIFT_AMOUNT,
    COMPOSITE_TASK_SPECS,
    ObjectState,
    _stage1_denoise_1c,
    _stage1_denoise_mc,
    _stage1_fill,
    _stage1_hollow,
    _stage1_move_dynamic,
    _stage1_shift,
    _stage2_copy,
    _stage2_denoise_1c,
    _stage2_hollow,
    _stage2_mirror,
    _stage2_move_dynamic,
    _stage2_shift,
    generate_dataset,
    generate_pair,
    generate_task,
)

SEQ_LEN = 24


def _active_cells(grid: list[int]) -> dict[int, int]:
    return {i: v for i, v in enumerate(grid) if v != BACKGROUND}


def _first_success(build):
    """Stage-2 functions can reject a random draw (e.g. an out-of-bounds pivot); retry
    across seeds since these tests exercise the transform's correctness, not the retry
    mechanism itself (that's covered separately by generate_pair's own tests)."""
    for seed in range(50):
        try:
            return build(seed)
        except _RetryGeneration:
            continue
    raise AssertionError("could not find a non-colliding random draw in 50 attempts")


# --------------------------------------------------------------------------------------
# Stage 2: golden/mathematical checks against hand-built ObjectStates.
# --------------------------------------------------------------------------------------


def test_stage2_shift_translates_every_active_cell_by_fixed_amount():
    state = ObjectState(
        active=frozenset({5, 6, 7}),
        colour=3,
        used_colours={3},
        used_positions={5, 6, 7},
        input_grid=[BACKGROUND] * SEQ_LEN,
    )
    _, output = _stage2_shift(random.Random(0), SEQ_LEN, state, n=3)
    assert _active_cells(output) == {8: 3, 9: 3, 10: 3}


def test_stage2_shift_preserves_persistent_markers():
    state = ObjectState(
        active=frozenset({5, 6}),
        colour=4,
        used_colours={4, MOVE_POINTER_COLOUR},
        used_positions={5, 6, 20},
        input_grid=[BACKGROUND] * SEQ_LEN,
        persistent_markers={20: MOVE_POINTER_COLOUR},
    )
    _, output = _stage2_shift(random.Random(0), SEQ_LEN, state, n=2)
    assert output[20] == MOVE_POINTER_COLOUR
    assert _active_cells(output) == {7: 4, 8: 4, 20: MOVE_POINTER_COLOUR}


def test_stage2_hollow_keeps_only_the_two_endpoints():
    state = ObjectState(
        active=frozenset(range(5, 10)),
        colour=6,
        used_colours={6},
        used_positions=set(range(5, 10)),
        input_grid=[BACKGROUND] * SEQ_LEN,
    )
    _, output = _stage2_hollow(random.Random(0), SEQ_LEN, state)
    assert _active_cells(output) == {5: 6, 9: 6}


def test_stage2_move_dynamic_moves_object_to_touch_a_placed_pointer():
    active = frozenset({5, 6, 7})
    state = ObjectState(
        active=active,
        colour=4,
        used_colours={4},
        used_positions=set(active),
        input_grid=_render_block(active, 4),
    )
    final_input, final_output = _first_success(
        lambda seed: _stage2_move_dynamic(random.Random(seed), SEQ_LEN, state)
    )

    added = {i for i, v in enumerate(final_input) if v != BACKGROUND} - set(active)
    assert len(added) == 1
    pointer_pos = added.pop()
    assert final_input[pointer_pos] == MOVE_POINTER_COLOUR
    assert pointer_pos > max(active) + 1  # right of the object, with a gap

    shift = (pointer_pos - 1) - max(active)
    expected_moved = {p + shift for p in active}
    output_cells = _active_cells(final_output)
    assert output_cells == {**{p: 4 for p in expected_moved}, pointer_pos: MOVE_POINTER_COLOUR}
    assert max(expected_moved) == pointer_pos - 1  # touching, not overlapping


def test_stage2_mirror_reflects_active_cells_around_the_placed_pivot():
    active = frozenset({10, 11, 12})
    state = ObjectState(
        active=active,
        colour=2,
        used_colours={2},
        used_positions=set(active),
        input_grid=[BACKGROUND] * SEQ_LEN,
    )
    final_input, final_output = _first_success(lambda seed: _stage2_mirror(random.Random(seed), SEQ_LEN, state))

    # The pivot is whatever new non-background cell mirror added to the input.
    added = {i for i, v in enumerate(final_input) if v != BACKGROUND} - set(active)
    assert len(added) == 1
    pivot_pos = added.pop()
    pivot_colour = final_input[pivot_pos]
    # 1d_mirror's pivot is colour 9 in literally every real example -- a hard constant, not an
    # arbitrary "some other colour" (verified: 200/200 raw benchmark, 3000/3000 sampled training rows).
    assert pivot_colour == MIRROR_PIVOT_COLOUR

    expected_reflected = {2 * pivot_pos - p for p in active}
    assert _active_cells(final_output) == {
        **{p: 2 for p in expected_reflected},
        pivot_pos: pivot_colour,
    }


def test_stage2_copy_stamps_the_template_shape_at_the_marker_using_markers_own_colour():
    active = frozenset({10, 11, 12})  # shape_len=3, center_offset=1
    state = ObjectState(
        active=active,
        colour=5,
        used_colours={5},
        used_positions=set(active),
        input_grid=[BACKGROUND] * SEQ_LEN,
    )
    final_input, final_output = _first_success(lambda seed: _stage2_copy(random.Random(seed), SEQ_LEN, state))

    added = {i for i, v in enumerate(final_input) if v != BACKGROUND} - set(active)
    assert len(added) == 1
    marker_pos = added.pop()
    marker_colour = final_input[marker_pos]
    assert marker_colour != 5

    expected_stamp = {marker_pos - 1, marker_pos, marker_pos + 1}
    output_cells = _active_cells(final_output)
    # Original template stays in place, unchanged.
    assert all(output_cells[p] == 5 for p in active)
    # Stamp appears at the marker, in the marker's own colour.
    assert all(output_cells[p] == marker_colour for p in expected_stamp)
    assert set(output_cells) == set(active) | expected_stamp


def test_stage2_denoise_1c_adds_noise_to_input_but_output_stays_clean():
    active = frozenset(range(10, 14))
    state = ObjectState(
        active=active,
        colour=7,
        used_colours={7},
        used_positions=set(active),
        input_grid=_render_block(active, 7),
    )
    final_input, final_output = _first_success(
        lambda seed: _stage2_denoise_1c(random.Random(seed), SEQ_LEN, state)
    )

    noise_cells = _active_cells(final_input).keys() - set(active)
    assert len(noise_cells) in (1, 2)
    assert all(final_input[p] == 7 for p in noise_cells)
    assert _active_cells(final_output) == {p: 7 for p in active}


def _render_block(active: frozenset, colour: int) -> list[int]:
    grid = [BACKGROUND] * SEQ_LEN
    for p in active:
        grid[p] = colour
    return grid


# --------------------------------------------------------------------------------------
# Stage 1: structural properties, checked across many seeds.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(20))
def test_stage1_denoise_1c_block_is_contiguous_and_noise_is_outside_it(seed):
    state = _stage1_denoise_1c(random.Random(seed), SEQ_LEN)
    active = sorted(state.active)
    assert active == list(range(active[0], active[-1] + 1))  # contiguous
    assert all(state.input_grid[p] == state.colour for p in active)

    noise = set(_active_cells(state.input_grid)) - set(active)
    assert len(noise) >= 1
    for p in noise:
        assert state.input_grid[p] == state.colour
        assert p < active[0] - 1 or p > active[-1] + 1  # margin from the block


@pytest.mark.parametrize("seed", range(20))
def test_stage1_denoise_mc_corrupts_interior_pixels_to_other_colours(seed):
    state = _stage1_denoise_mc(random.Random(seed), SEQ_LEN)
    active = sorted(state.active)
    assert active == list(range(active[0], active[-1] + 1))

    corrupted = {p for p in active if state.input_grid[p] != state.colour}
    assert len(corrupted) >= 1
    assert active[0] not in corrupted and active[-1] not in corrupted  # endpoints intact
    for p in corrupted:
        assert state.input_grid[p] != BACKGROUND


@pytest.mark.parametrize("seed", range(20))
def test_stage1_fill_input_shows_only_two_markers_spanning_the_eventual_block(seed):
    state = _stage1_fill(random.Random(seed), SEQ_LEN)
    active = sorted(state.active)
    assert active == list(range(active[0], active[-1] + 1))
    assert len(active) >= 4  # gap >= 2 either side of the two markers
    assert _active_cells(state.input_grid) == {active[0]: state.colour, active[-1]: state.colour}


@pytest.mark.parametrize("seed", range(20))
def test_stage1_hollow_input_is_solid_but_active_is_only_the_endpoints(seed):
    state = _stage1_hollow(random.Random(seed), SEQ_LEN)
    assert state.active == frozenset({min(state.used_positions), max(state.used_positions)})
    solid_span = range(min(state.used_positions), max(state.used_positions) + 1)
    assert _active_cells(state.input_grid) == {p: state.colour for p in solid_span}


@pytest.mark.parametrize("seed", range(20))
def test_stage1_move_dynamic_moves_block_adjacent_to_pointer(seed):
    state = _stage1_move_dynamic(random.Random(seed), SEQ_LEN)
    (pointer_pos, pointer_colour) = next(iter(state.persistent_markers.items()))
    assert pointer_colour == MOVE_POINTER_COLOUR
    assert state.input_grid[pointer_pos] == MOVE_POINTER_COLOUR

    active = sorted(state.active)
    assert active == list(range(active[0], active[-1] + 1))
    # Block ends up touching the pointer on one side.
    assert active[-1] == pointer_pos - 1 or active[0] == pointer_pos + 1


@pytest.mark.parametrize("seed", range(20))
def test_stage1_shift_moves_block_by_fixed_amount(seed):
    state = _stage1_shift(random.Random(seed), SEQ_LEN)
    original = {p - SHIFT_AMOUNT for p in state.active}
    assert _active_cells(state.input_grid) == {p: state.colour for p in original}


# --------------------------------------------------------------------------------------
# Full pipeline / schema validation.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("category", list(COMPOSITE_TASK_SPECS))
@pytest.mark.parametrize("seed", range(20))
def test_reserved_marker_colours_never_appear_as_the_primary_object_colour(category, seed):
    """8 (move pointer) and 9 (mirror pivot) are reserved marker roles -- the block/template
    colour itself must never coincide with them, or a marker would be visually indistinguishable
    from the object it's attached to."""
    rng = random.Random(seed)
    seq_len = rng.randint(MIN_SEQ_LEN, MAX_SEQ_LEN)
    input_seq, output_seq = generate_pair(rng, category, seq_len)
    # The object/template colour is whichever non-reserved colour has the longest run.
    for seq in (input_seq, output_seq):
        runs = []
        i = 0
        while i < len(seq):
            if seq[i] != BACKGROUND:
                j = i
                while j < len(seq) and seq[j] == seq[i]:
                    j += 1
                runs.append((seq[i], j - i))
                i = j
            else:
                i += 1
        non_reserved_runs = [r for r in runs if r[0] not in RESERVED_MARKER_COLOURS]
        if not non_reserved_runs:
            continue
        object_colour = max(non_reserved_runs, key=lambda r: r[1])[0]
        assert object_colour not in RESERVED_MARKER_COLOURS


@pytest.mark.parametrize("category", ["1d_comp_denoisemc_copy", "1d_comp_shift3_copy"])
@pytest.mark.parametrize("seed", range(20))
def test_copy_combos_use_one_of_the_two_allowed_template_lengths_consistently_per_task(category, seed):
    """1d_pcopy_1c/mc always use a length-3, centre-anchored template in the real benchmark, but
    this generator deliberately varies that (3 or 5, per user request) rather than only ever
    using the real distribution's constant -- every copy-paired combo must pick one of
    COPY_TEMPLATE_LENGTHS (not whatever length its upstream stage would otherwise produce), and
    stick to it across all of one task's own support/query pairs."""
    rng = random.Random(seed)
    task = generate_task(rng, category, task_id=0)

    def template_len(seq: list[int]) -> int:
        # The template run is the longest contiguous same-colour run in the output.
        runs = []
        i = 0
        while i < len(seq):
            if seq[i] != BACKGROUND:
                j = i
                while j < len(seq) and seq[j] == seq[i]:
                    j += 1
                runs.append(j - i)
                i = j
            else:
                i += 1
        return max(runs)

    lengths = {template_len(seq) for seq in [*task["support_outputs"], task["query_output"]]}
    assert lengths <= set(COPY_TEMPLATE_LENGTHS)
    assert len(lengths) == 1  # consistent across all of this task's own pairs


@pytest.mark.parametrize("category", list(COMPOSITE_TASK_SPECS))
@pytest.mark.parametrize("seed", range(5))
def test_generate_pair_produces_well_formed_grids(category, seed):
    rng = random.Random(seed)
    seq_len = rng.randint(MIN_SEQ_LEN, MAX_SEQ_LEN)
    input_seq, output_seq = generate_pair(rng, category, seq_len)
    assert len(input_seq) == seq_len
    assert len(output_seq) == seq_len
    assert all(0 <= v <= 9 for v in input_seq)
    assert all(0 <= v <= 9 for v in output_seq)


@pytest.mark.parametrize("category", list(COMPOSITE_TASK_SPECS))
def test_generate_task_has_the_build_arc_1d_schema(category):
    rng = random.Random(42)
    task = generate_task(rng, category, task_id=7)
    assert task["task_category"] == category
    assert task["task_id"] == 7
    assert len(task["support_inputs"]) == 3
    assert len(task["support_outputs"]) == 3
    seq_len = task["sequence_length"]
    lengths = {
        len(task["query_input"]),
        len(task["query_output"]),
        *(len(s) for s in task["support_inputs"]),
        *(len(s) for s in task["support_outputs"]),
    }
    assert lengths == {seq_len}


def test_generate_dataset_produces_n_tasks_per_category():
    tasks = generate_dataset(seed=0, n_per_category=3, categories=["1d_comp_hollow_shift3", "1d_comp_fill_mirror"])
    assert len(tasks) == 6
    by_category: dict[str, int] = {}
    for task in tasks:
        by_category[task["task_category"]] = by_category.get(task["task_category"], 0) + 1
    assert by_category == {"1d_comp_hollow_shift3": 3, "1d_comp_fill_mirror": 3}


def test_generate_dataset_deterministic_given_same_seed():
    first = generate_dataset(seed=5, n_per_category=2)
    second = generate_dataset(seed=5, n_per_category=2)
    assert first == second


def test_generate_dataset_different_seed_gives_different_layout():
    first = generate_dataset(seed=1, n_per_category=2, categories=["1d_comp_denoise1c_shift3"])
    second = generate_dataset(seed=2, n_per_category=2, categories=["1d_comp_denoise1c_shift3"])
    assert first != second
