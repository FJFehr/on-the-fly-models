"""Synthetic generators for compositional ARC-1D held-out tasks.

ARC-1D tasks aren't procedurally generated anywhere else in this repo (`scripts/build_arc_1d.py`
only ingests static JSON from the external 1D-ARC benchmark), and that benchmark has no notion
of chaining two rules together. This module builds new synthetic instances that do: each
composite category applies a "stage 1" rule (constructs a fresh random object, plus whatever
corruption/markers its own single-task rule needs) followed by a "stage 2" rule (further
transforms, corrupts, hollows, or duplicates that object) to produce the final input/output
pair. See experiments/04_compositional_generalization/README.md for the combo-selection
rationale (why these 10, why this order, why others were excluded).

Every task record uses the same schema as `scripts/build_arc_1d.py`'s output: task_category,
task_id, sequence_length, support_inputs/support_outputs (3), query_input/query_output (1).
"""

import random
from dataclasses import dataclass, field

MAX_SEQ_LEN = 33
MIN_SEQ_LEN = 18
BACKGROUND = 0
# 1d_mirror's pivot is colour 9 in literally every real example (200/200 in the raw benchmark,
# 3000/3000 sampled rows of the actual training set) -- not just consistent per task, a hard
# global constant the augmentation pipeline deliberately never remaps (FIXED_RULE_COLOURS in
# scripts/augment_arc_1d.py). move_dp/move_2p_dp's pointer colour is only consistent *within*
# one task (it varies across different task instances in real data); this generator keeps it a
# simple global constant too (matching the "current smaller [scope] is fine" simplification
# already agreed for block-length fidelity) -- kept distinct from MIRROR_PIVOT_COLOUR on
# principle, in case a future combo ever pairs move_dynamic with mirror in one grid.
MIRROR_PIVOT_COLOUR = 9
MOVE_POINTER_COLOUR = 8
RESERVED_MARKER_COLOURS = frozenset({MIRROR_PIVOT_COLOUR, MOVE_POINTER_COLOUR})
SHIFT_AMOUNT = 3  # "shift 3" per the user's worked example; reused for every "+shift" combo
# 1d_pcopy_1c/mc always use a length-3 template in the real benchmark (200/200 examples each),
# with the marker always exactly centered (offset 1, checked on 289 real pcopy_mc examples).
# Per user request, this generator deliberately varies that (3 or 5, chosen once per task
# instance so all its support/query pairs stay consistent) rather than only ever using the
# real distribution's constant 3 -- both are odd, so the centre-anchored marker convention
# (`center_offset = shape_len // 2`) still holds either way.
COPY_TEMPLATE_LENGTHS = (3, 5)


class _RetryGeneration(Exception):
    """Raised internally when a random draw produces an out-of-bounds or colliding layout."""


@dataclass
class ObjectState:
    """The object as stage 1's own rule would output it, plus bookkeeping for stage 2."""

    active: frozenset[int]
    colour: int
    used_colours: set[int]
    used_positions: set[int]
    input_grid: list[int]
    persistent_markers: dict[int, int] = field(default_factory=dict)


def _render(seq_len: int, cells: dict[int, int]) -> list[int]:
    grid = [BACKGROUND] * seq_len
    for pos, colour in cells.items():
        grid[pos] = colour
    return grid


def _pick_colour(rng: random.Random, exclude: set[int]) -> int:
    choices = [c for c in range(1, 10) if c not in exclude]
    return rng.choice(choices)


def _require(condition: bool) -> None:
    if not condition:
        raise _RetryGeneration


# --------------------------------------------------------------------------------------
# Stage 1: construct a fresh object from scratch, reporting what the rule's OWN single-task
# output would be (this is what stage 2 receives as its input object).
# --------------------------------------------------------------------------------------


def _stage1_denoise_1c(rng: random.Random, seq_len: int) -> ObjectState:
    colour = _pick_colour(rng, RESERVED_MARKER_COLOURS)
    block_len = rng.randint(3, max(3, seq_len // 3))
    start = rng.randint(2, seq_len - block_len - 3)
    end = start + block_len - 1
    active = set(range(start, end + 1))

    n_noise = rng.randint(1, 2)
    noise_positions: set[int] = set()
    for _ in range(50):
        if len(noise_positions) == n_noise:
            break
        pos = rng.randint(0, seq_len - 1)
        if pos in active or abs(pos - start) <= 1 or abs(pos - end) <= 1:
            continue
        if any(abs(pos - p) <= 1 for p in noise_positions):
            continue
        noise_positions.add(pos)
    _require(len(noise_positions) == n_noise)

    cells = {p: colour for p in active}
    cells.update({p: colour for p in noise_positions})
    return ObjectState(
        active=frozenset(active),
        colour=colour,
        used_colours={colour},
        used_positions=set(active) | noise_positions,
        input_grid=_render(seq_len, cells),
    )


def _stage1_denoise_mc(rng: random.Random, seq_len: int, block_len: int | None = None) -> ObjectState:
    colour = _pick_colour(rng, RESERVED_MARKER_COLOURS)
    if block_len is None:
        block_len = rng.randint(5, max(5, seq_len // 3))
    start = rng.randint(2, seq_len - block_len - 3)
    end = start + block_len - 1
    active = set(range(start, end + 1))

    interior = list(range(start + 1, end))
    _require(len(interior) >= 1)
    n_corrupt = min(rng.randint(1, 2), len(interior))
    corrupt_positions = rng.sample(interior, n_corrupt)
    used = {colour} | RESERVED_MARKER_COLOURS
    corrupt_colours = {}
    for pos in corrupt_positions:
        c2 = _pick_colour(rng, used)
        corrupt_colours[pos] = c2
        used.add(c2)

    cells = {p: colour for p in active}
    cells.update(corrupt_colours)
    return ObjectState(
        active=frozenset(active),
        colour=colour,
        used_colours=used,
        used_positions=set(active),
        input_grid=_render(seq_len, cells),
    )


def _stage1_fill(rng: random.Random, seq_len: int) -> ObjectState:
    colour = _pick_colour(rng, RESERVED_MARKER_COLOURS)
    gap = rng.randint(2, max(2, seq_len // 3))
    m1 = rng.randint(2, seq_len - gap - 4)
    m2 = m1 + gap + 1
    active = set(range(m1, m2 + 1))
    return ObjectState(
        active=frozenset(active),
        colour=colour,
        used_colours={colour},
        used_positions=set(active),
        input_grid=_render(seq_len, {m1: colour, m2: colour}),
    )


def _stage1_hollow(rng: random.Random, seq_len: int) -> ObjectState:
    colour = _pick_colour(rng, RESERVED_MARKER_COLOURS)
    block_len = rng.randint(3, max(3, seq_len // 3))
    start = rng.randint(2, seq_len - block_len - 3)
    end = start + block_len - 1
    active_before = set(range(start, end + 1))
    return ObjectState(
        active=frozenset({start, end}),
        colour=colour,
        used_colours={colour},
        used_positions=set(active_before),
        input_grid=_render(seq_len, {p: colour for p in active_before}),
    )


def _stage1_move_dynamic(rng: random.Random, seq_len: int) -> ObjectState:
    colour = _pick_colour(rng, RESERVED_MARKER_COLOURS)
    block_len = rng.randint(3, max(3, seq_len // 4))
    # The pointer is always to the right of the block, matching 1d_move_dp/1d_move_2p_dp's
    # convention in the actual training data (data/arc_1d_looped_augmented shows this 100% of
    # the time, 2000/2000 sampled rows) -- a leftward pointer is a geometry the model never
    # sees even in single-task training, so it must not appear in the held-out composite set.
    gap = rng.randint(2, 6)
    start = rng.randint(1, max(1, seq_len - block_len - gap - 2))
    end = start + block_len - 1
    pointer_pos = end + 1 + gap
    _require(pointer_pos <= seq_len - 2)
    shift = (pointer_pos - 1) - end

    active_before = set(range(start, end + 1))
    active_after = {p + shift for p in active_before}
    _require(min(active_after) >= 0 and max(active_after) < seq_len)
    _require(pointer_pos not in active_after)

    cells = {p: colour for p in active_before}
    grid = _render(seq_len, cells)
    grid[pointer_pos] = MOVE_POINTER_COLOUR
    return ObjectState(
        active=frozenset(active_after),
        colour=colour,
        used_colours={colour, MOVE_POINTER_COLOUR},
        used_positions=set(active_before) | set(active_after) | {pointer_pos},
        input_grid=grid,
        persistent_markers={pointer_pos: MOVE_POINTER_COLOUR},
    )


def _stage1_shift(
    rng: random.Random, seq_len: int, n: int = SHIFT_AMOUNT, block_len: int | None = None
) -> ObjectState:
    colour = _pick_colour(rng, RESERVED_MARKER_COLOURS)
    if block_len is None:
        block_len = rng.randint(3, max(3, seq_len // 4))
    start = rng.randint(1, max(1, seq_len - block_len - n - 2))
    end = start + block_len - 1
    active_before = set(range(start, end + 1))
    active_after = {p + n for p in active_before}
    _require(max(active_after) < seq_len - 1)
    return ObjectState(
        active=frozenset(active_after),
        colour=colour,
        used_colours={colour},
        used_positions=set(active_before) | set(active_after),
        input_grid=_render(seq_len, {p: colour for p in active_before}),
    )


# --------------------------------------------------------------------------------------
# Stage 2: given stage 1's object, produce the final (input, output) pair.
# --------------------------------------------------------------------------------------


def _stage2_shift(
    rng: random.Random, seq_len: int, state: ObjectState, n: int = SHIFT_AMOUNT
) -> tuple[list[int], list[int]]:
    new_active = {p + n for p in state.active}
    _require(min(new_active) >= 0 and max(new_active) < seq_len)
    final_input = list(state.input_grid)
    cells = {p: state.colour for p in new_active}
    cells.update(state.persistent_markers)
    return final_input, _render(seq_len, cells)


def _mirror_pivot_candidates(seq_len: int, lo: int, hi: int) -> list[int]:
    """Every pivot position for which reflecting [lo, hi] around it stays in bounds and
    the pivot itself sits at least one background cell away from the object.

    Right of the object only, matching 1d_mirror's convention in the actual training data
    (data/arc_1d_looped_augmented shows this 100% of the time, 2000/2000 sampled rows) -- a
    leftward pivot is a geometry the model never sees even in single-task training.
    """
    # pivot >= hi+2 (>=1 gap cell), and reflected-max = 2*pivot-lo <= seq_len-1.
    return [p for p in range(hi + 2, (seq_len - 1 + lo) // 2 + 1) if 0 <= p < seq_len]


def _stage2_mirror(rng: random.Random, seq_len: int, state: ObjectState) -> tuple[list[int], list[int]]:
    lo, hi = min(state.active), max(state.active)
    forbidden = state.used_positions | set(state.persistent_markers)
    candidates = [p for p in _mirror_pivot_candidates(seq_len, lo, hi) if p not in forbidden]
    _require(len(candidates) > 0)
    pivot_pos = rng.choice(candidates)

    reflected = {2 * pivot_pos - p for p in state.active}
    _require(pivot_pos not in reflected)
    _require(not (reflected & set(state.persistent_markers)))

    # Always colour 9 -- 1d_mirror's pivot is a hard global constant in the real data (see
    # MIRROR_PIVOT_COLOUR), never an arbitrary "some other colour" pick.
    pivot_colour = MIRROR_PIVOT_COLOUR
    final_input = list(state.input_grid)
    final_input[pivot_pos] = pivot_colour
    for pos, colour in state.persistent_markers.items():
        final_input[pos] = colour

    cells = {p: state.colour for p in reflected}
    cells[pivot_pos] = pivot_colour
    cells.update(state.persistent_markers)
    return final_input, _render(seq_len, cells)


def _move_dynamic_pointer_candidates(seq_len: int, hi: int) -> list[int]:
    """Every pointer position to the right of the object (>=1 gap cell) such that moving the
    object to touch it stays in bounds -- right of the object only, matching move_dp/move_2p_dp's
    convention in the actual training data (verified: 100% of sampled rows)."""
    return list(range(hi + 2, seq_len))


def _stage2_move_dynamic(rng: random.Random, seq_len: int, state: ObjectState) -> tuple[list[int], list[int]]:
    lo, hi = min(state.active), max(state.active)
    forbidden = state.used_positions | set(state.persistent_markers)
    candidates = [p for p in _move_dynamic_pointer_candidates(seq_len, hi) if p not in forbidden]
    _require(len(candidates) > 0)
    pointer_pos = rng.choice(candidates)

    shift = (pointer_pos - 1) - hi
    new_active = {p + shift for p in state.active}
    _require(max(new_active) < seq_len)

    final_input = list(state.input_grid)
    final_input[pointer_pos] = MOVE_POINTER_COLOUR
    for pos, colour in state.persistent_markers.items():
        final_input[pos] = colour

    cells = {p: state.colour for p in new_active}
    cells[pointer_pos] = MOVE_POINTER_COLOUR
    cells.update(state.persistent_markers)
    return final_input, _render(seq_len, cells)


def _stage2_hollow(rng: random.Random, seq_len: int, state: ObjectState) -> tuple[list[int], list[int]]:
    new_active = {min(state.active), max(state.active)}
    final_input = list(state.input_grid)
    cells = {p: state.colour for p in new_active}
    cells.update(state.persistent_markers)
    return final_input, _render(seq_len, cells)


def _copy_marker_candidates(seq_len: int, lo: int, hi: int, shape_len: int, center_offset: int) -> list[int]:
    """Every marker position whose stamped shape (length `shape_len`, anchored `center_offset`
    cells after its start) fits in bounds without overlapping [lo, hi], with >=1 gap cell.

    Right of the object only, matching 1d_pcopy_mc's convention in the actual training data --
    every sampled example stamps its marker(s) after the template, never before.
    """
    # stamp_start >= hi+2.
    right = range(hi + 2 + center_offset, seq_len - shape_len + center_offset + 1)
    return [p for p in right if 0 <= p < seq_len]


def _stage2_copy(rng: random.Random, seq_len: int, state: ObjectState) -> tuple[list[int], list[int]]:
    lo, hi = min(state.active), max(state.active)
    shape_len = hi - lo + 1
    center_offset = shape_len // 2
    forbidden = state.used_positions | set(state.persistent_markers)

    candidates = []
    for marker_pos in _copy_marker_candidates(seq_len, lo, hi, shape_len, center_offset):
        stamped = set(range(marker_pos - center_offset, marker_pos - center_offset + shape_len))
        if marker_pos in forbidden or (stamped & forbidden):
            continue
        candidates.append(marker_pos)
    _require(len(candidates) > 0)
    marker_pos = rng.choice(candidates)
    stamped = set(range(marker_pos - center_offset, marker_pos - center_offset + shape_len))

    marker_colour = _pick_colour(rng, state.used_colours | RESERVED_MARKER_COLOURS | {BACKGROUND})
    final_input = list(state.input_grid)
    final_input[marker_pos] = marker_colour
    for pos, colour in state.persistent_markers.items():
        final_input[pos] = colour

    cells = {p: state.colour for p in state.active}
    cells.update({p: marker_colour for p in stamped})
    cells.update(state.persistent_markers)
    return final_input, _render(seq_len, cells)


def _stage2_denoise_1c(rng: random.Random, seq_len: int, state: ObjectState) -> tuple[list[int], list[int]]:
    forbidden = state.used_positions | set(state.persistent_markers)
    n_noise = rng.randint(1, 2)
    noise_positions: set[int] = set()
    for _ in range(50):
        if len(noise_positions) == n_noise:
            break
        pos = rng.randint(0, seq_len - 1)
        if pos in forbidden or pos in noise_positions:
            continue
        if any(abs(pos - p) <= 1 for p in noise_positions):
            continue
        noise_positions.add(pos)
    _require(len(noise_positions) == n_noise)

    final_input = list(state.input_grid)
    for pos in noise_positions:
        final_input[pos] = state.colour
    for pos, colour in state.persistent_markers.items():
        final_input[pos] = colour

    cells = {p: state.colour for p in state.active}
    cells.update(state.persistent_markers)
    return final_input, _render(seq_len, cells)


# --------------------------------------------------------------------------------------
# Registry: the approved combos (see the plan's diversity-selection rationale).
# 1d_comp_movedynamic_mirror was dropped per user call (didn't work out in review), replaced
# by 1d_comp_fill_movedynamic to keep move_dynamic from being a one-off.
# --------------------------------------------------------------------------------------

COMPOSITE_TASK_SPECS = {
    "1d_comp_denoise1c_shift3": (_stage1_denoise_1c, _stage2_shift),
    "1d_comp_fill_mirror": (_stage1_fill, _stage2_mirror),
    "1d_comp_fill_shift3": (_stage1_fill, _stage2_shift),
    "1d_comp_fill_movedynamic": (_stage1_fill, _stage2_move_dynamic),
    "1d_comp_hollow_shift3": (_stage1_hollow, _stage2_shift),
    # denoise_mc (not denoise_1c) pairs with copy: denoise_1c's own noise pixels are isolated
    # dots of the SAME colour as the block, which visually collide with copy's isolated marker
    # dot (a different colour) -- two same-shaped "lone dot" elements with different meanings.
    # denoise_mc's corruption sits inside the block, so the only lone dot in the grid is the
    # copy marker. Its block_len is threaded through from generate_task as the chosen
    # COPY_TEMPLATE_LENGTHS value (see _COPY_PAIRED_CATEGORIES below), not left to its own
    # default random range.
    "1d_comp_denoisemc_copy": (_stage1_denoise_mc, _stage2_copy),
    "1d_comp_denoisemc_denoise1c": (_stage1_denoise_mc, _stage2_denoise_1c),
    "1d_comp_movedynamic_hollow": (_stage1_move_dynamic, _stage2_hollow),
    "1d_comp_shift3_copy": (_stage1_shift, _stage2_copy),
    "1d_comp_denoisemc_mirror": (_stage1_denoise_mc, _stage2_mirror),
}

# Categories whose stage-1 rule feeds into `copy` -- these need a `block_len` pinned to one of
# COPY_TEMPLATE_LENGTHS (chosen once per task in generate_task), not their stage-1's own default
# random range.
_COPY_PAIRED_CATEGORIES = frozenset(
    category for category, (_, stage2_fn) in COMPOSITE_TASK_SPECS.items() if stage2_fn is _stage2_copy
)


def generate_pair(
    rng: random.Random,
    category: str,
    seq_len: int,
    template_len: int | None = None,
    max_attempts: int = 200,
) -> tuple[list[int], list[int]]:
    stage1_fn, stage2_fn = COMPOSITE_TASK_SPECS[category]
    for _ in range(max_attempts):
        try:
            if template_len is not None:
                state = stage1_fn(rng, seq_len, block_len=template_len)
            else:
                state = stage1_fn(rng, seq_len)
            return stage2_fn(rng, seq_len, state)
        except _RetryGeneration:
            continue
    msg = f"Could not generate a valid instance for {category!r} after {max_attempts} attempts"
    raise RuntimeError(msg)


def generate_task(rng: random.Random, category: str, task_id: int) -> dict:
    seq_len = rng.randint(MIN_SEQ_LEN, MAX_SEQ_LEN)
    template_len = rng.choice(COPY_TEMPLATE_LENGTHS) if category in _COPY_PAIRED_CATEGORIES else None
    pairs = [generate_pair(rng, category, seq_len, template_len=template_len) for _ in range(4)]
    return {
        "task_category": category,
        "task_id": task_id,
        "sequence_length": seq_len,
        "support_inputs": [p[0] for p in pairs[:3]],
        "support_outputs": [p[1] for p in pairs[:3]],
        "query_input": pairs[3][0],
        "query_output": pairs[3][1],
    }


def generate_dataset(seed: int, n_per_category: int, categories: list[str] | None = None) -> list[dict]:
    categories = categories if categories is not None else list(COMPOSITE_TASK_SPECS)
    rng = random.Random(seed)
    return [
        generate_task(rng, category, task_id=i) for category in categories for i in range(n_per_category)
    ]
