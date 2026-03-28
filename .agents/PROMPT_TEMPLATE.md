# Prompt Template

Fill in the fields below when dispatching a task to an agent.

## Required fields

**Task type:** [feature | debug | refactor | docs]
**Mode:** [propose-only | implement]
**Objective:** One sentence stating the concrete outcome.
**Acceptance criteria:** What done looks like. Measurable if possible.
**Constraints:** Things that must not change or must not be added.
**Known unknowns:** Open questions or unclear inputs.
**Relevant files:** List of files the agent should read.
**Out-of-scope files:** Files that must not be modified.
**Verification:** How to confirm the task is complete.

## Execution mode

- `propose-only` — plan and explain only. Do not edit files or run mutating commands.
- `implement` — plan, then make the changes, then verify.

---

## Example (filled)

**Task type:** feature
**Mode:** implement
**Objective:** Add a CLI flag `--dry-run` to the training script that skips writing checkpoints.
**Acceptance criteria:** Running with `--dry-run` completes without writing to disk. Existing runs without the flag are unaffected.
**Constraints:** Do not change the checkpoint format or add new dependencies.
**Known unknowns:** Unclear whether the eval step should also be skipped; assume no unless training is the only concern.
**Relevant files:** `train.py`, `tasks/feature.md`
**Out-of-scope files:** `data/`, `tests/` (read only)
**Verification:** `uv run python train.py --dry-run` exits cleanly; existing tests pass.
