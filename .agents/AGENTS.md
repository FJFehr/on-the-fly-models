# AGENTS.md

This repository is designed to be used with multiple coding agents. Treat this file as the common operating contract for the repo.

## Instruction precedence

When instructions conflict, follow this order:
1. This file (`AGENTS.md`)
2. `STYLE.md`
3. `PROMPT_TEMPLATE.md`
4. The relevant task brief in `tasks/`
5. `README.md`

All paths above are relative to this `.agents/` directory unless otherwise noted.

Before making changes, read the applicable files in the order above, plus any files directly related to the change.

## Repo priorities

When working in this repo, optimise for:
- correctness over cleverness
- small, reviewable changes
- consistency with the existing codebase
- minimal unnecessary scope
- explicit trade-offs when choices matter

## Task files

Use the task brief that matches the kind of work being done:

- `tasks/feature.md` for new functionality
- `tasks/refactor.md` for internal code improvements without changing behaviour
- `tasks/debug.md` for diagnosing and fixing bugs

Task briefs define the **process** for each task type. The rules below apply to all tasks.

## Working rules

- Follow `STYLE.md` and the existing style of the repository unless the task explicitly asks for a change.
- Make the smallest change that solves the problem.
- Preserve existing behaviour unless the task explicitly changes it.
- Do not introduce new dependencies unless clearly justified.
- Avoid unrelated clean-up. Do not rename or move files unless the task requests it.
- Keep changes local unless the task requires a broader edit. Before editing, state which files you expect to change.
- No speculative additions — do not add features, abstractions, or refactors beyond what was asked.
- If the task is proposal-only or asks for a plan, do not make file edits or run mutating commands.

## Test planning

- Do not create, expand, or update tests by default.
- Treat test work as opt-in. Only add or change tests when the user explicitly asks for them or when you first state a concrete justification for why the change cannot be responsibly verified another way.
- If a task brief says to add tests, follow this stricter policy unless the user explicitly wants test work.
- Before writing any test, explain the planned set and keep it intentionally small.
- For each planned test, state:
	- the exact behaviour or failure mode it covers
	- why that behaviour is important enough to justify a test
	- why the chosen test is worth keeping instead of relying on a broader or more numerous test set
- Prefer one rigorous, high-signal test over several partially redundant ones.
- If no tests are written, document the alternative verification used instead.

## When uncertain

If requirements, code paths, or expected behaviour are ambiguous:
- If the ambiguity is low-risk and cannot change observable behaviour, state your assumption and proceed.
- If the ambiguity could materially affect correctness or behaviour, stop, explain the uncertainty, and ask for clarification.
- Never make hidden decisions. Always surface the assumptions you acted on.

## Version control

- Pre-commit hooks are installed and **must not be skipped** (`--no-verify` is not allowed).
- Hooks run: ruff lint with auto-fix, ruff format, trailing whitespace trimming, end-of-file fixing, YAML validation, and a 500 KB large-file guard.
- If a hook fails, fix the issue and create a new commit — do not amend the previous commit unless explicitly asked.
- Do not commit generated files (`*.egg-info/`, `.venv/`, `__pycache__/`, etc.). Check `.gitignore` before staging.
- For ML artifacts (model weights, datasets, large outputs), use Git LFS or store them outside the repo. Never commit files over 500 KB without justification.
- Write concise commit messages that explain *why*, not *what*. One logical change per commit.

## README

Before every commit, check whether `README.md` needs updating. If new files, directories, scripts, or functionality have been added or removed, update the repository layout table and any relevant sections so the README stays accurate.

## Verification

Before finishing, check your work in a way appropriate to the change:
- Run existing tests only when they are relevant to the change and worth the cost.
- Do not add or update tests unless the test-planning rules above have been satisfied.
- Sanity-check edge cases.
- When tests do not exist or cannot be run, use alternative evidence: manual review, static analysis, or reasoning about affected code paths.
- Explicitly state anything you could not verify and why.

## Response format

Every response uses these base fields. Task briefs may add task-specific fields.

- **Summary**: what changed and why
- **Verification**: what was checked and how
- **Risks / unknowns**: remaining assumptions, gaps, or follow-ups

## Coordination note

This repo is currently using manual orchestration between agents. Do not attempt to chain work to another agent unless explicitly instructed.
