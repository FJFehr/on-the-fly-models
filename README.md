# on-the-fly-models
Generate task-specific models on-the-fly by predicting their weights from data, enabling efficient generalisation with minimal training.

## Setup

```bash
uv sync --python 3.12 --managed-python
```

## Data

### 1D-ARC dataset

Download and convert the [1D-ARC](https://github.com/khalil-research/1D-ARC) dataset into a HuggingFace Dataset stored locally under `data/arc_1d/`:

```bash
uv run python build_arc_1d.py
```

Each row contains `task_category`, `task_id`, `split` (train/test), `example_idx`, `input`, and `output` fields.

### Visualisation

Render a sample input/output pair using the standard ARC colour palette:

```bash
uv run python visualise_data.py
```

Saves an example plot to `data/arc_1d_example.png`.

## Agentic workflow

The `.agents/` folder is a portable operating contract for coding agents (Claude Code, Copilot, Cursor, or any LLM-based tool). It defines how agents should behave, what style to follow, and how tasks are structured — so you get consistent, reviewable output regardless of which agent runs the work.

### What's inside

| File | Purpose |
|------|---------|
| `AGENTS.md` | The root contract — repo priorities, working rules, version control policy, verification steps, and response format. Every agent reads this first. |
| `STYLE.md` | Coding style guide — naming, formatting, modularity, error handling, tests. |
| `PROMPT_TEMPLATE.md` | A fill-in template for dispatching tasks to an agent. |
| `tasks/feature.md` | Process guide for adding new functionality. |
| `tasks/debug.md` | Process guide for diagnosing and fixing bugs. |
| `tasks/refactor.md` | Process guide for restructuring code without changing behaviour. |

Agents read files in precedence order: `AGENTS.md` > `STYLE.md` > `PROMPT_TEMPLATE.md` > task brief > `README.md`. When instructions conflict, higher-precedence files win.

### How to prompt an agent

Copy the template from `PROMPT_TEMPLATE.md` and fill in the fields:

```
Task type: feature
Mode: implement
Objective: Add a --dry-run flag to the training script that skips writing checkpoints.
Acceptance criteria: Running with --dry-run completes without writing to disk. Existing runs unaffected.
Constraints: Do not change the checkpoint format or add new dependencies.
Known unknowns: Unclear whether eval should also be skipped; assume no.
Relevant files: train.py, tasks/feature.md
Out-of-scope files: data/, tests/ (read only)
Verification: uv run python train.py --dry-run exits cleanly; existing tests pass.
```

The agent will then follow the matching task brief (`tasks/feature.md` in this case) for its process, apply the working rules from `AGENTS.md`, and respond using the structured output format.

**Tips:**
- Set **Mode** to `propose-only` when you want a plan without code changes.
- Use **Constraints** and **Out-of-scope files** to keep the agent focused.
- The **Known unknowns** field lets the agent handle ambiguity explicitly rather than guessing silently.