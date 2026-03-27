# on-the-fly-models

Generate task-specific models on-the-fly by predicting their weights from data, enabling efficient generalisation with minimal training.

The idea: instead of training a separate model per task, a **hyper-model** learns to predict the weights of a small **target model** given only a handful of input/output examples. We use the [1D-ARC](https://github.com/khalil-research/1D-ARC) dataset as our testbed.

## Setup

```bash
uv sync --python 3.12 --managed-python
```

## Data

### Building the 1D-ARC dataset

Download and convert the 1D-ARC dataset into a task-level HuggingFace `DatasetDict` with stratified `train`/`dev`/`test` splits (80/10/10):

```bash
uv run python build_arc_1d.py
```

Each record is one complete ARC task (3 support examples + 1 query) that always stays together in the same split. Fields: `task_category`, `task_id`, `sequence_length`, `support_inputs`, `support_outputs`, `query_input`, `query_output`.

**Simplified dataset** — binarised sequences (0/non-zero), padded to length 33, restricted to 7 simple categories:

```bash
uv run python build_arc_1d.py --simple
```

**Holdout category** — reserve an entire category for out-of-distribution evaluation:

```bash
uv run python build_arc_1d.py --holdout-category 1d_mirror
```

### Visualisation

**Task grids** — render tasks using the standard ARC colour palette (support examples + masked query):

```bash
uv run python visualise_tasks.py --data-dir data/arc_1d --split train
```

Useful flags: `--task-category`, `--task-id`, `--max-tasks`, `--first-per-category`.

**Sequence-length boxplots** — grouped by category and split, with token counts:

```bash
uv run python eda_arc_1d_boxplots.py --data-dir data/arc_1d
```

## Models

The `models/` directory contains the two planned architectures:

- **target-model** — a simple MLP that performs the 1D transformation for a single task.
- **hyper-model** — a Hypermixer-style network that predicts the target model's weights from support examples.

Training entrypoint: `train.py` (scaffolding).

## Tests

```bash
uv run pytest
```

Tests cover data pipeline correctness: no task leakage across splits, correct split ratios, stratification, holdout isolation, and per-task validation.

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
