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

## Example

Read the ./agents folder and the README

Then:

**Task type:** feature / refactor / debug
**Mode:** implement / plan
**Objective:** 
