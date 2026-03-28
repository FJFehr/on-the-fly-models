# Refactor Task

Goal: improve codebase structure, clarity, or maintainability without changing external behaviour.

All working rules from `AGENTS.md` apply.

## Process

### 1. Identify the refactor target

- State what is hard to read, maintain, duplicated, or brittle.
- Explain why the refactor is worth doing.
- State explicitly what APIs, interfaces, and observable behaviour must remain unchanged.

### 2. Plan the change

- Identify the smallest useful scope.
- Prefer local edits over broad restructuring.
- Reuse existing patterns where possible.
- Call out any risks to behaviour or compatibility.

### 3. Refactor

- Keep the external interface stable unless the task says otherwise.
- Make the code simpler, clearer, or more consistent.
- Remove duplication only when it genuinely improves the design.

### 4. Adjust tests

- Do not add or expand tests by default.
- Only adjust tests when the refactor changes an existing behavioural contract, invalidates a current test, or leaves no credible verification path without a targeted test change.
- Keep any test changes intentionally small and focused on confirming behaviour has not changed.
- If no test changes are made, state the alternative verification used instead.

### 5. Verify

Follow the verification steps in `AGENTS.md`.

## Additional response fields

Beyond the base fields in `AGENTS.md`, include:

- **Before / after**: what improved in the code
- **Behaviour**: why the change is behaviour-preserving
- **Tests**: what tests were run, updated, or intentionally left unchanged
