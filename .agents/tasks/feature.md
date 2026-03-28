# Feature Task

Goal: add a new capability with minimal, well-scoped changes.

All working rules from `AGENTS.md` apply.

## Process

### 1. Clarify the feature

- Restate the requested behaviour in concrete terms.
- Define acceptance criteria.
- Identify user-facing outcome and internal impact.
- Separate required behaviour from non-goals.
- If requirements are ambiguous, call out the uncertainty before proceeding.

### 2. Plan the change

- Identify affected files, modules, or components.
- Prefer extending existing patterns over introducing new ones.
- Note any dependencies, constraints, or trade-offs.

### 3. Implement

- Keep the code readable and consistent with the repository.
- Avoid broad structural changes unless the feature genuinely requires them.

### 4. Consider test coverage

- Do not add or update tests by default.
- If test work is warranted, first explain why the feature cannot be responsibly verified another way.
- Keep any planned test set intentionally small and high-signal.
- For each proposed test, state the main success path, edge case, or failure mode it covers and why that specific behaviour earns a test.
- Keep tests focused on observable feature behaviour, not implementation details.

### 5. Verify

Follow the verification steps in `AGENTS.md`.

## Additional response fields

Beyond the base fields in `AGENTS.md`, include:

- **Behaviour**: what the feature does now
- **Implementation**: what changed and where
- **Tests**: what tests were added, updated, run, or intentionally omitted
