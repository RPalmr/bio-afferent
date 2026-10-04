# Contributing

1. Implement in phases: correctness first, then performance.
2. Every public API needs types, docstrings (units, shapes), and tests.
3. Never claim a check passed without running it (`pytest`, `ruff`, `mypy`).
4. Scientific changes must cite primary literature and record the formulation
   in `docs/physiology.md`. Do not invent parameters; document engineering
   choices explicitly.
5. Failing tests must be fixed at the root cause — never disabled, deleted,
   or weakened without written justification.
6. Git commits require explicit human authorization. Use short lowercase
   imperative messages (e.g. `add spindle dynamics`).
