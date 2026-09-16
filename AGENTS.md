# instructions

- A repository for developing a preprocessing workflow for time-series spectral data using Flatten-PCA.

## Python

- Always use `uv run -m`. NEVER use `python -m`, `Set-Lacation` or one-liners and here-string in pwsh.

### Verification commands

- Full test suite: `uv run -m pytest`
- Targeted test: `uv run -m pytest <test-path>`
- Lint: `uv run -m ruff check .`

Run targeted tests while implementing a task. Run the full test suite and lint
before marking the task as completed.

### Test fixtures

- In every Flatten-PCA loop, add or update at least one test that directly reads
  Parquet data from `tests/fixtures/real_subset`.
- Do not modify, duplicate, or overwrite the fixture Parquet files.
- For error and boundary cases, derive a `pl.DataFrame` in memory from a real
  fixture and write it to pytest's `tmp_path` only when a Parquet file is needed.
- Use fully synthetic data only for a minimal test case that cannot be expressed
  from the real fixture. Document the reason in the test docstring or a comment.

### Architecture and maintainability

- Keep one primary responsibility per Python module.
- NEVER append a distinct responsibility to an existing module; create a focused
  module or package instead.
- Split independently testable stages such as input, validation, preprocessing,
  transformation, and orchestration into cohesive modules when appropriate.
- Keep public API modules and `__init__.py` files thin, explicitly re-export
  public symbols, and preserve documented import paths during refactoring.
- NEVER introduce circular imports or generic catch-all modules such as `utils.py`.
- Organize tests by responsibility and share setup through narrowly scoped pytest
  fixtures. NEVER split modules based on line count alone.

## Temporary files

- Create all temporary scripts and investigation files under `tmp/` at the repository root.
- NEVER create temporary files elsewhere; delete them when the task is complete.
- Store the persistent uv dependency cache in the ignored repository-root `.uv-cache/` directory.

## Rules
- Respond in Japanese
- Include probabilities when providing answers with uncertainty
- Start tasks with Plan mode when more than 3 steps required
- NEVER write code without reading it
- Make changes only where necessary, minimize impact
- Add docstrings for all functions and classes with NumPy style
- Add type hints for all function parameters and return types
