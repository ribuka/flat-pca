# instructions

- A repository for developing a preprocessing workflow for time-series spectral data using Flatten-PCA.

## Python

- Always use `uv run -m`. NEVER use `python -m`, `Set-Lacation` or one-liners and here-string in pwsh.

### Python coding

- Before working python coding, read `docs/agents/python.md` and follow it.

### Verification commands

- Run targeted tests while implementing a task.
- Run the full test suite and lint before marking the task as completed.

#### Test execution

- **NEVER** run pytest inside the Codex Windows sandbox.
- Before running any pytest command, read `docs/agents/pytest.md` and follow it.

#### Lint

- Lint: `uv run -m ruff check .`

### Test fixtures

- In every Flatten-PCA loop, add or update at least one test that directly reads
  Parquet data from `tests/fixtures/real_subset`.
- Do not modify, duplicate, or overwrite the fixture Parquet files.
- For error and boundary cases, derive a `pl.DataFrame` in memory from a real
  fixture and write it to pytest's `tmp_path` only when a Parquet file is needed.
- Use fully synthetic data only for a minimal test case that cannot be expressed
  from the real fixture. Document the reason in the test docstring or a comment.

## Temporary files

- Create all temporary scripts and investigation files under `tmp/` at the repository root.
- NEVER create temporary files elsewhere; delete them when the task is complete.
- Store the persistent uv dependency cache in the ignored repository-root `.uv-cache/` directory.

## Rules

- Respond in Japanese.
- Include probabilities when providing answers with uncertainty.
- Start tasks with Plan mode when more than 3 steps required.
- NEVER write code without reading it.
