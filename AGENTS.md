# instructions

- A repository for developing a preprocessing workflow for time-series spectral data using Flatten-PCA.
- Always respond in Japanese.

## Python

- Always use `uv run -m`.
- NEVER use `python -m`, `Set-Location` or one-liners and here-string in powershell.

### Python coding

- When working python coding, read `docs/agents/python.md` and follow it.

### Verification commands

- Run targeted tests (`uv run -m pytest <test-path>`) while implementing a task.
- Run lint (`uv run -m ruff check .`) before marking the task as completed.
- Run the full test suite (`uv run -m pytest --tb=short -q`) once before opening a pull request;
  CI re-runs lint and the full suite on every pull request to main.

#### Test execution

- **NEVER** run pytest inside the Codex Windows sandbox.
- Before running any pytest command, read `docs/agents/pytest.md` and follow it.

## Github

### Pull request

- When creating a pull request, read `docs/agents/pull-request.md` and follow it.

## Temporary files

- Create all temporary scripts and investigation files under `tmp/` at the repository root.
- NEVER create temporary files elsewhere; delete them when the task is complete.
