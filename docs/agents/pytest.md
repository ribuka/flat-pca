# Pytest execution

Read this file in full before running any pytest command.

- Use `uv run -m pytest` for the full test suite.
- Use `uv run -m pytest <test-path>` for a targeted test.
- Run every pytest command directly as `uv run -m pytest ...`.
- A preconfigured Codex command rule permits the exact prefix
  `uv run -m pytest` outside the sandbox. Do not request escalated execution.
- Use the normal Windows user Temp directory. Do not override `TMP`, `TEMP`,
  or `TMPDIR`.
- If the direct pytest command is denied or cannot run outside the sandbox,
  report that verification is blocked. Do not run pytest in the sandbox or use
  an alternative temporary directory.

These rules apply only to pytest's framework-managed temporary directories. They do not change the repository rule requiring temporary scripts and investigation files to be created under `tmp/` at the repository root.
