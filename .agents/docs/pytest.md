# Pytest execution

## Mandatory

- Run pytest only outside the Codex Windows sandbox.
- Request out-of-sandbox execution before the first pytest run; do not try it in the sandbox first.
- If out-of-sandbox execution is unavailable or denied, do not run pytest; report it instead.
- NEVER change pytest temp-directory settings or test code merely to work around sandbox permission errors.

## Command

- Use `uv run -m pytest` for the full test suite.
- Use `uv run -m pytest <test-path>` for a targeted test.

## E2E tests

- Browser tests of the Web UI live in `tests/webui/e2e/` and carry the `e2e` marker.
  `pyproject.toml` deselects them by default (`-m "not e2e"`), so `uv run -m pytest` does not run them.
- Run them with `uv run -m pytest -m e2e` (add a path to narrow them, e.g. `uv run -m pytest -m e2e tests/webui/e2e/test_data_selection.py`).
- They use pytest-playwright with the installed Google Chrome (`--browser-channel chrome` in `addopts`); do not run `playwright install` for a bundled browser.
- Pass `--headed` to watch the browser, or `--slowmo 500` to slow it down.
- Each test serves the app with uvicorn in a background thread on a free port (`server_url` / `cataloged_server_url` fixtures)
  and fails when the page logs a console error.
- Use browser tests only for what needs a browser (JS, htmx swaps, polling, navigation); check HTML content with the `TestClient` tests.
- CI runs them in the `E2E` job.
