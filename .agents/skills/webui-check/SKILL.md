---
name: webui-check
description: Launch the flat-pca Web UI on a temporary workspace, operate its screens in Chrome through Playwright MCP, and clean up afterwards. Use when asked to run, start, open, or screenshot the Web UI, or to confirm that a Web UI change works in a real browser.
---

# Web UI check

Serve the Web UI on a throwaway copy of the test fixtures, drive it with the
Playwright MCP server, then stop it and delete every temporary file.

## Prerequisites

- The Playwright MCP server is configured (see `.agents/docs/playwright-mcp.md`).
  Its tools are named `browser_*` (for example `mcp__playwright__browser_navigate` in Claude Code).
- Google Chrome is installed.
- Dependencies are synced: `uv sync`.

## 1. Prepare a temporary workspace

Copy the fixtures to `tmp/webui-check/` at the repository root.
Never point the server at `tests/fixtures/webui/` itself: it creates `workspace/` next to `settings.toml`.
`tmp/` is gitignored and may not exist in a fresh checkout or worktree, so create it first.

- Bash:
  1. `mkdir -p tmp`
  2. `cp -r tests/fixtures/webui tmp/webui-check`
- PowerShell:
  1. `New-Item -ItemType Directory -Force tmp`
  2. `Copy-Item -Recurse tests/fixtures/webui tmp/webui-check`

The copy holds:

- `settings.toml`: relative paths resolve against its directory; `workspace.dir` is `workspace`.
- `data/`: `run-1.parquet`, `run-2.parquet`, `sub/run-10.parquet`.
- `meta.csv`: metadata for `run-1` (lot `A`), `run-2` (lot `B`), and `ghost` (no Parquet file).
  `run-10` has no row.

To check other data, edit the copied `settings.toml` (`[data]`, `[metadata]`) instead of the fixtures.

## 2. Start the server

```bash
uv run -m flat_pca.webui --settings tmp/webui-check/settings.toml --port 8765
```

- Run it as a background process (Claude Code: Bash with `run_in_background`; Codex: a separate long-running command).
  Redirect the log to `tmp/webui-check/server.log` if you want to read it later.
- Port 8765 is the default for this check. If it is in use, pick another port; do not stop processes you did not start.
- Wait until `http://127.0.0.1:8765/` answers with status 200 (for example poll it with `curl`).
  `[Errno 10048]` / `address already in use` in the log means the port is taken.

## 3. Check the screens with Playwright MCP

Use `browser_navigate` to open `http://127.0.0.1:8765/`, then `browser_snapshot` to read the page
and get element references for `browser_click`, `browser_select_option`, and `browser_type`.

### Data selection (`/`)

| Operation | Expected result with the fixtures |
| --- | --- |
| Open the page | "catalog はまだ作成されていません。"; the file table shows 0 件; the sidebar shows catalog 未作成. |
| Click "catalog 更新" | The status shows `queued`/`running` and polls every second; the button is disabled. |
| Wait (`browser_wait_for` text `succeeded`) | Status `succeeded`, 3 ファイル; the file table reloads with `run-1`, `run-2`, `run-10`; the warnings list `run-10` (no metadata) and `ghost` (no file). |
| Category filter `lot` | Options are `(すべて)`, `A`, `B` after the update; choosing `A` leaves only `run-1`. |
| Numeric / datetime filters | e.g. `yield_pct 下限` = 90 leaves only `run-1`. |
| Sort / order selects | The table reorders. |
| "表示中をすべて選択" / "選択を外す" | All visible checkboxes are checked / cleared. |
| "このファイル集合を選択" | "選択中：N ファイル" and the sidebar's 選択中 count update. |

Other screens are added to the sidebar as they are implemented; check them the same way.

### Always check

- `browser_console_messages`: no errors.
- `browser_network_requests`: no 4xx/5xx responses.
- Save screenshots (`browser_take_screenshot`) only under `tmp/webui-check/`.

## 4. Clean up

1. Close the browser with `browser_close`.
2. Stop the server process you started (Claude Code: `TaskStop` on the background task; Codex: stop the process).
3. Confirm the port is free: the URL no longer answers.
4. Delete `tmp/webui-check/` and `tmp/playwright-mcp/`. On Windows the files stay locked until the server has exited.

## 5. Keep stable behavior as a regression test

When a checked behavior is stable and needs a browser (JS, htmx swaps, polling, navigation),
add a pytest-playwright test under `tests/webui/e2e/` (see `.agents/docs/pytest.md`).
Check HTML content and partial targets with the `TestClient` tests in `tests/webui/` instead.
