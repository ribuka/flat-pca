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
- While developing, add `--reload` to restart the server whenever a `*.py` file under `src/flat_pca` changes.
  - A reload interrupts running jobs (they become `cancelled`), so do not edit Python code while checking a job.
  - Templates and static files are not watched: templates apply on the next request, static files after a browser reload.
  - On Windows, uvicorn restarts the worker with a Ctrl+C event, which does not reach a process started without a console
    (e.g. a Claude Code background Bash). There the reload is detected but the old worker keeps running;
    start the server in its own console window (PowerShell `Start-Process ... -WindowStyle Minimized`) or restart it manually.

## 3. Check the screens with Playwright MCP

Use `browser_navigate` to open `http://127.0.0.1:8765/`, then `browser_snapshot` to read the page
and get element references for `browser_click`, `browser_select_option`, and `browser_type`.

### Data selection (`/`)

| Operation | Expected result with the fixtures |
| --- | --- |
| Open the page | "No catalog has been built yet."; the file table shows 0 of 0; the sidebar shows catalog not built. |
| Click "Update catalog" | The status shows `queued`/`running` and polls every second; the button is disabled. |
| Wait (`browser_wait_for` text `succeeded`) | Status `succeeded`, 3 files; the file table reloads with `run-1`, `run-2`, `run-10`; the warnings list `run-10` (no metadata) and `ghost` (no file). |
| Column headers | Each shows the name, `⋯`, and the type below it (`str`, `cat`, `datetime[μs]`, `f64`, `i64`). |
| Click `lot` (opens its column menu) | The menu shows Asc / Desc / Clear sort, the values `A 1`, `B 1`, "Null values 1", and Copy column name, and Apply / Cancel / Clear; checking `A` changes nothing but enables Apply and shows "Unapplied changes"; Apply leaves only `run-1`, closes the menu, and marks the header with a funnel. Escape, Cancel, or a click outside closes it and drops the unapplied changes. |
| `Is null` in the `lot` menu, then Apply | Only `run-10` (no metadata) remains. |
| `Search…` above the table | `1 RUN` leaves `run-1` and `run-10`; the input keeps the focus. |
| Filter chips above the table | Each filter in use shows as a chip (e.g. `lot ∈ {A}`, `yield_pct ≥ 90`); its × removes the filter. |
| `Columns` above the table | Unchecking a column hides it ("3 rows, 6 columns"); the choice survives a page reload; `file` cannot be hidden. |
| Numeric / datetime filters (in the menu) | e.g. `yield_pct lower` = 90 and Enter (or Apply) leaves only `run-1`. |
| Asc / Desc in a column menu | The table sorts ascending (▲) or descending (▼) and the menu closes. |
| Header checkbox of the file table | Checks / clears every row matching the filters, on all pages; with only some of them checked it shows the indeterminate state. |
| Click a row, then Shift + click another row | The first click toggles the row; Shift + click selects (or clears) every row between them. Under the table "3 rows, 7 columns", "k selected" with Clear, and "1–3 of 3" follow. |
| `Pin columns` above the table, then narrow the table and scroll it sideways | Off at first: every column scrolls. Turned on (the switch moves right, `aria-pressed="true"`), the checkbox and file columns stay at the left; the choice survives paging, filtering, and a page reload. |
| Click a group heading (catalog / Files) | The group folds and unfolds. |
| "Select" (below the table, right) | A green check icon (Material Symbols `check_circle`) and "Fit target: N files" appear left of the button; the sidebar's Fit target count updates. |

### Preprocessing and PCA (`/fit`)

Select the three files on the data selection screen first.

| Operation | Expected result with the fixtures |
| --- | --- |
| Open the page | Steps `1` and `2` are checked; the disabled ranges show the catalog ranges (400 – 402.5 nm, Time 0 – 3); the estimate shows 3 files × about 21 features; "No fit has run yet."; every group is a collapsible card. |
| Uncheck Step 1 | The estimate updates to about 12 features. |
| Enable Wavelength range with 402 – 401 and click "Run fit" (top left of the Run status card) | The form comes back with the error next to Wavelength range; no run is queued. |
| Fix the form and click "Run fit" | The status shows `queued`/`running` with the progress and a Cancel button, then a final status; the run list reloads. The fixture files differ only by a constant, so the run ends `failed` with "no residual variance remains …"; use your own data (edit the copied `settings.toml`) to see `succeeded`. |
| A run that ends `succeeded` | A green check icon (`check_circle`) appears right of "Run fit". |
| Click the run in the run list | `/fit?run=<run_id>` shows the run's settings in the form and its status. |

### Spectral exploration (`/explore`)

| Operation | Expected result |
| --- | --- |
| Open the raw view of a file | The StepTime slider stands vertically left of the heatmap, its thumb on the row of the crosshair; triangles above and left of the plot area point at the selected point. |
| Move either slider or click the heatmap | The crosshair, the triangles, and the trends follow the point. |

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
