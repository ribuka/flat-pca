# Ralph Project Configuration

This document applies the general rules in `RALPH.md` to the Flatten-PCA
repository. It is the authoritative source for repository-specific Ralph files,
commands, state formats, fixtures, runner behavior, and Git operations.

## Files to read

At the start of each loop, read the following files one at a time in this order:

1. `RALPH.md`
2. `RALPH_PROJECT.md`
3. `AGENTS.md`
4. `SPEC.md`
5. `TASKS.md`
6. `PROGRESS.md` if it exists
7. Existing code and tests related to the selected task

The runner prompt separately requires `RALPH.md` to be the first repository
operation. After reading it, continue with this list without rereading it.

On Windows PowerShell, always read Markdown and text files with an explicit
UTF-8 encoding, such as `Get-Content -Encoding UTF8`. Never rely on the default
text encoding.

Do not search the repository for Ralph instructions before reading the files
listed above. After selecting a task, begin with implementation and test files
directly required by that task. Expand to imported or dependent files only when
needed to understand an active contract or failure. Do not preload files for
later tasks, and do not read a guessed path that was absent from file-search
results.

`SPEC.md` defines the product specification, `TASKS.md` is the task ledger, and
`PROGRESS.md` is the append-only loop record. Do not invent requirements that
these files do not define.

## Task ledger and selection

Each task in `TASKS.md` uses a `TASK-XXX` heading and these fields or sections:

* `Status`: `pending`, `completed`, or `blocked`
* `Priority`: numeric selection priority
* `Depends on`: prerequisite task IDs or `none`
* `Requirements`: required behavior
* `Tests`: required verification coverage
* `Acceptance commands`: commands that must succeed

Apply the general selection rules as follows:

1. Consider only tasks with `Status: pending`.
2. Confirm that every task listed in `Depends on` has `Status: completed`.
3. Select the task with the lowest `Priority` value.
4. If multiple tasks have the same priority, select the lowest `TASK-XXX` ID.

In `TASKS.md`, normally change only the selected task's `Status`. Do not change
requirements, tests, dependencies, or priority during a loop.

## Tests and acceptance commands

Run targeted tests while implementing. Before marking a task completed, run all
commands in its `Acceptance commands` section and always run these project-wide
checks:

```powershell
uv run -m pytest
uv run -m ruff check .
```

Use this form for a targeted test:

```powershell
uv run -m pytest <test-path>
```

In every Flatten-PCA loop, add or update at least one test that directly reads
Parquet data from `tests/fixtures/real_subset`. Do not modify, duplicate, or
overwrite those fixture files. Derive error and boundary cases from a real
fixture in memory, writing only to pytest's `tmp_path` when a Parquet file is
required. Use fully synthetic data only when the case cannot be expressed from
the real fixture, and document the reason in the test.

## Architecture review

Review the files changed in the loop and confirm that:

* Each module has one clear primary responsibility.
* Public API modules and `__init__.py` files are thin and explicitly re-export
  public symbols.
* Dependencies flow in one direction and introduce no circular imports.
* Tests are organized by the behavior they verify.
* No dead code, unnecessary compatibility shim, or generic helper module was
  introduced without a documented need.
* Public import paths remain compatible unless the task explicitly changes them.
* Every function and class has type hints and a NumPy-style docstring.

## Progress updates

Append one entry to `PROGRESS.md` for every task loop. All headings and content
must be written in English using this format:

```markdown
## YYYY-MM-DD HH:MM - TASK-XXX

- Result: completed | pending | blocked
- Changes: Implementation or documentation changes made in this loop
- Tests: Commands run and their results
- Requirements: Verification result for each requirement
- Notes: Information required by the next loop
```

If new work is discovered, do not add it to `TASKS.md` without approval. Record
it as a candidate task in `PROGRESS.md`.

## Automated runner

Run consecutive loops with the repository runner:

```powershell
uv run -m scripts.ralph_runner
```

The runner starts a fresh `codex exec` process for each loop. It stops when all
tasks are complete, a task is incomplete or blocked, Codex fails, the output
protocol is invalid, or the configured loop limit is reached. Run
`uv run -m scripts.ralph_runner --help` for available options and exit codes.

## Git and external operations

* Each loop with repository changes must create exactly one commit containing
  only files changed in that loop, including `TASKS.md` and `PROGRESS.md` updates.
* The runner starts only from a clean working tree, stages the loop changes,
  checks unstaged and staged diffs for whitespace errors, and creates the
  commit. Codex must review its diff but must not stage or commit it.
* A completed-task commit message must contain `TASK-XXX` and a short
  description.
* Preserved incomplete or blocked work is committed as
  `wip(TASK-XXX): Ralph loop changes`, with failures and remaining work recorded
  in `PROGRESS.md`.
* Do not create an empty commit when a loop makes no file changes.
* Do not push, pull, publish, or deploy.
* Do not perform destructive operations or write outside the repository.
* Never overwrite, delete, or revert uncommitted user changes.
