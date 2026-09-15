# Ralph Loop Rules

This document defines the rules for processing exactly one task from `TASKS.md` per loop.

## Files to read

At the start of each loop, read the following files one at a time in this order:

1. `AGENTS.md`
2. `SPEC.md`
3. `TASKS.md`
4. `PROGRESS.md` if it exists
5. Existing code and tests related to the selected task

On Windows PowerShell, always read Markdown and text files with an explicit
UTF-8 encoding, such as `Get-Content -Encoding UTF8`. Never rely on the default
text encoding.

Do not search the repository for Ralph instructions before reading these files.
After selecting a task, begin with implementation and test files directly
required by that task. Expand to imported or dependent files only when needed
to understand an active contract or failure. Do not preload files for later
tasks, and do not read a guessed path that was absent from file-search results.

Do not invent requirements that are not documented. If specifications conflict, do not implement the task. Report it as blocked.

## Task selection

1. Consider only tasks with `Status: pending`.
2. Confirm that all tasks listed in `Depends on` are `completed`.
3. Select the task with the lowest `priority`.
4. If multiple tasks have the same priority, select the lowest `id`.
5. Select, implement, and complete only one task per loop.

If all incomplete tasks are blocked by dependencies, make no changes and end the loop as blocked.

## One loop

1. Read the selected task's `requirements`, `tests`, and `acceptance_commands`.
2. Search the codebase to check whether the requirements are already implemented.
3. Add or update tests that verify the task requirements.

   * For Flatten-PCA tasks, include at least one test that loads `tests/fixtures/real_subset`, as required by `AGENTS.md`.
4. Run the relevant tests and confirm they fail appropriately for unimplemented requirements.
5. Implement only the minimum changes required for the selected task.
6. Run the relevant tests again.
7. Run all `acceptance_commands` defined for the task.
8. Verify every `requirement` against the implementation and tests.
9. Change the selected task's `Status` to `completed` only if all completion criteria are met.
10. Append the loop result to `PROGRESS.md` in English.
11. Review only the files changed in this loop. The automated runner creates the
   required single Git commit after validating the loop result, then ends the loop.

NEVER weaken requirements or delete, skip, or xfail tests only to make tests pass.

## Completion criteria

The selected task is complete only if all of the following are true:

* All `requirements` are satisfied.
* Automated tests exist for the defined `tests`.
* All added or modified tests pass.
* `uv run -m pytest`, including existing tests, passes.
* All `acceptance_commands` exit with code 0.
* No unrelated functionality or public API is changed.
* No temporary debug code, generated artifacts, or unused files remain.

Passing tests alone does not mean the task is complete if requirement verification fails.

## Failure and blocked tasks

* If completion criteria are not met, do not set the task `Status` to `completed`.
* If human judgment, credentials, external services, or undefined specifications are required, stop instead of guessing.
* Record in `PROGRESS.md` what was done, successful checks, failures, and the next required decision.
* If you find an unrelated issue, do not fix it unless required by the selected task.
* If new work is needed, do not add tasks to `TASKS.md` without approval. Record them as candidate tasks in `PROGRESS.md`.

## State updates

In `TASKS.md`, normally change only the selected task's `Status`. Do not modify requirements, test conditions, dependencies, or priority during the loop.

Append to `PROGRESS.md` using this format. All headings and content must be written in English.

```markdown
## YYYY-MM-DD HH:MM - TASK-XXX

- Result: completed | pending | blocked
- Changes: Implementation or documentation changes made in this loop
- Tests: Commands run and their results
- Requirements: Verification result for each requirement
- Notes: Information required by the next loop
```

## Loop output

* Task completed: `TASK_COMPLETED: TASK-XXX`
* Task incomplete: `TASK_INCOMPLETE: TASK-XXX`
* Task blocked by human judgment or similar: `TASK_BLOCKED: TASK-XXX`
* All tasks completed: `ALL_TASKS_COMPLETED`

The final output must reflect the actual state. After one loop ends, do not start another task.

## Automated runner

Run consecutive loops with the repository runner:

```powershell
uv run -m scripts.ralph_runner
```

The runner starts a fresh `codex exec` process for each loop and stops when all
tasks are complete, a task is incomplete or blocked, Codex fails, the output
protocol is invalid, or the configured loop limit is reached. Run
`uv run -m scripts.ralph_runner --help` for available options and exit codes.

## Git and external operations

* Each loop with repository changes must create exactly one commit containing only files changed in that loop, including updates to `TASKS.md` and `PROGRESS.md`.
* The runner starts only from a clean working tree, stages the loop changes, checks the unstaged and staged diff for whitespace errors, and creates that commit. Codex must review its diff but must not stage or commit it.
* For completed tasks, the commit message must include `TASK-XXX` and a short description of the change.
* If incomplete or blocked work must be preserved, commit it as `wip(TASK-XXX): ...` and record failed checks and remaining work in `PROGRESS.md`.
* Do not create empty commits for loops with no file changes.
* Do not push, pull, publish, or deploy.
* Do not perform destructive operations or write outside the repository.
* NEVER overwrite, delete, or revert uncommitted user changes.
