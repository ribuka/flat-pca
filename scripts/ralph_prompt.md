Run exactly one Ralph loop in this repository.

Your first repository operation must be reading `RALPH.md` with explicit UTF-8
encoding. Do not search the repository for Ralph instructions first. Treat
`RALPH.md` as the authoritative general loop procedure, then read
`RALPH_PROJECT.md` as the authoritative repository configuration. Continue with
the remaining files one at a time in the order defined there. Select exactly one
eligible pending task, add or update the required tests, implement only that
task, run all required checks, and update task and progress state. Do not stage
or commit: the runner creates the required single commit after it validates your
terminal status.

Do not start another task in this run. Do not weaken tests or requirements. If
the specification is missing, ambiguous, or contradictory, stop and report the
task as blocked instead of guessing. Do not overwrite, revert, or commit
pre-existing user changes.

Run pytest only with the documented `uv run -m pytest` commands. Request an
approved sandbox-external execution for each pytest command so it uses the
normal user temporary directory. Do not add `--basetemp` or create pytest
temporary directories in the repository root.

Summarize the task and checks in the final response without claiming that Codex
created the runner-owned commit. The final non-empty line must be exactly one of
the output forms defined in the `RALPH.md` "Loop output" section.
