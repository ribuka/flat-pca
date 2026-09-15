Run exactly one Ralph loop in this repository.

Your first repository operation must be reading `RALPH.md` with explicit UTF-8
encoding. Do not search the repository for Ralph instructions. Treat `RALPH.md`
as the authoritative loop procedure, and then read each file it requires one at
a time in the documented order. Select exactly one eligible pending task, add
or update the required tests, implement only that task, run all required checks,
and update task and progress state. Do not stage or commit: the runner creates
the required single commit after it validates your terminal status.

Do not start another task in this run. Do not weaken tests or requirements. If
the specification is missing, ambiguous, or contradictory, stop and report the
task as blocked instead of guessing. Do not overwrite, revert, or commit
pre-existing user changes.

Summarize the task, checks, and commit in the final response. The final
non-empty line must be exactly one of the output forms defined in the
`RALPH.md` "Loop output" section.
