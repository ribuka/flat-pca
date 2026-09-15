Run exactly one Ralph loop in this repository.

Treat `RALPH.md` as the authoritative loop procedure. Read and follow every
file and rule it references. Select exactly one eligible pending task, add or
update the required tests, implement only that task, run all required checks,
update task and progress state, and create the required single commit.

Do not start another task in this run. Do not weaken tests or requirements. If
the specification is missing, ambiguous, or contradictory, stop and report the
task as blocked instead of guessing. Do not overwrite, revert, or commit
pre-existing user changes.

Summarize the task, checks, and commit in the final response. The final
non-empty line must be exactly one of the output forms defined in the
`RALPH.md` "Loop output" section.
