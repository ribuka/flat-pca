# Ralph Loop Rules

This document defines the general rules for completing work through Ralph loops.
Repository-specific files, commands, and operating constraints belong in the
project configuration rather than this document.

Project rules may specialize these rules, but must not weaken them. If the
applicable instructions are missing, ambiguous, or contradictory, do not guess;
report the selected task as blocked.

## Core invariant

One loop processes exactly one task. After that task reaches a terminal result,
end the loop without selecting or starting another task.

## Task selection

1. Consider only tasks whose status makes them eligible for work.
2. Exclude tasks with incomplete dependencies.
3. Apply the project's documented priority and tie-breaking rules.
4. Select exactly one task using that deterministic order.

If no incomplete task is eligible because of its dependencies, make no changes
and end the loop as blocked. If no incomplete task remains, end the loop with
the all-tasks-completed result.

## One loop

1. Read the selected task's requirements, expected tests, and acceptance checks.
2. Inspect the existing implementation and tests that directly affect the task.
3. Identify the responsibilities of the files that may change and preserve the
   project's architectural boundaries.
4. Add or update tests that demonstrate the required behavior.
5. For behavior not already implemented, run the relevant tests and confirm
   that they fail for the expected reason.
6. Implement only the minimum changes required for the selected task.
7. Run the relevant tests again.
8. Run every project and task acceptance check.
9. Verify each requirement against the resulting implementation and tests.
10. Review only the changes made in this loop for scope and maintainability.
11. Update the project-defined task and progress state.
12. Report exactly one terminal result and end the loop.

Never weaken requirements or delete, skip, or mark tests as expected failures
only to make the checks pass.

## Completion criteria

The selected task is complete only when all of the following are true:

* Every requirement is satisfied.
* Automated tests cover the behavior required by the task.
* All added or modified tests pass.
* All task-specific and project-wide acceptance checks pass.
* Requirement verification succeeds independently of the test result.
* No unrelated behavior or public API changed.
* No temporary debug code, generated artifacts, or unused files remain.
* The changes satisfy the project's architecture and maintainability rules.

Passing tests alone does not make a task complete.

## Incomplete and blocked tasks

* Report a task as incomplete when work or verification remains but no external
  decision is required.
* Report a task as blocked when completion requires human judgment,
  credentials, an external service, an unavailable dependency, or clarification
  of missing or conflicting requirements.
* Do not mark an incomplete or blocked task as completed.
* Do not fix unrelated issues unless the selected task requires the fix.
* Record failures, completed checks, and the next required action in the
  project-defined progress record.
* Do not add or redefine tasks without the approval required by the project.

## Loop output

The last non-empty line of the final response must be exactly one of these
forms and must match the actual repository state:

* Task completed: `TASK_COMPLETED: TASK-XXX`
* Task incomplete: `TASK_INCOMPLETE: TASK-XXX`
* Task blocked: `TASK_BLOCKED: TASK-XXX`
* All tasks completed: `ALL_TASKS_COMPLETED`

After emitting the terminal result, do not start another task.
