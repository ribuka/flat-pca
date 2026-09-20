Read `AGENTS.md` and follow.

## Terminal commands

- Run independent terminal commands as separate tool calls.
- NEVER chain commands with `;`, `&&`, or `||` when they can be run separately.
- Assume terminal commands run from the workspace root unless a different working directory is explicitly required.
- NEVER prepend `Set-Location` to routine commands when already operating in the workspace.
