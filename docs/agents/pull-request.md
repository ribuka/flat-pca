# Pull request

## Agent review

- When you need another agent to review a pull request, do NOT invoke `codex`/`claude` directly and do NOT call `gh pr comment` directly.
- If you are Claude, launch Codex for review via `scripts/run_codex_review.sh <pr-number>`.
- If you are Codex, launch Claude for review via `scripts/run_claude_review.sh <pr-number>`.
- Both scripts embed the instruction to post the review with `scripts/post_pr_comment.sh <pr-number> <body-file> <codex|claude>`, and default model/effort/sandbox/permission settings to values that let `gh` run. See each script's header comment for defaults and override flags.

## Model labeling

- **ALWAYS** add exactly one model-family label:
  - `model:gpt`
  - `model:claude`
- Create the required label if it does not exist.
- **NEVER** create any other `model:*` label.

## Optional information

- Optionally add the following information to the pull request description:
  - `Implementation agent: <agent>`
  - `Implementation model: <exact model>`
  - `Reasoning effort:  <reasoning effort>`

Examples:

```text
`model:gpt`
Implementation agent: Codex
Implementation model: gpt-5.6-terra
Reasoning effort: medium
```

```text
`model:claude`
Implementation agent: Claude Code
Implementation model: claude-sonnet-5
Reasoning effort: high
```

```text
`model:gpt`
Implementation agent: Copilot
Implementation model: gpt-5.6-luna
Reasoning effort: low
```
