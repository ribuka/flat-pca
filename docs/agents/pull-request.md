# Pull request

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
