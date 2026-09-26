# Pull request

## Agent review

- Only the agent that created a pull request may request cross-reviews. A review agent must not request an additional review.
- Cross-reviews are limited to three per pull request: one initial review and at most two re-reviews.
- Request the initial review once after creating the pull request. It covers the complete pull-request diff.
- Request a re-review only after fixing review findings in code and pushing a new commit. Do not request one when the previous review had no findings or when the response only explains or disputes a finding without changing code.
- A re-review covers only the changes from the commit recorded by the previous review through the current HEAD and the previous findings. It checks that the findings were resolved and that the fixes introduced no new problems. It must not raise new findings about existing code outside that diff.
- Fix actionable findings and reply on the pull request with what changed. After the third review, do not request another review. If a finding is deferred or requires a judgment call, do not merge and ask the user how to proceed.
- To request that review, do NOT invoke `codex`/`claude` directly and do NOT call `gh pr comment` directly.
- A Claude pull-request author launches Codex via `scripts/run_codex_review.sh <pr-number>`; a Codex pull-request author launches Claude via `scripts/run_claude_review.sh <pr-number>`.
- Both scripts reject a launch when they are run from an existing review environment, when three reviews have already been posted, or when HEAD has not changed since the previous review. This prevents nested reviews and enforces the re-review limits.
- Both scripts capture the current PR `headRefOid`. On re-review they automatically identify the previous reviewed commit and add the allowed commit range to the prompt.
- Both scripts embed the instruction to post the review with `scripts/post_pr_comment.sh <pr-number> <body-file> <codex|claude>`, and default model/effort/sandbox/permission settings to values that let `gh` run. See each script's header comment for defaults and override flags.
- `scripts/post_pr_comment.sh` appends `Reviewed commit: <sha>` using the commit captured by the parent wrapper; review agents must not set or change that value.
- Wait until the review script process exits. If command execution returns control while the process is still running, keep polling that same process; do not start another review for the PR.
- Do not decide the review result or merge until the script exits with status 0 and prints `review-posted: <comment-url>`. Open that URL and inspect the posted comment before proceeding.

## Merge conditions

- All CI checks have succeeded.
- Every finding has a pull-request reply stating either that it was fixed or why it was deferred.
- No unresolved findings remain. A deferred or disputed finding requires user confirmation before merge.

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

## Attribution

- NEVER include session URLs (e.g. `https://claude.ai/code/session_...`) in pull request descriptions.
  - Omit the trailing session link.
- The "Generated with Claude Code" line is allowed.
