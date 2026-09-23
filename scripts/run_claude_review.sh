#!/usr/bin/env bash
# Launch Claude CLI non-interactively to review a pull request of this repository.
#
# This is meant to be used by an agent (for example Codex CLI) that wants to
# spawn Claude Code as a review subprocess, instead of hand-rolling a
# `claude -p` invocation each time. Compared to calling `claude` directly, it:
#   - always appends the instruction to post the review via
#     `scripts/post_pr_comment.sh <pr-number> <body-file> claude`, so that
#     instruction cannot be dropped from a one-off subprocess prompt;
#   - defaults model/effort to the values `wiggum/wiggum_claude.bat` uses
#     (claude-sonnet-5 / medium), while allowing overrides;
#   - defaults the permission mode so that `gh` (used by
#     post_pr_comment.sh) does not fail on a permission prompt that a
#     non-interactive session cannot answer, while allowing overrides.
#
# Usage:
#   scripts/run_claude_review.sh <pr-number> [options] [-- <extra instructions>]
#
# Options:
#   --model <model>            Claude model (default: claude-sonnet-5)
#   --effort <effort>          Effort level: low|medium|high|xhigh|max
#                               (default: medium)
#   --permission-mode <mode>   acceptEdits|auto|bypassPermissions|manual|
#                               dontAsk|plan (default: bypassPermissions)
#
# Anything after a literal `--` is appended to the review prompt as extra
# instructions (for example, what to focus the review on).

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <pr-number> [options] [-- <extra instructions>]" >&2
    exit 1
fi

pr_number=$1
shift

if ! [[ "$pr_number" =~ ^[0-9]+$ ]]; then
    echo "error: <pr-number> must be a positive integer, got: $pr_number" >&2
    exit 1
fi

model="claude-sonnet-5"
effort="medium"
permission_mode="bypassPermissions"
extra_instructions=""

require_value() {
    if [ "$#" -lt 2 ]; then
        echo "error: $1 requires a value" >&2
        exit 1
    fi
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --model)
            require_value "$@"
            model=$2
            shift 2
            ;;
        --effort)
            require_value "$@"
            effort=$2
            shift 2
            ;;
        --permission-mode)
            require_value "$@"
            permission_mode=$2
            shift 2
            ;;
        --)
            shift
            extra_instructions="$*"
            break
            ;;
        *)
            echo "error: unknown option: $1" >&2
            exit 1
            ;;
    esac
done

if ! command -v claude >/dev/null 2>&1; then
    echo "error: claude command not found in PATH" >&2
    exit 1
fi

prompt="このリポジトリの PR #${pr_number} をレビューしてください。差分を確認し、日本語でレビューコメントを作成してください。"
prompt+=$'\n'"レビューが完了したら、コメント本文をファイルに書き出し、必ず次のコマンドで投稿してください（\`gh pr comment\` を直接使わないこと）: scripts/post_pr_comment.sh ${pr_number} <body-file> claude"

if [ -n "$extra_instructions" ]; then
    prompt+=$'\n\n'"追加指示: ${extra_instructions}"
fi

exec claude -p --model "$model" --effort "$effort" --permission-mode "$permission_mode" "$prompt"
