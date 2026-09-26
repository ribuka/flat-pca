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
#   - waits for `claude -p` to exit and confirms that it posted a new review
#     comment before printing `review-posted: <url>` and exiting successfully.
#   - limits each PR to three reviews, rejects an unchanged re-review, and
#     narrows re-reviews to commits added after the previous review.
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

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${script_dir}/review_comment_verification.sh"

review_environment_variable="FLAT_PCA_REVIEW_IN_PROGRESS"

if [ -n "${!review_environment_variable:-}" ]; then
    echo "error: nested review launch is not allowed while ${review_environment_variable} is set" >&2
    exit 1
fi

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

repo=$(gh repo view --json nameWithOwner --jq .nameWithOwner)
prepare_review_context "$repo" "$pr_number"
review_comment_ids_before=$(snapshot_review_comment_ids "$repo" "$pr_number" claude)

if [ "$FLAT_PCA_REVIEW_COUNT" -eq 0 ]; then
    prompt="このリポジトリの PR #${pr_number} をレビューしてください。PR 全体の差分を確認し、日本語でレビューコメントを作成してください。"
else
    prompt="このリポジトリの PR #${pr_number} を再レビューしてください。レビュー対象は ${FLAT_PCA_PREVIOUS_REVIEW_COMMIT}..${FLAT_PCA_REVIEW_HEAD_COMMIT} の差分と前回の指摘だけです。前回の指摘が解消されているか、修正によって新しい問題が入っていないかを確認してください。対象差分の外にある既存コードへの新しい指摘はしないでください。日本語でレビューコメントを作成してください。"
fi
prompt+=$'\n'"レビューが完了したら、コメント本文をファイルに書き出し、必ず次のコマンドで投稿してください（\`gh pr comment\` を直接使わないこと）: scripts/post_pr_comment.sh ${pr_number} <body-file> claude"

if [ -n "$extra_instructions" ]; then
    prompt+=$'\n\n'"追加指示: ${extra_instructions}"
fi

export "${review_environment_variable}=1"
export FLAT_PCA_REVIEWED_COMMIT="$FLAT_PCA_REVIEW_HEAD_COMMIT"

# Run Claude in its own process group so cancelling this wrapper can terminate
# Claude and any tool processes it spawned before they post a late comment.
set -m
claude -p --model "$model" --effort "$effort" --permission-mode "$permission_mode" "$prompt" &
claude_pid=$!
set +m

cleanup() {
    if kill -0 "$claude_pid" 2>/dev/null; then
        kill -TERM -- "-${claude_pid}" 2>/dev/null || kill -TERM "$claude_pid" 2>/dev/null || true
        sleep 2
        kill -KILL -- "-${claude_pid}" 2>/dev/null || kill -KILL "$claude_pid" 2>/dev/null || true
    fi
    wait "$claude_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

set +e
wait "$claude_pid"
exit_code=$?
set -e

if [ "$exit_code" -ne 0 ]; then
    exit "$exit_code"
fi

if comment_url=$(find_new_review_comment_url "$repo" "$pr_number" claude "$review_comment_ids_before"); then
    echo "review-posted: ${comment_url}"
    exit 0
else
    verification_status=$?
fi

if [ "$verification_status" -eq 1 ]; then
    echo "error: Claude exited successfully, but no new review comment was found for PR #${pr_number}" >&2
fi
exit "$verification_status"
