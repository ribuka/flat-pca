#!/usr/bin/env bash
# Launch Codex CLI non-interactively to review a pull request of this repository.
#
# This is meant to be used by an agent (for example Claude Code) that wants
# to spawn Codex CLI as a review subprocess, instead of hand-rolling a
# `codex exec` invocation each time. Compared to calling `codex exec`
# directly, it:
#   - always appends the instruction to post the review via
#     `scripts/post_pr_comment.sh <pr-number> <body-file> codex`, so that
#     instruction cannot be dropped from a one-off subprocess prompt;
#   - defaults model/effort to the values `wiggum/wiggum_codex.bat` uses
#     (gpt-5.6-terra / medium), while allowing overrides;
#   - defaults the sandbox to a mode where `gh` (used by
#     post_pr_comment.sh) can reach the network, while allowing overrides.
#
# Usage:
#   scripts/run_codex_review.sh <pr-number> [options] [-- <extra instructions>]
#
# Options:
#   --model <model>                Codex model (default: gpt-5.6-terra)
#   --effort <effort>              Reasoning effort, passed through as
#                                   `-c model_reasoning_effort=<effort>`
#                                   (default: medium)
#   --sandbox <mode>                read-only|workspace-write|danger-full-access
#                                   (default: workspace-write)
#   --network-access <true|false>  Only applies when --sandbox=workspace-write;
#                                   controls sandbox_workspace_write.network_access.
#                                   `gh pr comment` needs this to be true.
#                                   (default: true)
#   --approve-for-me                Pass `--approve-for-me` to `codex exec`
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

model="gpt-5.6-terra"
effort="medium"
sandbox="workspace-write"
network_access="true"
approve_for_me="false"
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
        --sandbox)
            require_value "$@"
            sandbox=$2
            shift 2
            ;;
        --network-access)
            require_value "$@"
            network_access=$2
            shift 2
            ;;
        --approve-for-me)
            approve_for_me="true"
            shift
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

if ! command -v codex >/dev/null 2>&1; then
    echo "error: codex command not found in PATH" >&2
    exit 1
fi

prompt="このリポジトリの PR #${pr_number} をレビューしてください。差分を確認し、日本語でレビューコメントを作成してください。"
prompt+=$'\n'"レビューが完了したら、コメント本文をファイルに書き出し、必ず次のコマンドで投稿してください（\`gh pr comment\` を直接使わないこと）: scripts/post_pr_comment.sh ${pr_number} <body-file> codex"

if [ -n "$extra_instructions" ]; then
    prompt+=$'\n\n'"追加指示: ${extra_instructions}"
fi

codex_args=(exec --model "$model" -c "model_reasoning_effort=${effort}" --sandbox "$sandbox")

if [ "$sandbox" = "workspace-write" ]; then
    codex_args+=(-c "sandbox_workspace_write.network_access=${network_access}")
fi

if [ "$approve_for_me" = "true" ]; then
    codex_args+=(--approve-for-me)
fi

exec codex "${codex_args[@]}" "$prompt"
