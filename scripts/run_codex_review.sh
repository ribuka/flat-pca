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
#     post_pr_comment.sh) can reach the network, while allowing overrides;
#   - enforces a hard timeout and a heartbeat (no-progress) timeout around
#     `codex exec`, so a hang is killed and reported instead of being left
#     running unnoticed (see issue #35).
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
#   --timeout-seconds <n>           Hard upper bound on the whole `codex exec`
#                                   run. On expiry the process is killed and
#                                   the script exits 124. (default: 1800, i.e.
#                                   30 minutes)
#   --heartbeat-timeout-seconds <n> If the progress log (see --log-file) does
#                                   not grow for this many seconds, `codex
#                                   exec` is assumed to be hung, killed, and
#                                   the script exits 125. (default: 600, i.e.
#                                   10 minutes)
#   --log-file <path>               Where to write the JSONL progress log
#                                   that `codex exec --json` produces (default:
#                                   tmp/codex_review_logs/pr<N>-<timestamp>.jsonl).
#                                   The path is also printed to stderr at
#                                   start-up so it can be tailed from outside
#                                   to check liveness/progress. Removed on
#                                   success; kept for debugging on failure,
#                                   timeout, or hang.
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
timeout_seconds=1800
heartbeat_timeout_seconds=600
log_file=""

require_value() {
    if [ "$#" -lt 2 ]; then
        echo "error: $1 requires a value" >&2
        exit 1
    fi
}

require_positive_int() {
    if ! [[ "$2" =~ ^[0-9]+$ ]] || [ "$2" -le 0 ]; then
        echo "error: $1 must be a positive integer, got: $2" >&2
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
        --timeout-seconds)
            require_value "$@"
            require_positive_int "$1" "$2"
            timeout_seconds=$2
            shift 2
            ;;
        --heartbeat-timeout-seconds)
            require_value "$@"
            require_positive_int "$1" "$2"
            heartbeat_timeout_seconds=$2
            shift 2
            ;;
        --log-file)
            require_value "$@"
            log_file=$2
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

if ! command -v codex >/dev/null 2>&1; then
    echo "error: codex command not found in PATH" >&2
    exit 1
fi

if [ -z "$log_file" ]; then
    log_dir="tmp/codex_review_logs"
    mkdir -p "$log_dir"
    log_file="${log_dir}/pr${pr_number}-$(date +%Y%m%d-%H%M%S)-$$.jsonl"
else
    mkdir -p "$(dirname "$log_file")"
fi
: >"$log_file"

echo "info: codex progress log: ${log_file}" >&2
echo "info: hard timeout: ${timeout_seconds}s, heartbeat (no-progress) timeout: ${heartbeat_timeout_seconds}s" >&2

prompt="このリポジトリの PR #${pr_number} をレビューしてください。差分を確認し、日本語でレビューコメントを作成してください。"
prompt+=$'\n'"レビューが完了したら、コメント本文をファイルに書き出し、必ず次のコマンドで投稿してください（\`gh pr comment\` を直接使わないこと）: scripts/post_pr_comment.sh ${pr_number} <body-file> codex"

if [ -n "$extra_instructions" ]; then
    prompt+=$'\n\n'"追加指示: ${extra_instructions}"
fi

# --json makes codex emit one JSONL event per line of progress, so the
# growth of $log_file can be used as a heartbeat signal (as opposed to just
# checking whether the process is alive, which says nothing about whether it
# is actually making progress).
codex_args=(exec --json --model "$model" -c "model_reasoning_effort=${effort}" --sandbox "$sandbox")

if [ "$sandbox" = "workspace-write" ]; then
    codex_args+=(-c "sandbox_workspace_write.network_access=${network_access}")
fi

if [ "$approve_for_me" = "true" ]; then
    codex_args+=(--approve-for-me)
fi

# Run codex in its own process group (via job control) so that a kill can
# target the whole group -- codex exec spawns child tool/shell processes of
# its own, and killing only codex_pid would leave those behind.
set -m
codex "${codex_args[@]}" "$prompt" >"$log_file" 2>&1 &
codex_pid=$!
set +m

# Kills the codex process group. Registered as a trap so that the group is
# also reaped if this wrapper itself is killed or interrupted (it used to
# `exec codex ...`, which let signals reach codex directly; now that codex
# runs in the background, that no longer happens without an explicit trap).
cleanup() {
    if kill -0 "$codex_pid" 2>/dev/null; then
        kill -TERM -- "-${codex_pid}" 2>/dev/null || kill -TERM "$codex_pid" 2>/dev/null || true
        sleep 2
        kill -KILL -- "-${codex_pid}" 2>/dev/null || kill -KILL "$codex_pid" 2>/dev/null || true
    fi
    wait "$codex_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

poll_interval_seconds=10
start_time=$(date +%s)
last_log_size=-1
last_progress_time=$start_time

while true; do
    # Re-check aliveness both before and after the sleep: codex may have
    # exited *during* the sleep, in which case the timeout checks below must
    # be skipped rather than misreported as a timeout/hang.
    if ! kill -0 "$codex_pid" 2>/dev/null; then
        break
    fi
    sleep "$poll_interval_seconds"
    if ! kill -0 "$codex_pid" 2>/dev/null; then
        break
    fi

    now=$(date +%s)
    elapsed=$((now - start_time))

    if [ "$elapsed" -ge "$timeout_seconds" ]; then
        echo "error: codex exec exceeded the hard timeout of ${timeout_seconds}s; killing pid ${codex_pid}. Progress log: ${log_file}" >&2
        exit 124
    fi

    current_log_size=$(wc -c <"$log_file" 2>/dev/null || echo 0)
    if [ "$current_log_size" != "$last_log_size" ]; then
        last_log_size=$current_log_size
        last_progress_time=$now
    elif [ "$((now - last_progress_time))" -ge "$heartbeat_timeout_seconds" ]; then
        echo "error: no progress in codex log for ${heartbeat_timeout_seconds}s (hang suspected); killing pid ${codex_pid}. Progress log: ${log_file}" >&2
        exit 125
    fi
done

set +e
wait "$codex_pid"
exit_code=$?
set -e

# Keep the progress log around for post-mortem debugging when codex failed,
# timed out, or hung; remove it on success so tmp/codex_review_logs/ does not
# accumulate indefinitely (per AGENTS.md's rule to clean up temp files).
if [ "$exit_code" -eq 0 ]; then
    rm -f "$log_file"
fi

exit "$exit_code"
