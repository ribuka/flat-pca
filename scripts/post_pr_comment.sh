#!/usr/bin/env bash
# Post exactly one issue-level comment to a pull request of this repository.
#
# This is a deliberately narrow wrapper around `gh pr comment`, meant to be
# handed to an agent (for example an external code-review agent) instead of
# raw `gh`/network access. Compared to calling `gh` directly, it:
#   - always targets the current checkout's own repository (derived from
#     `gh repo view`), so it cannot be pointed at an unrelated repository;
#   - only ever runs the single `gh pr comment` write action -- it has no
#     code path for editing/deleting comments, merging, closing, or
#     changing labels/reviewers;
#   - refuses to run unless the target PR actually exists in this repo;
#   - appends a fixed provenance footer itself, so the footer cannot be
#     omitted or altered by whatever produced the comment body.
#   - records the PR head commit captured by the parent review wrapper.
#
# Usage:
#   scripts/post_pr_comment.sh <pr-number> <body-file> <agent>
#
# <body-file> is the path to a text file containing the Markdown comment
# body. Use "-" to read the body from stdin instead.
#
# <agent> identifies which CLI agent produced the comment. It must be
# either "codex" or "claude"; the script records it in the footer itself
# so the caller cannot misreport which agent actually ran.

set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "usage: $0 <pr-number> <body-file> <codex|claude>" >&2
    exit 1
fi

pr_number=$1
body_file=$2
agent=$3

if ! [[ "$pr_number" =~ ^[0-9]+$ ]]; then
    echo "error: <pr-number> must be a positive integer, got: $pr_number" >&2
    exit 1
fi

if [ "$body_file" != "-" ] && [ ! -f "$body_file" ]; then
    echo "error: body file not found: $body_file" >&2
    exit 1
fi

case "$agent" in
    codex) agent_label="Codex" ;;
    claude) agent_label="Claude" ;;
    *)
        echo "error: <agent> must be 'codex' or 'claude', got: $agent" >&2
        exit 1
        ;;
esac

reviewed_commit=${FLAT_PCA_REVIEWED_COMMIT:-}
if ! [[ "$reviewed_commit" =~ ^[0-9a-fA-F]{40}$ ]]; then
    echo "error: FLAT_PCA_REVIEWED_COMMIT must be set to the 40-character PR head commit" >&2
    exit 1
fi

repo=$(gh repo view --json nameWithOwner --jq .nameWithOwner)

# Fail fast if the PR does not exist in this repo, rather than letting
# `gh pr comment` create ambiguity about what was targeted.
gh pr view "$pr_number" --repo "$repo" --json url >/dev/null

comment_file=$(mktemp)
trap 'rm -f "$comment_file"' EXIT

if [ "$body_file" = "-" ]; then
    cat >"$comment_file"
else
    cat "$body_file" >"$comment_file"
fi

printf '\n\n_Reviewed by %s. Posted via `scripts/post_pr_comment.sh`._\n' "$agent_label" >>"$comment_file"
printf 'Reviewed commit: %s\n' "$reviewed_commit" >>"$comment_file"

gh pr comment "$pr_number" --repo "$repo" --body-file "$comment_file"
