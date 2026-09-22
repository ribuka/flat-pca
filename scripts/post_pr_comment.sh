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
#
# Usage:
#   scripts/post_pr_comment.sh <pr-number> <body-file>
#
# <body-file> is the path to a text file containing the Markdown comment
# body. Use "-" to read the body from stdin instead.

set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "usage: $0 <pr-number> <body-file>" >&2
    exit 1
fi

pr_number=$1
body_file=$2

if ! [[ "$pr_number" =~ ^[0-9]+$ ]]; then
    echo "error: <pr-number> must be a positive integer, got: $pr_number" >&2
    exit 1
fi

if [ "$body_file" != "-" ] && [ ! -f "$body_file" ]; then
    echo "error: body file not found: $body_file" >&2
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

printf '\n\n_Posted via `scripts/post_pr_comment.sh`_\n' >>"$comment_file"

gh pr comment "$pr_number" --repo "$repo" --body-file "$comment_file"
