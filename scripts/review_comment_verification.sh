#!/usr/bin/env bash
# Shared helpers for confirming that a review run posted a new PR comment.

review_comment_marker() {
    case "$1" in
        codex) printf '%s\n' '_Reviewed by Codex. Posted via `scripts/post_pr_comment.sh`._' ;;
        claude) printf '%s\n' '_Reviewed by Claude. Posted via `scripts/post_pr_comment.sh`._' ;;
        *)
            echo "error: unknown review agent: $1" >&2
            return 1
            ;;
    esac
}

list_review_comments() {
    local repo=$1
    local pr_number=$2
    local agent=$3
    local marker
    local jq_filter

    marker=$(review_comment_marker "$agent")
    jq_filter=".[] | select(.body | contains(\"${marker}\")) | [.id, .html_url] | @tsv"
    gh api --paginate "repos/${repo}/issues/${pr_number}/comments" --jq "$jq_filter"
}

snapshot_review_comment_ids() {
    list_review_comments "$1" "$2" "$3" | cut -f1
}

find_new_review_comment_url() {
    local repo=$1
    local pr_number=$2
    local agent=$3
    local previous_ids=$4
    local comment_id
    local comment_url
    local comments

    if ! comments=$(list_review_comments "$repo" "$pr_number" "$agent"); then
        echo "error: failed to query review comments for PR #${pr_number}" >&2
        return 2
    fi

    while IFS=$'\t' read -r comment_id comment_url; do
        if [ -z "$comment_id" ]; then
            continue
        fi

        case $'\n'"${previous_ids}"$'\n' in
            *$'\n'"${comment_id}"$'\n'*) ;;
            *)
                printf '%s\n' "$comment_url"
                return 0
                ;;
        esac
    done <<<"$comments"

    return 1
}
