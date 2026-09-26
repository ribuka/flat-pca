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

review_comment_provenance_marker() {
    printf '%s\n' 'Posted via `scripts/post_pr_comment.sh`._'
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

list_review_history() {
    local repo=$1
    local pr_number=$2
    local marker
    local jq_filter

    marker=$(review_comment_provenance_marker)
    jq_filter=".[] | select(.body | contains(\"${marker}\")) | [.id, .html_url, (.body | try capture(\"Reviewed commit: (?<sha>[0-9a-fA-F]{40})\").sha catch \"\")] | @tsv"
    gh api --paginate "repos/${repo}/issues/${pr_number}/comments" --jq "$jq_filter"
}

prepare_review_context() {
    local repo=$1
    local pr_number=$2
    local review_history

    FLAT_PCA_REVIEW_HEAD_COMMIT=$(
        gh pr view "$pr_number" --repo "$repo" --json headRefOid --jq .headRefOid
    )
    if ! [[ "$FLAT_PCA_REVIEW_HEAD_COMMIT" =~ ^[0-9a-fA-F]{40}$ ]]; then
        echo "error: failed to resolve the 40-character headRefOid for PR #${pr_number}" >&2
        return 2
    fi

    if ! review_history=$(list_review_history "$repo" "$pr_number"); then
        echo "error: failed to query review history for PR #${pr_number}" >&2
        return 2
    fi

    FLAT_PCA_REVIEW_COUNT=$(
        printf '%s\n' "$review_history" | awk 'NF { count++ } END { print count + 0 }'
    )
    if [ "$FLAT_PCA_REVIEW_COUNT" -ge 3 ]; then
        echo "error: PR #${pr_number} already has ${FLAT_PCA_REVIEW_COUNT} cross-reviews; the maximum is 3" >&2
        return 1
    fi

    FLAT_PCA_PREVIOUS_REVIEW_COMMIT=$(
        printf '%s\n' "$review_history" | awk -F '\t' 'NF { commit=$3 } END { print commit }'
    )
    if [ "$FLAT_PCA_REVIEW_COUNT" -gt 0 ] && [ -z "$FLAT_PCA_PREVIOUS_REVIEW_COMMIT" ]; then
        echo "error: the previous review comment does not record a Reviewed commit" >&2
        return 1
    fi
    if [ "$FLAT_PCA_PREVIOUS_REVIEW_COMMIT" = "$FLAT_PCA_REVIEW_HEAD_COMMIT" ]; then
        echo "error: PR #${pr_number} has no commits after the previous review (${FLAT_PCA_REVIEW_HEAD_COMMIT})" >&2
        return 1
    fi
}
