#!/usr/bin/env bash
# Shared helpers for confirming that a review run posted a new PR comment.

review_agent_label() {
    case "$1" in
        codex) printf '%s\n' 'Codex' ;;
        claude) printf '%s\n' 'Claude' ;;
        *)
            echo "error: unknown review agent: $1" >&2
            return 1
            ;;
    esac
}

list_review_history() {
    local repo=$1
    local pr_number=$2
    local jq_filter

    jq_filter='.[] | (.body | try capture("_Reviewed by (?<agent>Codex|Claude)\\. Posted via `scripts/post_pr_comment\\.sh`\\._\\r?\\nReviewed commit: (?<sha>[0-9a-fA-F]{40})\\r?\\n?$") catch empty) as $footer | [.id, .html_url, $footer.sha, $footer.agent] | @tsv'
    gh api --paginate "repos/${repo}/issues/${pr_number}/comments" --jq "$jq_filter"
}

list_review_comments() {
    local repo=$1
    local pr_number=$2
    local agent=$3
    local agent_label
    local review_history

    agent_label=$(review_agent_label "$agent")
    review_history=$(list_review_history "$repo" "$pr_number") || return $?
    printf '%s\n' "$review_history" |
        awk -F '\t' -v agent="$agent_label" '$4 == agent { print $1 "\t" $2 }'
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

prepare_review_context() {
    local repo=$1
    local pr_number=$2
    local agent=$3
    local agent_label
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

    agent_label=$(review_agent_label "$agent")
    FLAT_PCA_REVIEW_COMMENT_IDS_BEFORE=$(
        printf '%s\n' "$review_history" |
            awk -F '\t' -v agent="$agent_label" '$4 == agent { print $1 }'
    )
}
