"""Review wrapper completion reporting tests."""

from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
HEAD_COMMIT = "b" * 40
PREVIOUS_COMMIT = "a" * 40


def find_bash() -> str:
    """Return an available Bash executable."""
    bash = shutil.which("bash")
    if bash is not None:
        return bash

    git_bash = Path(r"C:\Program Files\Git\bin\bash.exe")
    if git_bash.is_file():
        return str(git_bash)

    pytest.skip("Bash is required to test review wrapper scripts")


def run_review_script(
    script_name: str,
    *,
    post_comment: bool,
    existing_review_commits: tuple[str, ...] = (),
    head_commit: str = HEAD_COMMIT,
    api_failure_after_review: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a review wrapper with deterministic fake CLI and GitHub commands."""
    state_name = f"review-test-{uuid.uuid4().hex}"
    state_file = Path("tmp") / f"{state_name}.state"
    log_file = Path("tmp") / f"{state_name}.jsonl"
    capture_file = Path("tmp") / f"{state_name}.capture"
    harness = r"""
gh() {
    case "$1 $2" in
        "repo view") printf '%s\n' 'ribuka/flat-pca' ;;
        "pr view") printf '%s\n' "$REVIEW_HEAD_COMMIT" ;;
        "api --paginate")
            if [ "$REVIEW_API_FAILURE_AFTER" = "true" ] && [ -f "$REVIEW_STATE_FILE" ]; then
                return 22
            fi
            include_commit=false
            for argument in "$@"; do
                if [[ "$argument" == *"Reviewed commit"* ]]; then
                    include_commit=true
                fi
            done
            comment_id=900
            for commit in $REVIEW_EXISTING_COMMITS; do
                comment_id=$((comment_id + 1))
                if [ "$include_commit" = "false" ]; then
                    printf '%s\thttps://github.com/ribuka/flat-pca/pull/123#issuecomment-%s\n' "$comment_id" "$comment_id"
                elif [ "$commit" = "missing" ]; then
                    printf '%s\thttps://github.com/ribuka/flat-pca/pull/123#issuecomment-%s\t\n' "$comment_id" "$comment_id"
                else
                    printf '%s\thttps://github.com/ribuka/flat-pca/pull/123#issuecomment-%s\t%s\n' "$comment_id" "$comment_id" "$commit"
                fi
            done
            if [ -f "$REVIEW_STATE_FILE" ]; then
                if [ "$include_commit" = "true" ]; then
                    printf '1001\thttps://github.com/ribuka/flat-pca/pull/123#issuecomment-1001\t%s\n' "$REVIEW_HEAD_COMMIT"
                else
                    printf '1001\thttps://github.com/ribuka/flat-pca/pull/123#issuecomment-1001\n'
                fi
            fi
            ;;
        *) return 1 ;;
    esac
}

post_fake_review() {
    if [ "$REVIEW_POST_COMMENT" = "true" ]; then
        mkdir -p "$(dirname "$REVIEW_STATE_FILE")"
        : >"$REVIEW_STATE_FILE"
    fi
}

claude() {
    printf 'prompt: %s\n' "${!#}" >"$REVIEW_CAPTURE_FILE"
    printf 'reviewed-commit: %s\n' "$FLAT_PCA_REVIEWED_COMMIT" >>"$REVIEW_CAPTURE_FILE"
    post_fake_review
}

codex() {
    printf 'prompt: %s\n' "${!#}" >"$REVIEW_CAPTURE_FILE"
    printf 'reviewed-commit: %s\n' "$FLAT_PCA_REVIEWED_COMMIT" >>"$REVIEW_CAPTURE_FILE"
    post_fake_review
}

export -f gh post_fake_review claude codex
export REVIEW_STATE_FILE REVIEW_POST_COMMENT

review_args=("$REVIEW_SCRIPT" 123)
if [ "$REVIEW_SCRIPT" = "scripts/run_codex_review.sh" ]; then
    review_args+=(--log-file "$REVIEW_LOG_FILE")
fi

bash "${review_args[@]}"
review_status=$?
if [ -f "$REVIEW_CAPTURE_FILE" ]; then
    cat "$REVIEW_CAPTURE_FILE"
fi
exit "$review_status"
"""
    environment = {
        "REVIEW_SCRIPT": f"scripts/{script_name}",
        "REVIEW_STATE_FILE": state_file.as_posix(),
        "REVIEW_LOG_FILE": log_file.as_posix(),
        "REVIEW_CAPTURE_FILE": capture_file.as_posix(),
        "REVIEW_POST_COMMENT": str(post_comment).lower(),
        "REVIEW_EXISTING_COMMITS": " ".join(existing_review_commits),
        "REVIEW_HEAD_COMMIT": head_commit,
        "REVIEW_API_FAILURE_AFTER": str(api_failure_after_review).lower(),
    }

    try:
        return subprocess.run(
            [find_bash(), "-c", harness],
            cwd=REPOSITORY_ROOT,
            env={**os.environ, **environment},
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    finally:
        (REPOSITORY_ROOT / state_file).unlink(missing_ok=True)
        (REPOSITORY_ROOT / log_file).unlink(missing_ok=True)
        (REPOSITORY_ROOT / capture_file).unlink(missing_ok=True)


@pytest.mark.parametrize("script_name", ["run_claude_review.sh", "run_codex_review.sh"])
def test_review_script_reports_new_comment_url(script_name: str) -> None:
    """A newly posted review comment is reported as successful completion."""
    result = run_review_script(script_name, post_comment=True)

    assert result.returncode == 0, result.stderr
    assert (
        "review-posted: "
        "https://github.com/ribuka/flat-pca/pull/123#issuecomment-1001\n"
    ) in result.stdout
    assert f"reviewed-commit: {HEAD_COMMIT}" in result.stdout
    assert "PR 全体の差分" in result.stdout


@pytest.mark.parametrize("script_name", ["run_claude_review.sh", "run_codex_review.sh"])
def test_review_script_fails_without_new_comment(script_name: str) -> None:
    """A successful CLI exit without a new comment is treated as failure."""
    result = run_review_script(script_name, post_comment=False)

    assert result.returncode == 1
    assert "no new review comment was found" in result.stderr
    assert "review-posted:" not in result.stdout


@pytest.mark.parametrize("script_name", ["run_claude_review.sh", "run_codex_review.sh"])
def test_review_script_does_not_accept_existing_comment(script_name: str) -> None:
    """A review comment that predates the wrapper run is not accepted."""
    result = run_review_script(
        script_name,
        post_comment=False,
        existing_review_commits=(PREVIOUS_COMMIT,),
    )

    assert result.returncode == 1
    assert "no new review comment was found" in result.stderr
    assert "review-posted:" not in result.stdout


@pytest.mark.parametrize("script_name", ["run_claude_review.sh", "run_codex_review.sh"])
def test_review_script_rejects_fourth_review(script_name: str) -> None:
    """A fourth cross-review is rejected before an agent is launched."""
    result = run_review_script(
        script_name,
        post_comment=False,
        existing_review_commits=("1" * 40, "2" * 40, "3" * 40),
    )

    assert result.returncode == 1
    assert "maximum is 3" in result.stderr
    assert "prompt:" not in result.stdout


@pytest.mark.parametrize("script_name", ["run_claude_review.sh", "run_codex_review.sh"])
def test_review_script_rejects_unchanged_head(script_name: str) -> None:
    """A re-review without a new commit is rejected before launch."""
    result = run_review_script(
        script_name,
        post_comment=False,
        existing_review_commits=(HEAD_COMMIT,),
    )

    assert result.returncode == 1
    assert "has no commits after the previous review" in result.stderr
    assert "prompt:" not in result.stdout


@pytest.mark.parametrize("script_name", ["run_claude_review.sh", "run_codex_review.sh"])
def test_review_script_rejects_untracked_previous_review(script_name: str) -> None:
    """A previous wrapper comment without a commit cannot define a re-review range."""
    result = run_review_script(
        script_name,
        post_comment=False,
        existing_review_commits=(PREVIOUS_COMMIT, "missing"),
    )

    assert result.returncode == 1
    assert "previous review comment does not record" in result.stderr
    assert "prompt:" not in result.stdout


@pytest.mark.parametrize("script_name", ["run_claude_review.sh", "run_codex_review.sh"])
def test_review_script_limits_rereview_prompt(script_name: str) -> None:
    """A re-review prompt identifies its commit range and allowed scope."""
    result = run_review_script(
        script_name,
        post_comment=True,
        existing_review_commits=(PREVIOUS_COMMIT,),
    )

    assert result.returncode == 0, result.stderr
    assert f"{PREVIOUS_COMMIT}..{HEAD_COMMIT}" in result.stdout
    assert "差分と前回の指摘だけ" in result.stdout
    assert "既存コードへの新しい指摘はしない" in result.stdout


@pytest.mark.parametrize("script_name", ["run_claude_review.sh", "run_codex_review.sh"])
def test_review_script_distinguishes_api_failure(script_name: str) -> None:
    """A GitHub API failure is reported separately from a missing comment."""
    result = run_review_script(
        script_name,
        post_comment=True,
        api_failure_after_review=True,
    )

    assert result.returncode == 2
    assert "failed to query review comments" in result.stderr
    assert "no new review comment was found" not in result.stderr
    assert "review-posted:" not in result.stdout


def run_post_comment_script(
    reviewed_commit: str | None,
) -> subprocess.CompletedProcess[str]:
    """Run the comment wrapper with a fake GitHub CLI."""
    body_file = Path("tmp") / f"review-body-{uuid.uuid4().hex}.md"
    body_file.parent.mkdir(parents=True, exist_ok=True)
    (REPOSITORY_ROOT / body_file).write_text("レビュー本文", encoding="utf-8")
    harness = r"""
gh() {
    case "$1 $2" in
        "repo view") printf '%s\n' 'ribuka/flat-pca' ;;
        "pr view") return 0 ;;
        "pr comment")
            while [ "$#" -gt 0 ]; do
                if [ "$1" = "--body-file" ]; then
                    cat "$2"
                    return 0
                fi
                shift
            done
            return 1
            ;;
        *) return 1 ;;
    esac
}

export -f gh
bash scripts/post_pr_comment.sh 123 "$REVIEW_BODY_FILE" claude
"""
    environment = {**os.environ, "REVIEW_BODY_FILE": body_file.as_posix()}
    if reviewed_commit is None:
        environment.pop("FLAT_PCA_REVIEWED_COMMIT", None)
    else:
        environment["FLAT_PCA_REVIEWED_COMMIT"] = reviewed_commit

    try:
        return subprocess.run(
            [find_bash(), "-c", harness],
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    finally:
        (REPOSITORY_ROOT / body_file).unlink(missing_ok=True)


def test_post_comment_records_reviewed_commit() -> None:
    """The trusted wrapper commit is appended to the review footer."""
    result = run_post_comment_script(HEAD_COMMIT)

    assert result.returncode == 0, result.stderr
    assert "_Reviewed by Claude. Posted via `scripts/post_pr_comment.sh`._" in result.stdout
    assert f"Reviewed commit: {HEAD_COMMIT}" in result.stdout


@pytest.mark.parametrize("reviewed_commit", [None, "not-a-sha"])
def test_post_comment_requires_reviewed_commit(reviewed_commit: str | None) -> None:
    """Posting is rejected without the wrapper's valid reviewed commit."""
    result = run_post_comment_script(reviewed_commit)

    assert result.returncode == 1
    assert "FLAT_PCA_REVIEWED_COMMIT must be set" in result.stderr
