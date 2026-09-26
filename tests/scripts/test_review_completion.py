"""Review wrapper completion reporting tests."""

from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


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
    existing_comment: bool = False,
    api_failure_after_review: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a review wrapper with deterministic fake CLI and GitHub commands."""
    state_name = f"review-test-{uuid.uuid4().hex}"
    state_file = Path("tmp") / f"{state_name}.state"
    log_file = Path("tmp") / f"{state_name}.jsonl"
    harness = r"""
gh() {
    case "$1 $2" in
        "repo view") printf '%s\n' 'ribuka/flat-pca' ;;
        "pr view") return 0 ;;
        "api --paginate")
            if [ "$REVIEW_API_FAILURE_AFTER" = "true" ] && [ -f "$REVIEW_STATE_FILE" ]; then
                return 22
            fi
            if [ -f "$REVIEW_STATE_FILE" ]; then
                printf '1001\thttps://github.com/ribuka/flat-pca/pull/123#issuecomment-1001\n'
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
    post_fake_review
}

codex() {
    post_fake_review
}

export -f gh post_fake_review claude codex
export REVIEW_STATE_FILE REVIEW_POST_COMMENT

if [ "$REVIEW_EXISTING_COMMENT" = "true" ]; then
    mkdir -p "$(dirname "$REVIEW_STATE_FILE")"
    : >"$REVIEW_STATE_FILE"
fi

review_args=("$REVIEW_SCRIPT" 123)
if [ "$REVIEW_SCRIPT" = "scripts/run_codex_review.sh" ]; then
    review_args+=(--log-file "$REVIEW_LOG_FILE")
fi

bash "${review_args[@]}"
"""
    environment = {
        "REVIEW_SCRIPT": f"scripts/{script_name}",
        "REVIEW_STATE_FILE": state_file.as_posix(),
        "REVIEW_LOG_FILE": log_file.as_posix(),
        "REVIEW_POST_COMMENT": str(post_comment).lower(),
        "REVIEW_EXISTING_COMMENT": str(existing_comment).lower(),
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
        )
    finally:
        (REPOSITORY_ROOT / state_file).unlink(missing_ok=True)
        (REPOSITORY_ROOT / log_file).unlink(missing_ok=True)


@pytest.mark.parametrize("script_name", ["run_claude_review.sh", "run_codex_review.sh"])
def test_review_script_reports_new_comment_url(script_name: str) -> None:
    """A newly posted review comment is reported as successful completion."""
    result = run_review_script(script_name, post_comment=True)

    assert result.returncode == 0, result.stderr
    assert result.stdout.endswith(
        "review-posted: https://github.com/ribuka/flat-pca/pull/123#issuecomment-1001\n"
    )


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
        existing_comment=True,
    )

    assert result.returncode == 1
    assert "no new review comment was found" in result.stderr
    assert "review-posted:" not in result.stdout


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
