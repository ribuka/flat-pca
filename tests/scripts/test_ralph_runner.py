"""Tests for the Ralph loop runner protocol."""

import subprocess
from pathlib import Path

import pytest

from scripts.ralph_runner import (
    _build_codex_command,
    _classify_output,
    _commit_loop_changes,
    _resolve_codex_executable,
    _run_codex,
)


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("TASK_COMPLETED: TASK-001", ("task_completed", "TASK-001")),
        ("TASK_INCOMPLETE: TASK-002", ("task_incomplete", "TASK-002")),
        ("TASK_BLOCKED: TASK-003", ("task_blocked", "TASK-003")),
        ("ALL_TASKS_COMPLETED", ("all_completed", None)),
    ],
)
def test_classify_output_uses_last_non_empty_line(
    token: str,
    expected: tuple[str, str | None],
) -> None:
    """Recognize each valid terminal status after a human-readable summary."""
    message = f"Loop summary\n\n{token}\n"
    assert _classify_output(message) == expected


@pytest.mark.parametrize(
    "message",
    [
        "",
        "TASK_COMPLETED: task-001",
        "TASK_COMPLETED: TASK-1",
        "TASK_COMPLETED: TASK-001 trailing text",
        "TASK_COMPLETED: TASK-001\nsummary after token",
    ],
)
def test_classify_output_rejects_invalid_terminal_status(message: str) -> None:
    """Reject missing, malformed, or non-terminal Ralph status tokens."""
    with pytest.raises(ValueError, match="invalid Ralph status line"):
        _classify_output(message)


def test_resolve_codex_executable_uses_which_absolute_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the path resolved during preflight to launch the Codex subprocess."""
    executable_path = Path("C:/tools/codex.exe")
    monkeypatch.setattr(
        "scripts.ralph_runner.shutil.which",
        lambda executable: str(executable_path),
    )

    assert _resolve_codex_executable("codex") == str(executable_path.resolve())


def test_build_codex_command_requires_confirmation_by_default() -> None:
    """Use the workspace-write sandbox without automatic approval by default."""
    command = _build_codex_command(
        executable="codex",
        repo=Path("repo"),
        output_path=Path("tmp/last-message.txt"),
        prompt="one loop",
        model="test-model",
    )

    assert command == [
        "codex",
        "exec",
        "--ephemeral",
        "--cd",
        "repo",
        "--output-last-message",
        str(Path("tmp/last-message.txt")),
        "--sandbox",
        "workspace-write",
        "--model",
        "test-model",
        "one loop",
    ]


def test_build_codex_command_auto_approves_only_when_requested() -> None:
    """Use automatic approval without a conflicting sandbox argument."""
    command = _build_codex_command(
        executable="codex",
        repo=Path("repo"),
        output_path=Path("tmp/last-message.txt"),
        prompt="one loop",
        model=None,
        auto_approve=True,
    )

    assert command == [
        "codex",
        "exec",
        "--ephemeral",
        "--cd",
        "repo",
        "--output-last-message",
        str(Path("tmp/last-message.txt")),
        "--approve-for-me",
        "one loop",
    ]


def test_run_codex_writes_combined_output_to_loop_log(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep Codex output out of the terminal while preserving it in the log."""

    def fake_run(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        """Write representative Codex output to the configured log stream."""
        stdout = kwargs["stdout"]
        assert hasattr(stdout, "write")
        stdout.write("standard output\nstandard error\n")  # type: ignore[union-attr]
        assert kwargs["stderr"] is subprocess.STDOUT
        assert kwargs["cwd"] == tmp_path
        assert kwargs["check"] is False
        assert kwargs["text"] is True
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("scripts.ralph_runner.subprocess.run", fake_run)
    log_path = tmp_path / "loop_001.log"

    completed = _run_codex(["codex", "exec"], tmp_path, log_path)

    assert completed.returncode == 0
    assert log_path.read_text(encoding="utf-8") == "standard output\nstandard error\n"


def test_commit_loop_changes_stages_and_commits_completed_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Commit the changes from a clean-start completed loop once."""
    commands: list[tuple[str, ...]] = []

    def fake_run_git(
        repo: Path,
        *args: str,
    ) -> subprocess.CompletedProcess[str]:
        """Record Git calls and report staged changes after ``git add``."""
        del repo
        commands.append(args)
        returncode = 1 if args == ("diff", "--cached", "--quiet") else 0
        return subprocess.CompletedProcess(["git", *args], returncode, "", "")

    monkeypatch.setattr("scripts.ralph_runner._run_git", fake_run_git)

    _commit_loop_changes(Path("repo"), "TASK-001", "task_completed")

    assert commands == [
        ("diff", "--check"),
        ("add", "--all"),
        ("diff", "--cached", "--quiet"),
        ("diff", "--cached", "--check"),
        ("commit", "-m", "feat(TASK-001): Ralph loop changes"),
    ]


def test_commit_loop_changes_uses_wip_subject_for_blocked_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve blocked-loop progress in a WIP commit."""
    commands: list[tuple[str, ...]] = []

    def fake_run_git(
        repo: Path,
        *args: str,
    ) -> subprocess.CompletedProcess[str]:
        """Report staged changes after ``git add``."""
        del repo
        commands.append(args)
        returncode = 1 if args == ("diff", "--cached", "--quiet") else 0
        return subprocess.CompletedProcess(["git", *args], returncode, "", "")

    monkeypatch.setattr("scripts.ralph_runner._run_git", fake_run_git)

    _commit_loop_changes(Path("repo"), "TASK-002", "task_blocked")

    assert commands[-1] == ("commit", "-m", "wip(TASK-002): Ralph loop changes")
