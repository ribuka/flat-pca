"""Tests for the Ralph loop runner protocol."""

import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from scripts.ralph_runner import (
    ExitCode,
    _build_codex_command,
    _build_codex_environment,
    _classify_output,
    _commit_loop_changes,
    _finalize_log,
    _parse_args,
    _resolve_codex_executable,
    _run_codex,
    _task_progress,
    _task_snapshot,
    _validate_selected_task,
    run,
)


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("TASK_COMPLETED: TASK-001", ("completed", "TASK-001")),
        ("TASK_INCOMPLETE: TASK-002", ("incompleted", "TASK-002")),
        ("TASK_BLOCKED: TASK-003", ("blocked", "TASK-003")),
    ],
)
def test_classify_output_uses_last_non_empty_line(
    token: str,
    expected: tuple[str, str],
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
        "ALL_TASKS_COMPLETED",
    ],
)
def test_classify_output_rejects_invalid_terminal_status(message: str) -> None:
    """Reject missing, malformed, or non-terminal Ralph status tokens."""
    with pytest.raises(ValueError, match="invalid Ralph status line"):
        _classify_output(message)


def test_validate_selected_task_rejects_a_different_reported_task() -> None:
    """Reject a terminal token for a task other than the pre-launch selection."""
    with pytest.raises(ValueError, match="reported TASK-003, expected TASK-002"):
        _validate_selected_task("completed", "TASK-003", "TASK-002")


def test_task_progress_counts_uncompleted_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Count every task while treating only completed status as resolved."""
    task_file = Path("TASKS.md")
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, *, encoding: """## TASK-001: first

- Status: completed
- Priority: 1
- Depends on: none

## TASK-002: second

- Status: pending
- Priority: 2
- Depends on: TASK-001

## TASK-003: third

- Status: blocked
- Priority: 3
- Depends on: TASK-002""",
    )

    assert _task_progress(task_file) == (2, 3)


def test_task_snapshot_selects_by_dependencies_priority_and_id(tmp_path: Path) -> None:
    """Select the lowest-priority eligible task and use its ID as a tie-breaker."""
    task_file = tmp_path / "TASKS.md"
    task_file.write_text(
        """## TASK-001: completed dependency

- Status: completed
- Priority: 1
- Depends on: none

## TASK-004: blocked by an incomplete dependency

- Status: pending
- Priority: 1
- Depends on: TASK-003

## TASK-003: eligible later ID

- Status: pending
- Priority: 2
- Depends on: TASK-001

## TASK-002: eligible earlier ID

- Status: pending
- Priority: 2
- Depends on: TASK-001
""",
        encoding="utf-8",
    )

    assert _task_snapshot(task_file) == (3, 4, "TASK-002")


def test_run_logs_runner_progress_loop_and_selected_task_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Log the agreed lifecycle and selected task before a dry-run command."""
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("one loop", encoding="utf-8")
    (tmp_path / "TASKS.md").write_text(
        """## TASK-001: completed

- Status: completed
- Priority: 1
- Depends on: none

## TASK-002: next

- Status: pending
- Priority: 2
- Depends on: TASK-001
""",
        encoding="utf-8",
    )
    messages: list[str] = []

    def fake_git_output(repo: Path, *args: str) -> str:
        """Return the values required by dry-run Git preflight."""
        if args == ("rev-parse", "--show-toplevel"):
            return str(repo)
        if args == ("rev-parse", "HEAD"):
            return "before"
        raise AssertionError(args)

    def record_info(message: str, *args: object) -> None:
        """Record a rendered Loguru-style informational message."""
        messages.append(message.format(*args))

    monkeypatch.setattr("scripts.ralph_runner._resolve_codex_executable", lambda value: value)
    monkeypatch.setattr("scripts.ralph_runner._require_git_output", fake_git_output)
    monkeypatch.setattr("scripts.ralph_runner._require_clean_worktree", lambda repo: None)
    monkeypatch.setattr("scripts.ralph_runner._build_codex_environment", lambda repo: {})
    monkeypatch.setattr("scripts.ralph_runner.logger.info", record_info)

    result = run(
        repo=tmp_path,
        prompt_path=prompt_path,
        max_loops=3,
        codex_executable="codex",
        dry_run=True,
    )

    assert result == 0
    assert messages == [
        "Ralph runner start",
        "Incompleted tasks: 1 / All tasks: 2",
        "Total loops: 3",
        "Ralph loop start (1)",
        "Ralph task TASK-002 started",
        "Ralph runner end",
    ]


def test_run_does_not_start_codex_when_all_tasks_are_already_complete(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Return success without starting a loop for an already-complete ledger."""
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("one loop", encoding="utf-8")
    (tmp_path / "TASKS.md").write_text(
        """## TASK-001: completed

- Status: completed
- Priority: 1
- Depends on: none
""",
        encoding="utf-8",
    )
    success_messages: list[str] = []

    def fake_git_output(repo: Path, *args: str) -> str:
        """Return the Git root required by preflight."""
        if args == ("rev-parse", "--show-toplevel"):
            return str(repo)
        raise AssertionError(args)

    monkeypatch.setattr("scripts.ralph_runner._resolve_codex_executable", lambda value: value)
    monkeypatch.setattr("scripts.ralph_runner._require_git_output", fake_git_output)
    monkeypatch.setattr("scripts.ralph_runner._require_clean_worktree", lambda repo: None)
    monkeypatch.setattr(
        "scripts.ralph_runner._run_codex",
        lambda *args, **kwargs: pytest.fail("Codex must not be started"),
    )
    monkeypatch.setattr(
        "scripts.ralph_runner.logger.success",
        lambda message, *args: success_messages.append(message.format(*args)),
    )

    result = run(
        repo=tmp_path,
        prompt_path=prompt_path,
        max_loops=3,
        codex_executable="codex",
    )

    assert result == ExitCode.SUCCESS
    assert success_messages == ["All Ralph tasks are complete"]
    assert list((tmp_path / "logs").iterdir()) == []


def test_run_does_not_start_codex_when_no_incomplete_task_is_eligible(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Return blocked without starting a loop when dependencies prevent selection."""
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("one loop", encoding="utf-8")
    (tmp_path / "TASKS.md").write_text(
        """## TASK-001: blocked dependency

- Status: blocked
- Priority: 1
- Depends on: none

## TASK-002: dependent task

- Status: pending
- Priority: 2
- Depends on: TASK-001
""",
        encoding="utf-8",
    )
    warning_messages: list[str] = []

    def fake_git_output(repo: Path, *args: str) -> str:
        """Return the Git root required by preflight."""
        if args == ("rev-parse", "--show-toplevel"):
            return str(repo)
        raise AssertionError(args)

    monkeypatch.setattr("scripts.ralph_runner._resolve_codex_executable", lambda value: value)
    monkeypatch.setattr("scripts.ralph_runner._require_git_output", fake_git_output)
    monkeypatch.setattr("scripts.ralph_runner._require_clean_worktree", lambda repo: None)
    monkeypatch.setattr(
        "scripts.ralph_runner._run_codex",
        lambda *args, **kwargs: pytest.fail("Codex must not be started"),
    )
    monkeypatch.setattr(
        "scripts.ralph_runner.logger.warning",
        lambda message, *args: warning_messages.append(message.format(*args)),
    )

    result = run(
        repo=tmp_path,
        prompt_path=prompt_path,
        max_loops=3,
        codex_executable="codex",
    )

    assert result == ExitCode.TASK_BLOCKED
    assert warning_messages == ["Stopped because no incomplete Ralph task is eligible"]
    assert list((tmp_path / "logs").iterdir()) == []


def test_run_stops_after_completing_the_last_task(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Complete the final task without starting a redundant confirmation loop."""
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("one loop", encoding="utf-8")
    tasks_path = tmp_path / "TASKS.md"
    tasks_path.write_text(
        """## TASK-009: final task

- Status: pending
- Priority: 1
- Depends on: none
""",
        encoding="utf-8",
    )
    commit_created = False
    codex_calls = 0
    success_messages: list[str] = []

    def fake_git_output(repo: Path, *args: str) -> str:
        """Return deterministic repository states around the runner commit."""
        if args == ("rev-parse", "--show-toplevel"):
            return str(repo)
        if args == ("rev-parse", "HEAD"):
            return "after" if commit_created else "before"
        raise AssertionError(args)

    def fake_commit_loop_changes(repo: Path, task_id: str, status: str) -> None:
        """Record the single runner-owned commit."""
        nonlocal commit_created
        assert repo == tmp_path
        assert task_id == "TASK-009"
        assert status == "completed"
        commit_created = True

    def fake_run_codex(
        command: list[str],
        repo: Path,
        log_path: Path,
        environment: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        """Complete the ledger task and emit its valid terminal token."""
        nonlocal codex_calls
        del environment
        codex_calls += 1
        assert repo == tmp_path
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text("TASK_COMPLETED: TASK-009\n", encoding="utf-8")
        log_path.write_text("codex output\n", encoding="utf-8")
        tasks_path.write_text(
            tasks_path.read_text(encoding="utf-8").replace(
                "- Status: pending",
                "- Status: completed",
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("scripts.ralph_runner._resolve_codex_executable", lambda value: value)
    monkeypatch.setattr("scripts.ralph_runner._require_git_output", fake_git_output)
    monkeypatch.setattr("scripts.ralph_runner._require_clean_worktree", lambda repo: None)
    monkeypatch.setattr("scripts.ralph_runner._build_codex_environment", lambda repo: {})
    monkeypatch.setattr("scripts.ralph_runner._run_codex", fake_run_codex)
    monkeypatch.setattr("scripts.ralph_runner._commit_loop_changes", fake_commit_loop_changes)
    monkeypatch.setattr(
        "scripts.ralph_runner._commit_count",
        lambda repo, before, after: int(before != after),
    )
    monkeypatch.setattr(
        "scripts.ralph_runner.logger.success",
        lambda message, *args: success_messages.append(message.format(*args)),
    )

    result = run(
        repo=tmp_path,
        prompt_path=prompt_path,
        max_loops=3,
        codex_executable="codex",
    )

    assert result == ExitCode.SUCCESS
    assert codex_calls == 1
    assert success_messages[0].startswith("Completed TASK-009 (logs")
    assert success_messages[1] == "All Ralph tasks are complete"
    assert "starting the next loop" not in "\n".join(success_messages)
    log_names = [path.name for path in (tmp_path / "logs").iterdir()]
    assert len(log_names) == 1
    assert log_names[0].endswith("_009_completed.log")


def test_run_retries_protocol_failure_and_preserves_attempt_log(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Retry an invalid Codex response without consuming another Ralph loop."""
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("one loop", encoding="utf-8")
    tasks_path = tmp_path / "TASKS.md"
    tasks_path.write_text(
        """## TASK-010: retryable task

- Status: pending
- Priority: 1
- Depends on: none
""",
        encoding="utf-8",
    )
    codex_calls = 0
    commit_created = False
    delays: list[int] = []

    def fake_git_output(repo: Path, *args: str) -> str:
        """Return deterministic repository states around the runner commit."""
        if args == ("rev-parse", "--show-toplevel"):
            return str(repo)
        if args == ("rev-parse", "HEAD"):
            return "after" if commit_created else "before"
        raise AssertionError(args)

    def fake_run_codex(
        command: list[str],
        repo: Path,
        log_path: Path,
        environment: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        """Emit an invalid response once, then complete the selected task."""
        nonlocal codex_calls
        del environment
        codex_calls += 1
        assert repo == tmp_path
        output_path = Path(command[command.index("--output-last-message") + 1])
        log_path.write_text(f"attempt {codex_calls}\n", encoding="utf-8")
        if codex_calls == 1:
            output_path.write_text("connection interrupted\n", encoding="utf-8")
        else:
            output_path.write_text("TASK_COMPLETED: TASK-010\n", encoding="utf-8")
            tasks_path.write_text(
                tasks_path.read_text(encoding="utf-8").replace(
                    "- Status: pending",
                    "- Status: completed",
                ),
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(command, 0)

    def fake_commit_loop_changes(repo: Path, task_id: str, status: str) -> None:
        """Record the runner-owned task completion commit."""
        nonlocal commit_created
        assert repo == tmp_path
        assert task_id == "TASK-010"
        assert status == "completed"
        commit_created = True

    monkeypatch.setattr("scripts.ralph_runner._resolve_codex_executable", lambda value: value)
    monkeypatch.setattr("scripts.ralph_runner._require_git_output", fake_git_output)
    monkeypatch.setattr("scripts.ralph_runner._require_clean_worktree", lambda repo: None)
    monkeypatch.setattr("scripts.ralph_runner._build_codex_environment", lambda repo: {})
    monkeypatch.setattr("scripts.ralph_runner._run_codex", fake_run_codex)
    monkeypatch.setattr("scripts.ralph_runner._commit_loop_changes", fake_commit_loop_changes)
    monkeypatch.setattr(
        "scripts.ralph_runner._commit_count",
        lambda repo, before, after: int(before != after),
    )
    monkeypatch.setattr("scripts.ralph_runner.time.sleep", delays.append)

    result = run(
        repo=tmp_path,
        prompt_path=prompt_path,
        max_loops=1,
        codex_executable="codex",
        api_retry_count=1,
        api_retry_interval_sec=7,
    )

    assert result == ExitCode.SUCCESS
    assert codex_calls == 2
    assert delays == [7]
    log_names = sorted(path.name for path in (tmp_path / "logs").iterdir())
    assert any(name.endswith("_000_api-retry-001.log") for name in log_names)
    assert any(name.endswith("_010_completed.log") for name in log_names)


def test_parse_args_uses_api_retry_defaults_and_overrides() -> None:
    """Expose API retry configuration with the documented defaults."""
    defaults = _parse_args([])
    overrides = _parse_args(
        ["--api-retry-count", "3", "--api-retry-interval-sec", "9"]
    )

    assert defaults.api_retry_count == 10
    assert defaults.api_retry_interval_sec == 5
    assert overrides.api_retry_count == 3
    assert overrides.api_retry_interval_sec == 9


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


def test_build_codex_environment_uses_repository_temporary_directories(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Direct child-process caches and temporary files into the repository."""
    monkeypatch.setenv("UV_CACHE_DIR", "outside-uv-cache")
    monkeypatch.setenv("TMP", "outside-tmp")
    monkeypatch.setenv("TEMP", "outside-temp")

    environment = _build_codex_environment(tmp_path)
    next_environment = _build_codex_environment(tmp_path)

    assert environment["UV_CACHE_DIR"] == str((tmp_path / "tmp" / "uv-cache").resolve())
    runtime_directory = Path(environment["TMP"])
    assert runtime_directory.parent == (tmp_path / "tmp" / "runtime").resolve()
    assert runtime_directory.name.startswith("ralph-")
    assert environment["TEMP"] == environment["TMP"]
    assert next_environment["TMP"] != environment["TMP"]
    assert (tmp_path / "tmp" / "uv-cache").is_dir()
    assert runtime_directory.is_dir()


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
        assert kwargs["env"] is environment
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("scripts.ralph_runner.subprocess.run", fake_run)
    log_path = tmp_path / "loop_001.log"
    environment = {"UV_CACHE_DIR": "repo-cache"}

    completed = _run_codex(["codex", "exec"], tmp_path, log_path, environment)

    assert completed.returncode == 0
    assert log_path.read_text(encoding="utf-8") == "standard output\nstandard error\n"


def test_finalize_log_uses_timestamp_task_number_and_status(tmp_path: Path) -> None:
    """Name a completed task log using the documented final filename format."""
    temporary_path = tmp_path / ".ralph-running.log"
    temporary_path.write_text("loop output", encoding="utf-8")

    final_path = _finalize_log(
        temporary_path,
        tmp_path,
        datetime(2026, 9, 15, 20, 11, 19, tzinfo=UTC),
        "TASK-002",
        "completed",
    )

    assert final_path.name == "ralph_20260915T201119_002_completed.log"
    assert final_path.read_text(encoding="utf-8") == "loop output"
    assert not temporary_path.exists()


def test_finalize_log_keeps_logs_when_a_name_already_exists(tmp_path: Path) -> None:
    """Advance the timestamp to preserve a log with an identical base name."""
    timestamp = datetime(2026, 9, 15, 20, 11, 19, tzinfo=UTC)
    existing_path = tmp_path / "ralph_20260915T201119_002_completed.log"
    existing_path.write_text("first run", encoding="utf-8")
    temporary_path = tmp_path / ".ralph-running.log"
    temporary_path.write_text("second run", encoding="utf-8")

    final_path = _finalize_log(
        temporary_path,
        tmp_path,
        timestamp,
        "TASK-002",
        "completed",
    )

    assert final_path.name == "ralph_20260915T201120_002_completed.log"
    assert existing_path.read_text(encoding="utf-8") == "first run"
    assert final_path.read_text(encoding="utf-8") == "second run"


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

    _commit_loop_changes(Path("repo"), "TASK-001", "completed")

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

    _commit_loop_changes(Path("repo"), "TASK-002", "blocked")

    assert commands[-1] == ("commit", "-m", "wip(TASK-002): Ralph loop changes")
