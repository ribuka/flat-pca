"""Run Ralph tasks repeatedly through the non-interactive Codex CLI."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import IntEnum
from pathlib import Path

from loguru import logger

from scripts.console_logging import configure_console_logging

DEFAULT_MAX_LOOPS = 20
DEFAULT_API_RETRY_COUNT = 10
DEFAULT_API_RETRY_INTERVAL_SEC = 5
DEFAULT_CODEX_TIMEOUT_SEC = 1_800
TASK_COMPLETED_PATTERN = re.compile(r"TASK_COMPLETED: (TASK-\d{3})")
TASK_INCOMPLETE_PATTERN = re.compile(r"TASK_INCOMPLETE: (TASK-\d{3})")
TASK_BLOCKED_PATTERN = re.compile(r"TASK_BLOCKED: (TASK-\d{3})")
TASK_SECTION_PATTERN = re.compile(
    r"^## (?P<task_id>TASK-\d{3}):.*?$"
    r"(?P<body>.*?)(?=^## TASK-\d{3}:|\Z)",
    re.MULTILINE | re.DOTALL,
)
TASK_ID_PATTERN = re.compile(r"TASK-\d{3}")
TASK_STATUS_FIELD_PATTERN = re.compile(r"^- Status: (?P<value>\w+)$", re.MULTILINE)
TASK_PRIORITY_FIELD_PATTERN = re.compile(r"^- Priority: (?P<value>\d+)$", re.MULTILINE)
TASK_DEPENDENCIES_FIELD_PATTERN = re.compile(
    r"^- Depends on: (?P<value>[^\r\n]+)$",
    re.MULTILINE,
)
TASK_STATUSES = frozenset({"pending", "completed", "blocked"})


@dataclass(frozen=True)
class _RalphTask:
    """Task metadata required for Ralph progress and selection.

    Attributes
    ----------
    task_id : str
        Stable ``TASK-XXX`` identifier.
    status : str
        Current task ledger status.
    priority : int
        Numeric task selection priority.
    dependencies : tuple[str, ...]
        Task identifiers that must be completed first.
    """

    task_id: str
    status: str
    priority: int
    dependencies: tuple[str, ...]


class ExitCode(IntEnum):
    """Process exit codes produced by the Ralph runner."""

    SUCCESS = 0
    CODEX_FAILURE = 10
    PROTOCOL_ERROR = 11
    GIT_STATE_ERROR = 12
    TASK_INCOMPLETE = 20
    TASK_BLOCKED = 21
    MAX_LOOPS_REACHED = 22
    PREFLIGHT_ERROR = 23


def _run_git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run a Git command and capture its text output.

    Parameters
    ----------
    repo : Path
        Repository working directory.
    *args : str
        Arguments passed to Git.

    Returns
    -------
    subprocess.CompletedProcess[str]
        Completed Git process.
    """
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )


def _require_git_output(repo: Path, *args: str) -> str:
    """Return stripped Git output or raise a runtime error.

    Parameters
    ----------
    repo : Path
        Repository working directory.
    *args : str
        Arguments passed to Git.

    Returns
    -------
    str
        Stripped standard output.

    Raises
    ------
    RuntimeError
        If Git exits unsuccessfully.
    """
    result = _run_git(repo, *args)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(detail or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def _require_clean_worktree(repo: Path) -> None:
    """Raise when the repository contains tracked or untracked changes.

    Parameters
    ----------
    repo : Path
        Repository working directory.

    Raises
    ------
    RuntimeError
        If the working tree is not clean.
    """
    status = _require_git_output(repo, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise RuntimeError("working tree is not clean:\n" + status)


def _commit_count(repo: Path, before: str, after: str) -> int:
    """Count commits added between two repository states.

    Parameters
    ----------
    repo : Path
        Repository working directory.
    before : str
        Commit hash before the loop.
    after : str
        Commit hash after the loop.

    Returns
    -------
    int
        Number of commits reachable from ``after`` but not ``before``.

    Raises
    ------
    RuntimeError
        If history was rewritten or Git cannot inspect the range.
    """
    ancestor = _run_git(repo, "merge-base", "--is-ancestor", before, after)
    if ancestor.returncode != 0:
        raise RuntimeError("the loop rewrote or replaced repository history")
    return int(_require_git_output(repo, "rev-list", "--count", f"{before}..{after}"))


def _commit_loop_changes(repo: Path, task_id: str, status: str) -> None:
    """Commit all changes produced by one clean-start Ralph loop.

    Parameters
    ----------
    repo : Path
        Repository working directory, verified clean before the loop started.
    task_id : str
        Ralph task identifier reported by Codex.
    status : str
        Terminal Ralph status, either ``completed``, ``incompleted``, or
        ``blocked``.

    Raises
    ------
    RuntimeError
        If the working tree contains no changes or Git cannot stage or commit
        the loop changes.
    ValueError
        If ``status`` is not a task terminal status.
    """
    subjects = {
        "completed": f"feat({task_id}): Ralph loop changes",
        "incompleted": f"wip({task_id}): Ralph loop changes",
        "blocked": f"wip({task_id}): Ralph loop changes",
    }
    try:
        subject = subjects[status]
    except KeyError as error:
        raise ValueError(f"cannot commit loop status: {status}") from error

    _require_git_output(repo, "diff", "--check")
    _require_git_output(repo, "add", "--all")
    staged = _run_git(repo, "diff", "--cached", "--quiet")
    if staged.returncode == 0:
        raise RuntimeError("task loop produced no changes to commit")
    if staged.returncode != 1:
        detail = staged.stderr.strip() or staged.stdout.strip()
        raise RuntimeError(detail or "could not inspect staged changes")
    _require_git_output(repo, "diff", "--cached", "--check")
    _require_git_output(repo, "commit", "-m", subject)


def _last_non_empty_line(message: str) -> str:
    """Return the last non-empty line from an agent message.

    Parameters
    ----------
    message : str
        Final Codex agent message.

    Returns
    -------
    str
        Last non-empty stripped line, or an empty string.
    """
    return next((line.strip() for line in reversed(message.splitlines()) if line.strip()), "")


def _classify_output(message: str) -> tuple[str, str]:
    """Classify the Ralph protocol token in a final agent message.

    Parameters
    ----------
    message : str
        Final Codex agent message.

    Returns
    -------
    tuple[str, str]
        Protocol status and task ID.

    Raises
    ------
    ValueError
        If the last non-empty line is not a valid Ralph status token.
    """
    token = _last_non_empty_line(message)
    for status, pattern in (
        ("completed", TASK_COMPLETED_PATTERN),
        ("incompleted", TASK_INCOMPLETE_PATTERN),
        ("blocked", TASK_BLOCKED_PATTERN),
    ):
        match = pattern.fullmatch(token)
        if match:
            return status, match.group(1)

    raise ValueError(f"invalid Ralph status line: {token!r}")


def _validate_selected_task(
    status: str,
    reported_task_id: str,
    selected_task_id: str,
) -> None:
    """Validate that Codex processed the task selected before its launch.

    Parameters
    ----------
    status : str
        Classified Ralph terminal status.
    reported_task_id : str
        Task identifier reported by Codex.
    selected_task_id : str
        Task identifier selected from the pre-launch ledger.

    Raises
    ------
    ValueError
        If the terminal status contradicts the pre-launch selection.
    """
    if reported_task_id != selected_task_id:
        raise ValueError(
            f"Codex reported {reported_task_id}, expected {selected_task_id}"
        )


def _require_task_field(pattern: re.Pattern[str], body: str, field: str, task_id: str) -> str:
    """Return one required task field from a task section.

    Parameters
    ----------
    pattern : re.Pattern[str]
        Compiled pattern containing a named ``value`` group.
    body : str
        Markdown content belonging to one task.
    field : str
        Human-readable field name for an error message.
    task_id : str
        Identifier of the task being parsed.

    Returns
    -------
    str
        Parsed field value.

    Raises
    ------
    ValueError
        If the required field is absent.
    """
    match = pattern.search(body)
    if match is None:
        raise ValueError(f"{task_id} has no {field} field")
    return match.group("value")


def _read_ralph_tasks(tasks_path: Path) -> tuple[_RalphTask, ...]:
    """Read task metadata used by the Ralph runner.

    Parameters
    ----------
    tasks_path : Path
        UTF-8 task definition file containing task headings and status fields.

    Returns
    -------
    tuple[_RalphTask, ...]
        Parsed task metadata in ledger order.

    Raises
    ------
    ValueError
        If task metadata is missing, duplicated, or invalid.
    """
    text = tasks_path.read_text(encoding="utf-8")
    tasks: list[_RalphTask] = []
    for section in TASK_SECTION_PATTERN.finditer(text):
        task_id = section.group("task_id")
        body = section.group("body")
        status = _require_task_field(
            TASK_STATUS_FIELD_PATTERN,
            body,
            "Status",
            task_id,
        )
        if status not in TASK_STATUSES:
            raise ValueError(f"{task_id} has invalid status: {status}")
        priority = int(
            _require_task_field(
                TASK_PRIORITY_FIELD_PATTERN,
                body,
                "Priority",
                task_id,
            )
        )
        dependency_field = _require_task_field(
            TASK_DEPENDENCIES_FIELD_PATTERN,
            body,
            "Depends on",
            task_id,
        )
        dependencies = (
            ()
            if dependency_field == "none"
            else tuple(value.strip() for value in dependency_field.split(","))
        )
        if any(TASK_ID_PATTERN.fullmatch(value) is None for value in dependencies):
            raise ValueError(f"{task_id} has invalid dependencies: {dependency_field}")
        tasks.append(_RalphTask(task_id, status, priority, dependencies))

    if not tasks:
        raise ValueError(f"no tasks found: {tasks_path}")
    task_ids = {task.task_id for task in tasks}
    if len(task_ids) != len(tasks):
        raise ValueError(f"duplicate task IDs found: {tasks_path}")
    missing_dependencies = {
        dependency
        for task in tasks
        for dependency in task.dependencies
        if dependency not in task_ids
    }
    if missing_dependencies:
        missing = ", ".join(sorted(missing_dependencies))
        raise ValueError(f"unknown task dependencies: {missing}")
    return tuple(tasks)


def _task_snapshot(tasks_path: Path) -> tuple[int, int, str | None]:
    """Return task counts and the next eligible task identifier.

    Parameters
    ----------
    tasks_path : Path
        UTF-8 Ralph task ledger.

    Returns
    -------
    tuple[int, int, str | None]
        Incomplete count, total count, and next eligible task identifier.
    """
    tasks = _read_ralph_tasks(tasks_path)
    statuses = {task.task_id: task.status for task in tasks}
    candidates = [
        task
        for task in tasks
        if task.status == "pending"
        and all(statuses[dependency] == "completed" for dependency in task.dependencies)
    ]
    selected = min(candidates, key=lambda task: (task.priority, task.task_id), default=None)
    return (
        sum(task.status != "completed" for task in tasks),
        len(tasks),
        selected.task_id if selected is not None else None,
    )


def _task_progress(tasks_path: Path) -> tuple[int, int]:
    """Return incomplete and total task counts from a Ralph task file.

    Parameters
    ----------
    tasks_path : Path
        UTF-8 Ralph task ledger.

    Returns
    -------
    tuple[int, int]
        Number of incomplete tasks and total task count.
    """
    incomplete, total, _ = _task_snapshot(tasks_path)
    return incomplete, total


def _resolve_codex_executable(executable: str) -> str | None:
    """Resolve a Codex executable name to an absolute executable path.

    Parameters
    ----------
    executable : str
        Codex executable name or path.

    Returns
    -------
    str | None
        Absolute executable path, or ``None`` when it cannot be found.
    """
    executable_path = Path(executable)
    if executable_path.is_file():
        return str(executable_path.resolve())

    resolved_path = shutil.which(executable)
    if resolved_path is None:
        return None
    return str(Path(resolved_path).resolve())


def _build_codex_command(
    executable: str,
    repo: Path,
    output_path: Path,
    prompt: str,
    model: str | None,
    auto_approve: bool = False,
) -> list[str]:
    """Build the non-interactive Codex command for one loop.

    Parameters
    ----------
    executable : str
        Codex executable name or path.
    repo : Path
        Repository working directory.
    output_path : Path
        File receiving the final Codex message.
    prompt : str
        Prompt for one Ralph loop.
    model : str | None
        Optional model override.
    auto_approve : bool, default False
        Automatically approve Codex requests in the workspace-write sandbox.

    Returns
    -------
    list[str]
        Subprocess argument vector.
    """
    command = [
        executable,
        "exec",
        "--ephemeral",
        "--disable",
        "unbounded_connection_retries",
        "--cd",
        str(repo),
        "--output-last-message",
        str(output_path),
    ]
    if auto_approve:
        command.append("--approve-for-me")
    else:
        command.extend(
            [
                "--sandbox",
                "workspace-write",
                "--ask-for-approval",
                "on-request",
            ]
        )
    if model is not None:
        command.extend(["--model", model])
    command.append(prompt)
    return command


def _build_codex_environment(repo: Path) -> dict[str, str]:
    """Build a child-process environment with repository-local cache and temporary paths.

    Parameters
    ----------
    repo : Path
        Repository root containing the Ralph ``tmp`` directory and uv cache.

    Returns
    -------
    dict[str, str]
        Copy of the current environment with the uv cache and runtime temporary
        paths directed into the repository.
    """
    temporary_directory = (repo / "tmp").resolve()
    uv_cache_directory = (repo / ".uv-cache").resolve()
    runtime_root = temporary_directory / "runtime"
    uv_cache_directory.mkdir(parents=True, exist_ok=True)
    runtime_root.mkdir(parents=True, exist_ok=True)
    runtime_directory = Path(tempfile.mkdtemp(prefix="ralph-", dir=runtime_root))

    environment = os.environ.copy()
    environment["UV_CACHE_DIR"] = str(uv_cache_directory)
    environment["TMP"] = str(runtime_directory)
    environment["TEMP"] = str(runtime_directory)
    return environment


def _run_codex(
    command: Sequence[str],
    repo: Path,
    log_path: Path,
    environment: dict[str, str],
    timeout_sec: int = DEFAULT_CODEX_TIMEOUT_SEC,
) -> subprocess.CompletedProcess[str]:
    """Run Codex and write its combined output to a loop log.

    Parameters
    ----------
    command : Sequence[str]
        Codex subprocess argument vector.
    repo : Path
        Repository working directory.
    log_path : Path
        File receiving Codex standard output and standard error.
    environment : dict[str, str]
        Environment passed to the Codex child process.
    timeout_sec : int, default 1800
        Maximum time to wait for the Codex child process.

    Returns
    -------
    subprocess.CompletedProcess[str]
        Completed Codex process.
    """
    with log_path.open("w", encoding="utf-8") as log_file:
        return subprocess.run(
            command,
            cwd=repo,
            check=False,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
            timeout=timeout_sec,
        )


def _create_running_log(
    logs_directory: Path,
    started_at: datetime,
    task_id: str | None,
) -> Path:
    """Create an empty log using the known start-time and task identifier.

    Parameters
    ----------
    logs_directory : Path
        Directory that stores loop logs.
    started_at : datetime
        Time at which the loop started.
    task_id : str | None
        Ralph task identifier selected before the loop starts.

    Returns
    -------
    Path
        Newly created running log path in the form
        ``ralph_YYYYMMDDTHHMMSS_NNN_running.log``.
    """
    task_number = int(task_id.removeprefix("TASK-")) if task_id is not None else 0
    timestamp = started_at
    while True:
        filename = f"ralph_{timestamp:%Y%m%dT%H%M%S}_{task_number:03d}_running.log"
        running_path = logs_directory / filename
        try:
            with running_path.open("x", encoding="utf-8"):
                pass
        except FileExistsError:
            timestamp += timedelta(seconds=1)
        else:
            return running_path


def _finalize_log(
    temporary_path: Path,
    logs_directory: Path,
    started_at: datetime,
    task_id: str | None,
    status: str,
) -> Path:
    """Move a loop log to its final, descriptive filename.

    Parameters
    ----------
    temporary_path : Path
        Path of the log written while the loop was running.
    logs_directory : Path
        Directory that stores finalized logs.
    started_at : datetime
        Time at which the loop started.
    task_id : str | None
        Ralph task identifier, if a valid task result was reported.
    status : str
        Final loop status.

    Returns
    -------
    Path
        Final log path in the form
        ``ralph_YYYYMMDDTHHMMSS_NNN_status.log``.
    """
    task_number = int(task_id.removeprefix("TASK-")) if task_id is not None else 0
    timestamp = started_at
    while True:
        filename = f"ralph_{timestamp:%Y%m%dT%H%M%S}_{task_number:03d}_{status}.log"
        final_path = logs_directory / filename
        if not final_path.exists():
            temporary_path.replace(final_path)
            return final_path
        timestamp += timedelta(seconds=1)


def _run(
    repo: Path,
    prompt_path: Path,
    max_loops: int,
    codex_executable: str,
    model: str | None = None,
    auto_approve: bool = False,
    dry_run: bool = False,
    api_retry_count: int = DEFAULT_API_RETRY_COUNT,
    api_retry_interval_sec: int = DEFAULT_API_RETRY_INTERVAL_SEC,
    codex_timeout_sec: int = DEFAULT_CODEX_TIMEOUT_SEC,
) -> ExitCode:
    """Run Ralph loops until a terminal status or safety limit is reached.

    Parameters
    ----------
    repo : Path
        Git repository to modify.
    prompt_path : Path
        UTF-8 prompt file used for every loop.
    max_loops : int
        Maximum number of Codex processes to start.
    codex_executable : str
        Codex executable name or path.
    model : str | None, default None
        Optional model override.
    auto_approve : bool, default False
        Automatically approve Codex requests in the workspace-write sandbox.
    dry_run : bool, default False
        Validate inputs and print the command without invoking Codex.
    api_retry_count : int, default 10
        Number of additional attempts after a Codex API or protocol failure.
    api_retry_interval_sec : int, default 5
        Seconds to wait between Codex API retry attempts.
    codex_timeout_sec : int, default 1800
        Maximum time to wait for each Codex child process.

    Returns
    -------
    ExitCode
        Runner outcome.
    """
    repo = repo.resolve()
    prompt_path = prompt_path.resolve()
    if max_loops < 1:
        logger.error("--max-loops must be at least 1")
        return ExitCode.PREFLIGHT_ERROR
    if api_retry_count < 0:
        logger.error("--api-retry-count must be at least 0")
        return ExitCode.PREFLIGHT_ERROR
    if api_retry_interval_sec < 0:
        logger.error("--api-retry-interval-sec must be at least 0")
        return ExitCode.PREFLIGHT_ERROR
    if codex_timeout_sec < 1:
        logger.error("--codex-timeout-sec must be at least 1")
        return ExitCode.PREFLIGHT_ERROR
    if not prompt_path.is_file():
        logger.error("Prompt file does not exist: {}", prompt_path)
        return ExitCode.PREFLIGHT_ERROR
    resolved_codex_executable = _resolve_codex_executable(codex_executable)
    if resolved_codex_executable is None:
        logger.error("Codex executable not found: {}", codex_executable)
        return ExitCode.PREFLIGHT_ERROR

    try:
        top_level = Path(_require_git_output(repo, "rev-parse", "--show-toplevel")).resolve()
        if top_level != repo:
            raise RuntimeError(f"--repo must be the Git root: {top_level}")
        _require_clean_worktree(repo)
    except RuntimeError as error:
        logger.error("{}", error)
        return ExitCode.PREFLIGHT_ERROR

    prompt = prompt_path.read_text(encoding="utf-8")
    tasks_path = repo / "TASKS.md"
    temp_directory = repo / "tmp"
    temp_directory.mkdir(exist_ok=True)
    logs_directory = repo / "logs"
    logs_directory.mkdir(exist_ok=True)

    try:
        incomplete_tasks, total_tasks, selected_task_id = _task_snapshot(tasks_path)
    except (OSError, UnicodeError, ValueError) as error:
        logger.error("{}", error)
        return ExitCode.PREFLIGHT_ERROR
    logger.info(
        "Incompleted tasks: {} / All tasks: {}",
        incomplete_tasks,
        total_tasks,
    )
    logger.info("Total loops: {}", max_loops)
    if incomplete_tasks == 0:
        logger.success("All Ralph tasks are complete")
        return ExitCode.SUCCESS
    if selected_task_id is None:
        logger.warning("Stopped because no incomplete Ralph task is eligible")
        return ExitCode.TASK_BLOCKED

    for loop_number in range(1, max_loops + 1):
        if loop_number > 1:
            try:
                incomplete_tasks, _, selected_task_id = _task_snapshot(tasks_path)
            except (OSError, UnicodeError, ValueError) as error:
                logger.error("{}", error)
                return ExitCode.PREFLIGHT_ERROR
            if incomplete_tasks == 0:
                logger.success("All Ralph tasks are complete")
                return ExitCode.SUCCESS
            if selected_task_id is None:
                logger.warning("Stopped because no incomplete Ralph task is eligible")
                return ExitCode.TASK_BLOCKED
        logger.info("Ralph loop start ({})", loop_number)
        before = _require_git_output(repo, "rev-parse", "HEAD")
        started_at = datetime.now(tz=UTC).astimezone()
        codex_environment = _build_codex_environment(repo)
        with tempfile.NamedTemporaryFile(
            dir=temp_directory,
            prefix="ralph-last-message-",
            suffix=".txt",
            delete=False,
        ) as output_file:
            output_path = Path(output_file.name)

        command = _build_codex_command(
            resolved_codex_executable,
            repo,
            output_path,
            prompt,
            model,
            auto_approve,
        )
        temporary_log_path = _create_running_log(
            logs_directory,
            started_at,
            selected_task_id,
        )
        if selected_task_id is not None:
            logger.info("Ralph task {} started", selected_task_id)
        if dry_run:
            print(subprocess.list2cmdline(command))
            output_path.unlink(missing_ok=True)
            temporary_log_path.unlink(missing_ok=True)
            return ExitCode.SUCCESS

        for api_attempt in range(1, api_retry_count + 2):
            try:
                completed = _run_codex(
                    command,
                    repo,
                    temporary_log_path,
                    codex_environment,
                    codex_timeout_sec,
                )
                if completed.returncode != 0:
                    failure = f"Codex exited with code {completed.returncode}"
                    exit_code = ExitCode.CODEX_FAILURE
                else:
                    message = output_path.read_text(encoding="utf-8")
                    status, task_id = _classify_output(message)
                    _validate_selected_task(status, task_id, selected_task_id)
                    break
            except subprocess.TimeoutExpired:
                failure = f"Codex timed out after {codex_timeout_sec} seconds"
                exit_code = ExitCode.CODEX_FAILURE
            except (OSError, UnicodeError, ValueError) as error:
                failure = str(error)
                exit_code = ExitCode.PROTOCOL_ERROR

            if api_attempt > api_retry_count:
                log_path = _finalize_log(
                    temporary_log_path,
                    logs_directory,
                    started_at,
                    selected_task_id,
                    "codex-failure" if exit_code == ExitCode.CODEX_FAILURE else "protocol-error",
                )
                logger.error("{}; see {}", failure, log_path.relative_to(repo))
                output_path.unlink(missing_ok=True)
                return exit_code

            logger.warning(
                "Codex API attempt {}/{} failed ({}); retrying in {} seconds",
                api_attempt,
                api_retry_count + 1,
                failure,
                api_retry_interval_sec,
            )
            output_path.unlink(missing_ok=True)
            time.sleep(api_retry_interval_sec)
        else:
            raise AssertionError("Codex retry loop must return or succeed")

        output_path.unlink(missing_ok=True)

        try:
            after_agent = _require_git_output(repo, "rev-parse", "HEAD")
            agent_commits = _commit_count(repo, before, after_agent)
            if agent_commits != 0:
                raise RuntimeError(
                    f"Codex created {agent_commits} commits; the runner owns loop commits"
                )
            _commit_loop_changes(repo, task_id, status)
            _require_clean_worktree(repo)
            after = _require_git_output(repo, "rev-parse", "HEAD")
            new_commits = _commit_count(repo, before, after)
        except RuntimeError as error:
            log_path = _finalize_log(
                temporary_log_path,
                logs_directory,
                started_at,
                task_id,
                "git-error",
            )
            logger.error("{}; see {}", error, log_path.relative_to(repo))
            return ExitCode.GIT_STATE_ERROR

        if status == "completed":
            if new_commits != 1:
                log_path = _finalize_log(
                    temporary_log_path,
                    logs_directory,
                    started_at,
                    task_id,
                    "git-error",
                )
                logger.error(
                    "{} reported completion but created {} commits; see {}",
                    task_id,
                    new_commits,
                    log_path.relative_to(repo),
                )
                return ExitCode.GIT_STATE_ERROR
            log_path = _finalize_log(
                temporary_log_path,
                logs_directory,
                started_at,
                task_id,
                status,
            )
            try:
                incomplete_tasks, _, next_task_id = _task_snapshot(tasks_path)
            except (OSError, UnicodeError, ValueError) as error:
                logger.error("{}", error)
                return ExitCode.PREFLIGHT_ERROR
            if incomplete_tasks == 0:
                logger.success(
                    "Completed {} ({})",
                    task_id,
                    log_path.relative_to(repo),
                )
                logger.success("All Ralph tasks are complete")
                return ExitCode.SUCCESS
            if next_task_id is None:
                logger.success(
                    "Completed {} ({})",
                    task_id,
                    log_path.relative_to(repo),
                )
                logger.warning("Stopped because no incomplete Ralph task is eligible")
                return ExitCode.TASK_BLOCKED
            logger.success(
                "Completed {}; starting the next loop ({})",
                task_id,
                log_path.relative_to(repo),
            )
            continue

        if new_commits != 1:
            log_path = _finalize_log(
                temporary_log_path,
                logs_directory,
                started_at,
                task_id,
                "git-error",
            )
            logger.error(
                "Terminal loop created {} commits; see {}",
                new_commits,
                log_path.relative_to(repo),
            )
            return ExitCode.GIT_STATE_ERROR
        if status == "incompleted":
            _finalize_log(
                temporary_log_path,
                logs_directory,
                started_at,
                task_id,
                status,
            )
            logger.warning("Stopped because {} is incomplete", task_id)
            return ExitCode.TASK_INCOMPLETE

        _finalize_log(
            temporary_log_path,
            logs_directory,
            started_at,
            task_id,
            status,
        )
        logger.warning("Stopped because {} is blocked", task_id)
        return ExitCode.TASK_BLOCKED

    logger.warning("Stopped after reaching the {}-loop limit", max_loops)
    return ExitCode.MAX_LOOPS_REACHED


def run(
    repo: Path,
    prompt_path: Path,
    max_loops: int,
    codex_executable: str,
    model: str | None = None,
    auto_approve: bool = False,
    dry_run: bool = False,
    api_retry_count: int = DEFAULT_API_RETRY_COUNT,
    api_retry_interval_sec: int = DEFAULT_API_RETRY_INTERVAL_SEC,
    codex_timeout_sec: int = DEFAULT_CODEX_TIMEOUT_SEC,
) -> ExitCode:
    """Run Ralph with one pair of runner lifecycle log messages.

    Parameters
    ----------
    repo : Path
        Git repository to modify.
    prompt_path : Path
        UTF-8 prompt file used for every loop.
    max_loops : int
        Maximum number of Codex processes to start.
    codex_executable : str
        Codex executable name or path.
    model : str | None, default None
        Optional model override.
    auto_approve : bool, default False
        Automatically approve Codex requests in the workspace-write sandbox.
    dry_run : bool, default False
        Validate inputs and print the command without invoking Codex.
    api_retry_count : int, default 10
        Number of additional attempts after a Codex API or protocol failure.
    api_retry_interval_sec : int, default 5
        Seconds to wait between Codex API retry attempts.
    codex_timeout_sec : int, default 1800
        Maximum time to wait for each Codex child process.

    Returns
    -------
    ExitCode
        Runner outcome.
    """
    logger.info("Ralph runner start")
    try:
        return _run(
            repo=repo,
            prompt_path=prompt_path,
            max_loops=max_loops,
            codex_executable=codex_executable,
            model=model,
            auto_approve=auto_approve,
            dry_run=dry_run,
            api_retry_count=api_retry_count,
            api_retry_interval_sec=api_retry_interval_sec,
            codex_timeout_sec=codex_timeout_sec,
        )
    finally:
        logger.info("Ralph runner end")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : Sequence[str] | None, default None
        Optional argument sequence. Defaults to ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run one Ralph task per Codex process until work stops.",
        epilog=(
            "Exit codes: 0=all complete/dry run, 10=Codex failure, "
            "11=protocol error, 12=Git state error, 20=task incomplete, "
            "21=task blocked, 22=max loops, 23=preflight error."
        ),
    )
    parser.add_argument("--repo", type=Path, default=repository_root)
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=repository_root / "scripts" / "ralph_prompt.md",
    )
    parser.add_argument("--max-loops", type=int, default=DEFAULT_MAX_LOOPS)
    parser.add_argument("--api-retry-count", type=int, default=DEFAULT_API_RETRY_COUNT)
    parser.add_argument(
        "--api-retry-interval-sec",
        type=int,
        default=DEFAULT_API_RETRY_INTERVAL_SEC,
    )
    parser.add_argument(
        "--codex-timeout-sec",
        type=int,
        default=DEFAULT_CODEX_TIMEOUT_SEC,
        help="maximum seconds to wait for each Codex child process",
    )
    parser.add_argument("--codex", default="codex", help="Codex executable name or path")
    parser.add_argument("--model", help="optional Codex model override")
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="automatically approve Codex requests in the workspace-write sandbox",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inputs and print one Codex command without running it",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line entry point.

    Parameters
    ----------
    argv : Sequence[str] | None, default None
        Optional argument sequence. Defaults to ``sys.argv``.

    Returns
    -------
    int
        Process exit code.
    """
    args = _parse_args(argv)
    configure_console_logging()
    return int(
        run(
            repo=args.repo,
            prompt_path=args.prompt_file,
            max_loops=args.max_loops,
            codex_executable=args.codex,
            model=args.model,
            auto_approve=args.auto_approve,
            dry_run=args.dry_run,
            api_retry_count=args.api_retry_count,
            api_retry_interval_sec=args.api_retry_interval_sec,
            codex_timeout_sec=args.codex_timeout_sec,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
