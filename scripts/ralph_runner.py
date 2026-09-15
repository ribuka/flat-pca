"""Run Ralph tasks repeatedly through the non-interactive Codex CLI."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from enum import IntEnum
from pathlib import Path

DEFAULT_MAX_LOOPS = 20
TASK_COMPLETED_PATTERN = re.compile(r"TASK_COMPLETED: (TASK-\d{3})")
TASK_INCOMPLETE_PATTERN = re.compile(r"TASK_INCOMPLETE: (TASK-\d{3})")
TASK_BLOCKED_PATTERN = re.compile(r"TASK_BLOCKED: (TASK-\d{3})")
TASK_STATUS_PATTERN = re.compile(
    r"^## TASK-\d{3}:.*?\r?\n\r?\n- Status: (?P<status>\w+)$",
    re.MULTILINE,
)


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


def _classify_output(message: str) -> tuple[str, str | None]:
    """Classify the Ralph protocol token in a final agent message.

    Parameters
    ----------
    message : str
        Final Codex agent message.

    Returns
    -------
    tuple[str, str | None]
        Protocol status and optional task ID.

    Raises
    ------
    ValueError
        If the last non-empty line is not a valid Ralph status token.
    """
    token = _last_non_empty_line(message)
    if token == "ALL_TASKS_COMPLETED":
        return "all-completed", None

    for status, pattern in (
        ("completed", TASK_COMPLETED_PATTERN),
        ("incompleted", TASK_INCOMPLETE_PATTERN),
        ("blocked", TASK_BLOCKED_PATTERN),
    ):
        match = pattern.fullmatch(token)
        if match:
            return status, match.group(1)

    raise ValueError(f"invalid Ralph status line: {token!r}")


def _task_progress(tasks_path: Path) -> tuple[int, int]:
    """Return remaining and total task counts from a Ralph task file.

    Parameters
    ----------
    tasks_path : Path
        UTF-8 task definition file containing task headings and status fields.

    Returns
    -------
    tuple[int, int]
        Number of tasks not yet completed and total task count.

    Raises
    ------
    ValueError
        If the file does not contain any task status fields.
    """
    statuses = [match.group("status") for match in TASK_STATUS_PATTERN.finditer(
        tasks_path.read_text(encoding="utf-8")
    )]
    if not statuses:
        raise ValueError(f"no task statuses found: {tasks_path}")
    return sum(status != "completed" for status in statuses), len(statuses)


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
        "--cd",
        str(repo),
        "--output-last-message",
        str(output_path),
    ]
    if auto_approve:
        command.append("--approve-for-me")
    else:
        command.extend(["--sandbox", "workspace-write"])
    if model is not None:
        command.extend(["--model", model])
    command.append(prompt)
    return command


def _build_codex_environment(repo: Path) -> dict[str, str]:
    """Build a child-process environment with repository-local temporary paths.

    Parameters
    ----------
    repo : Path
        Repository root containing the Ralph ``tmp`` directory.

    Returns
    -------
    dict[str, str]
        Copy of the current environment with uv and runtime temporary paths
        directed into the repository.
    """
    temporary_directory = (repo / "tmp").resolve()
    uv_cache_directory = temporary_directory / "uv-cache"
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
        )


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


def run(
    repo: Path,
    prompt_path: Path,
    max_loops: int,
    codex_executable: str,
    model: str | None = None,
    auto_approve: bool = False,
    dry_run: bool = False,
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

    Returns
    -------
    ExitCode
        Runner outcome.
    """
    repo = repo.resolve()
    prompt_path = prompt_path.resolve()
    if max_loops < 1:
        print("error: --max-loops must be at least 1", file=sys.stderr)
        return ExitCode.PREFLIGHT_ERROR
    if not prompt_path.is_file():
        print(f"error: prompt file does not exist: {prompt_path}", file=sys.stderr)
        return ExitCode.PREFLIGHT_ERROR
    resolved_codex_executable = _resolve_codex_executable(codex_executable)
    if resolved_codex_executable is None:
        print(f"error: Codex executable not found: {codex_executable}", file=sys.stderr)
        return ExitCode.PREFLIGHT_ERROR

    try:
        top_level = Path(_require_git_output(repo, "rev-parse", "--show-toplevel")).resolve()
        if top_level != repo:
            raise RuntimeError(f"--repo must be the Git root: {top_level}")
        _require_clean_worktree(repo)
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return ExitCode.PREFLIGHT_ERROR

    prompt = prompt_path.read_text(encoding="utf-8")
    tasks_path = repo / "TASKS.md"
    temp_directory = repo / "tmp"
    temp_directory.mkdir(exist_ok=True)
    logs_directory = repo / "logs"
    logs_directory.mkdir(exist_ok=True)

    for loop_number in range(1, max_loops + 1):
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
        with tempfile.NamedTemporaryFile(
            dir=logs_directory,
            prefix=".ralph-running-",
            suffix=".log",
            delete=False,
        ) as log_file:
            temporary_log_path = Path(log_file.name)
        try:
            remaining_tasks, total_tasks = _task_progress(tasks_path)
        except (OSError, UnicodeError, ValueError) as error:
            output_path.unlink(missing_ok=True)
            temporary_log_path.unlink(missing_ok=True)
            print(f"error: {error}", file=sys.stderr)
            return ExitCode.PREFLIGHT_ERROR
        print(f"Ralph tasks {remaining_tasks}/{total_tasks} started")
        if dry_run:
            print(subprocess.list2cmdline(command))
            output_path.unlink(missing_ok=True)
            temporary_log_path.unlink(missing_ok=True)
            return ExitCode.SUCCESS

        try:
            completed = _run_codex(
                command,
                repo,
                temporary_log_path,
                codex_environment,
            )
            if completed.returncode != 0:
                log_path = _finalize_log(
                    temporary_log_path,
                    logs_directory,
                    started_at,
                    None,
                    "codex-failure",
                )
                print(
                    f"error: Codex exited with code {completed.returncode}; "
                    f"see {log_path.relative_to(repo)}",
                    file=sys.stderr,
                )
                return ExitCode.CODEX_FAILURE
            message = output_path.read_text(encoding="utf-8")
            status, task_id = _classify_output(message)
        except (OSError, UnicodeError, ValueError) as error:
            log_path = _finalize_log(
                temporary_log_path,
                logs_directory,
                started_at,
                None,
                "protocol-error",
            )
            print(f"see {log_path.relative_to(repo)}", file=sys.stderr)
            print(f"error: {error}", file=sys.stderr)
            return ExitCode.PROTOCOL_ERROR
        finally:
            output_path.unlink(missing_ok=True)

        try:
            after_agent = _require_git_output(repo, "rev-parse", "HEAD")
            agent_commits = _commit_count(repo, before, after_agent)
            if agent_commits != 0:
                raise RuntimeError(
                    f"Codex created {agent_commits} commits; the runner owns loop commits"
                )
            if status != "all-completed":
                if task_id is None:
                    raise RuntimeError("task status did not include a task ID")
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
            print(f"see {log_path.relative_to(repo)}", file=sys.stderr)
            print(f"error: {error}", file=sys.stderr)
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
                print(
                    f"error: {task_id} reported completion but created "
                    f"{new_commits} commits",
                    file=sys.stderr,
                )
                print(f"see {log_path.relative_to(repo)}", file=sys.stderr)
                return ExitCode.GIT_STATE_ERROR
            log_path = _finalize_log(
                temporary_log_path,
                logs_directory,
                started_at,
                task_id,
                status,
            )
            print(f"Completed {task_id}; starting the next loop ({log_path.relative_to(repo)}).")
            continue

        if new_commits != 1 and status != "all-completed":
            log_path = _finalize_log(
                temporary_log_path,
                logs_directory,
                started_at,
                task_id,
                "git-error",
            )
            print(
                f"error: terminal loop created {new_commits} commits",
                file=sys.stderr,
            )
            print(f"see {log_path.relative_to(repo)}", file=sys.stderr)
            return ExitCode.GIT_STATE_ERROR
        if new_commits > 1:
            log_path = _finalize_log(
                temporary_log_path,
                logs_directory,
                started_at,
                task_id,
                "git-error",
            )
            print(
                f"error: terminal loop created {new_commits} commits",
                file=sys.stderr,
            )
            print(f"see {log_path.relative_to(repo)}", file=sys.stderr)
            return ExitCode.GIT_STATE_ERROR
        if status == "all-completed":
            if new_commits != 0:
                log_path = _finalize_log(
                    temporary_log_path,
                    logs_directory,
                    started_at,
                    task_id,
                    "git-error",
                )
                print("error: all-completed loop created a commit", file=sys.stderr)
                print(f"see {log_path.relative_to(repo)}", file=sys.stderr)
                return ExitCode.GIT_STATE_ERROR
            _finalize_log(
                temporary_log_path,
                logs_directory,
                started_at,
                task_id,
                status,
            )
            print("All Ralph tasks are complete.")
            return ExitCode.SUCCESS
        if status == "incompleted":
            _finalize_log(
                temporary_log_path,
                logs_directory,
                started_at,
                task_id,
                status,
            )
            print(f"Stopped because {task_id} is incomplete.", file=sys.stderr)
            return ExitCode.TASK_INCOMPLETE

        _finalize_log(
            temporary_log_path,
            logs_directory,
            started_at,
            task_id,
            status,
        )
        print(f"Stopped because {task_id} is blocked.", file=sys.stderr)
        return ExitCode.TASK_BLOCKED

    print(f"Stopped after reaching the {max_loops}-loop limit.", file=sys.stderr)
    return ExitCode.MAX_LOOPS_REACHED


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
    return int(
        run(
            repo=args.repo,
            prompt_path=args.prompt_file,
            max_loops=args.max_loops,
            codex_executable=args.codex,
            model=args.model,
            auto_approve=args.auto_approve,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
