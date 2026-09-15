"""Run Ralph tasks repeatedly through the non-interactive Codex CLI."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from enum import IntEnum
from pathlib import Path

DEFAULT_MAX_LOOPS = 20
TASK_COMPLETED_PATTERN = re.compile(r"TASK_COMPLETED: (TASK-\d{3})")
TASK_INCOMPLETE_PATTERN = re.compile(r"TASK_INCOMPLETE: (TASK-\d{3})")
TASK_BLOCKED_PATTERN = re.compile(r"TASK_BLOCKED: (TASK-\d{3})")


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
        return "all_completed", None

    for status, pattern in (
        ("task_completed", TASK_COMPLETED_PATTERN),
        ("task_incomplete", TASK_INCOMPLETE_PATTERN),
        ("task_blocked", TASK_BLOCKED_PATTERN),
    ):
        match = pattern.fullmatch(token)
        if match:
            return status, match.group(1)

    raise ValueError(f"invalid Ralph status line: {token!r}")


def _build_codex_command(
    executable: str,
    repo: Path,
    output_path: Path,
    prompt: str,
    model: str | None,
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

    Returns
    -------
    list[str]
        Subprocess argument vector.
    """
    command = [
        executable,
        "exec",
        "--ephemeral",
        "--sandbox",
        "workspace-write",
        "--approve-for-me",
        "--cd",
        str(repo),
        "--output-last-message",
        str(output_path),
    ]
    if model is not None:
        command.extend(["--model", model])
    command.append(prompt)
    return command


def run(
    repo: Path,
    prompt_path: Path,
    max_loops: int,
    codex_executable: str,
    model: str | None = None,
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
    if shutil.which(codex_executable) is None and not Path(codex_executable).is_file():
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
    temp_directory = repo / "tmp"
    temp_directory.mkdir(exist_ok=True)

    for loop_number in range(1, max_loops + 1):
        before = _require_git_output(repo, "rev-parse", "HEAD")
        with tempfile.NamedTemporaryFile(
            dir=temp_directory,
            prefix="ralph-last-message-",
            suffix=".txt",
            delete=False,
        ) as output_file:
            output_path = Path(output_file.name)

        command = _build_codex_command(
            codex_executable,
            repo,
            output_path,
            prompt,
            model,
        )
        print(f"Ralph loop {loop_number}/{max_loops}")
        if dry_run:
            print(subprocess.list2cmdline(command))
            output_path.unlink(missing_ok=True)
            return ExitCode.SUCCESS

        try:
            completed = subprocess.run(command, cwd=repo, check=False)
            if completed.returncode != 0:
                print(
                    f"error: Codex exited with code {completed.returncode}",
                    file=sys.stderr,
                )
                return ExitCode.CODEX_FAILURE
            message = output_path.read_text(encoding="utf-8")
            status, task_id = _classify_output(message)
        except (OSError, UnicodeError, ValueError) as error:
            print(f"error: {error}", file=sys.stderr)
            return ExitCode.PROTOCOL_ERROR
        finally:
            output_path.unlink(missing_ok=True)

        after = _require_git_output(repo, "rev-parse", "HEAD")
        try:
            _require_clean_worktree(repo)
            new_commits = _commit_count(repo, before, after)
        except RuntimeError as error:
            print(f"error: {error}", file=sys.stderr)
            return ExitCode.GIT_STATE_ERROR

        if status == "task_completed":
            if new_commits != 1:
                print(
                    f"error: {task_id} reported completion but created "
                    f"{new_commits} commits",
                    file=sys.stderr,
                )
                return ExitCode.GIT_STATE_ERROR
            print(f"Completed {task_id}; starting the next loop.")
            continue

        if new_commits > 1:
            print(
                f"error: terminal loop created {new_commits} commits",
                file=sys.stderr,
            )
            return ExitCode.GIT_STATE_ERROR
        if status == "all_completed":
            if new_commits != 0:
                print("error: all-completed loop created a commit", file=sys.stderr)
                return ExitCode.GIT_STATE_ERROR
            print("All Ralph tasks are complete.")
            return ExitCode.SUCCESS
        if status == "task_incomplete":
            print(f"Stopped because {task_id} is incomplete.", file=sys.stderr)
            return ExitCode.TASK_INCOMPLETE

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
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
