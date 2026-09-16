@echo off
setlocal EnableExtensions

rem Run the GitHub-installed ribuka/wiggum loop from the repository root.
chcp 65001 > nul
set "REPO_ROOT=%~dp0.."

pushd "%REPO_ROOT%" || exit /b 1
uv run wiggum run --no-managed-env %*
set "WIGGUM_EXIT_CODE=%ERRORLEVEL%"
popd

exit /b %WIGGUM_EXIT_CODE%
