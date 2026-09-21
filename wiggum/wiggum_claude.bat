@echo off
setlocal EnableExtensions

rem Run the GitHub-installed ribuka/wiggum loop from the repository root.
chcp 65001 > nul
set "REPO_ROOT=%~dp0.."

set "PROVIDER=claude"
set "MODEL=claude-sonnet-5"
set "EFFORT=medium"
set "MANAGE_PROCESS_ENV=true"

pushd "%REPO_ROOT%" || exit /b 1

uv run wiggum run ^
    --provider %PROVIDER% ^
    --model %MODEL% ^
    --reasoning-effort %EFFORT% ^
    --manage-process-env %MANAGE_PROCESS_ENV%

set "WIGGUM_EXIT_CODE=%ERRORLEVEL%"
popd

exit /b %WIGGUM_EXIT_CODE%
