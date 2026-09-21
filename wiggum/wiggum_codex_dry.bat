@echo off
setlocal EnableExtensions

rem Run the GitHub-installed ribuka/wiggum loop from the repository root.
chcp 65001 > nul
set "REPO_ROOT=%~dp0.."

set "PROVIDER=codex"
@REM  set "MODEL=gpt-5.6-luna"
set "MODEL=gpt-5.6-terra"
set "EFFORT=medium"
set "MANAGE_PROCESS_ENV=false"

pushd "%REPO_ROOT%" || exit /b 1
uv run wiggum run ^
    --dry-run ^
    --provider %PROVIDER% ^
    --manage-process-env %MANAGE_PROCESS_ENV% ^
    --model %MODEL% ^
    --reasoning-effort %EFFORT%

set "WIGGUM_EXIT_CODE=%ERRORLEVEL%"
popd

exit /b %WIGGUM_EXIT_CODE%
