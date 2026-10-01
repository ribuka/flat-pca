# Commit

## Message format

- Write the subject line as `<type>: <summary>`.
  - Write `<summary>` in Japanese.
  - Keep code identifiers (function names, file paths, etc.) as-is.
- Optionally add a body after a blank line to explain why the change was made.

### Types

- `feat`: A new feature
- `fix`: A bug fix
- `perf`: A change that improves performance
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `test`: Adding or updating tests
- `docs`: Documentation-only changes
- `style`: Formatting changes that do not affect behavior
- `build`: Changes to the build system or dependencies (e.g. `pyproject.toml`, `uv.lock`)
- `ci`: Changes to CI configuration and scripts
- `chore`: Other maintenance that does not modify `src/` or tests
- `revert`: Reverting a previous commit

Example:

```text
perf: build_flattened_frame を縦長フレーム経由の transpose で widening する
```

## Attribution

- NEVER include session URLs (e.g. `https://claude.ai/code/session_...`) in commit messages.
  - Omit the `Claude-Session:` trailer.
- `Co-Authored-By:` trailers are allowed.
