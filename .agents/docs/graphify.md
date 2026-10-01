# graphify

This repository keeps a code knowledge graph at `graphify-out/graph.json` (gitignored, AST-only, no LLM cost).
Use it to locate code before reading raw files, to save context tokens.

## Setup

- Install the CLI once per machine: `uv tool install graphifyy`.
- Enable the repository git hooks once per clone: `git config core.hooksPath .githooks`.
  - `post-checkout` / `post-commit` / `post-merge` / `post-rewrite` rebuild or update the graph automatically.
- If `graphify-out/graph.json` does not exist, build it: `graphify extract . --code-only`.
- If the `graphify` command is not available, skip this document and explore code as usual.

## Workflow

1. Orient with graphify first.
   - `graphify query "<question>"`: find the symbols and files related to a question.
   - `graphify explain "<symbol>"`: a symbol and its neighbors (callers, callees, imports).
   - `graphify path "<A>" "<B>"`: how two symbols are connected.
   - `graphify affected "<symbol>"`: what calls or imports a symbol (impact analysis before a change, incl. tests).
   - `graphify god-nodes`: the most connected hubs (for architecture overviews).
2. Read only the targeted ranges.
   - Results include `src=<path> loc=L<line>`; read those files with an offset/limit instead of whole files or broad greps.
3. Use Grep/Glob only after graphify has narrowed the target, or to find exact strings (error messages, config keys, literals).

## Tips

- Output is capped by `--budget` (default 2000 tokens).
  If a result says `TRUNCATED`, narrow the question or use `explain` on a specific symbol rather than raising the budget.
- Pass plain symbol names, e.g. `graphify explain "flatten_pca"` or `graphify explain "PreprocessConfig"`.
- `--code-only` indexes code only; read `SPEC.md` and `docs/` directly for specifications.
- Uncommitted edits are not in the graph until the next commit.
  Run `graphify update .` if you need the graph to reflect them mid-task
  (add `--force` after deleting code, since update refuses to shrink the graph by default).
- Do not commit or edit `graphify-out/`; dirty files there are expected.
