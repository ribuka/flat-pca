# Playwright MCP

Agents check the Web UI in Google Chrome through the Playwright MCP server (`@playwright/mcp`).
The procedure for launching and checking the Web UI is the `webui-check` skill.

## Requirements

- Node.js (`npx`) and Google Chrome.
- The server runs `--browser chrome` (the installed Chrome; no Playwright browser download)
  and `--isolated` (an in-memory profile, so several worktrees can use it at once).
- Snapshots and other output files go to `tmp/playwright-mcp/`, relative to the directory the server starts in (the repository root).

## Claude Code

- The server is registered in `.mcp.json` at the repository root.
- Approve the `playwright` server when Claude Code asks; `claude mcp list` shows its status.

## Codex

Codex reads MCP servers from `~/.codex/config.toml` (`.codex/` in the repository is gitignored).
Register the same command:

```bash
codex mcp add playwright -- npx -y @playwright/mcp@0.0.83 --browser chrome --isolated --output-dir tmp/playwright-mcp
```

or add it to `~/.codex/config.toml` directly:

```toml
[mcp_servers.playwright]
command = "npx"
args = ["-y", "@playwright/mcp@0.0.83", "--browser", "chrome", "--isolated", "--output-dir", "tmp/playwright-mcp"]
```

Codex finds the project skill at `.agents/skills/webui-check/SKILL.md`.

## Updating the version

The `@playwright/mcp` version is pinned. When you change it, update `.mcp.json` and this document together.
