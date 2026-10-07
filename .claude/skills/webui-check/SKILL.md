---
name: webui-check
description: Launch the flat-pca Web UI on a temporary workspace, operate its screens in Chrome through Playwright MCP, and clean up afterwards. Use when asked to run, start, open, or screenshot the Web UI, or to confirm that a Web UI change works in a real browser.
---

# Web UI check

The procedure is shared with Codex. Read `.agents/skills/webui-check/SKILL.md` and follow it.

In Claude Code:

- The Playwright MCP server comes from the project's `.mcp.json`; its tools are `mcp__playwright__browser_*`.
- Start the server with the Bash tool and `run_in_background: true`, and stop it with `TaskStop`.
