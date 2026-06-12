"""
Smritikosh MCP server — exposes the memory API as Model Context Protocol tools.

Any MCP-capable agent (Claude Code, Claude Desktop, etc.) becomes a
zero-integration Smritikosh client. See smritikosh.mcp.server for the
tool definitions and local_docs/MCP_SERVER.md for setup.
"""

from smritikosh.mcp.server import mcp, main

__all__ = ["mcp", "main"]
