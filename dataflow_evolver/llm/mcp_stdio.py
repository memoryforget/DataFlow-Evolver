"""Expose the DataFlow operator-reference MCP server over standard I/O."""
from __future__ import annotations

import anyio
from mcp.server.stdio import stdio_server


def _server_instance(server_config):
    try:
        return server_config["instance"]
    except (KeyError, TypeError):
        return server_config.instance


def build_server():
    """Return the same MCP server instance used in-process by Claude."""
    from dataflow_evolver.llm.mcp_operators import operator_mcp_server

    return _server_instance(operator_mcp_server)


async def _serve() -> None:
    server = build_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main() -> None:
    anyio.run(_serve)


if __name__ == "__main__":
    main()
