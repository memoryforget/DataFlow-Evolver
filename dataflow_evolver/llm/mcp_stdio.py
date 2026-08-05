"""Expose the existing Claude SDK MCP tools over standard MCP stdio.

The tool implementations stay single-sourced in ``mcp_*.py``.  Claude uses
their in-process server instances; OpenCode starts this module as a local MCP
subprocess and talks to the exact same server over stdio.
"""
from __future__ import annotations

import argparse
import os
from typing import Any

import anyio
from mcp.server.stdio import stdio_server


def _cfg_instance(server_cfg: Any) -> Any:
    try:
        return server_cfg["instance"]
    except (KeyError, TypeError):
        return server_cfg.instance


def build_server(profile: str) -> Any:
    if profile == "operators":
        from dataflow_evolver.llm.mcp_operators import operator_mcp_server

        return _cfg_instance(operator_mcp_server)
    if profile == "datasets":
        from dataflow_evolver.llm import mcp_datasets

        cfg = {
            "max_download_mb": os.environ.get("DF_DATASET_MAX_DOWNLOAD_MB", ""),
            "endpoint": os.environ.get("HF_ENDPOINT", ""),
            "token": os.environ.get("HF_TOKEN", ""),
            "cache_dir": os.environ.get("HF_HOME", ""),
            "modelscope_endpoint": os.environ.get("MODELSCOPE_ENDPOINT", ""),
            "modelscope_token": os.environ.get("MODELSCOPE_API_TOKEN", ""),
        }
        mcp_datasets.configure(
            os.environ.get("DF_DATA_LINKS_DIR", "workspace/data_links"),
            {key: value for key, value in cfg.items() if value},
        )
        return _cfg_instance(mcp_datasets.dataset_mcp_server)
    if profile == "discovery":
        from dataflow_evolver.llm import mcp_discovery

        mcp_discovery.configure(
            {
                "serper_api_key": os.environ.get("DF_DISCOVERY_SERPER_API_KEY", ""),
                "github_token": os.environ.get("DF_DISCOVERY_GITHUB_TOKEN", ""),
            }
        )
        return _cfg_instance(mcp_discovery.discovery_mcp_server)
    raise ValueError(f"unknown MCP profile: {profile!r}")


async def _serve(profile: str) -> None:
    server = build_server(profile)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="DataFlow-Evolver MCP stdio bridge")
    parser.add_argument("profile", choices=("operators", "datasets", "discovery"))
    args = parser.parse_args()
    anyio.run(_serve, args.profile)


if __name__ == "__main__":
    main()
