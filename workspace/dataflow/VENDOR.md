# Vendored third-party source

This directory is a **vendored copy** of the `dataflow` package from
[open-dataflow-mm](https://github.com/OpenDCAI/DataFlow), redistributed here so that
DataFlow-Evolver works out-of-the-box (no manual source preparation needed).

- Upstream package: `open-dataflow-mm`
- Version: `0.0.1`
- Vendored upstream commit: `d706997`
- License: Apache-2.0 (see `LICENSE` in this directory)

## Why it's here

Only the in-process MCP server (`dataflow_evolver/llm/mcp_operators.py`) depends on
DataFlow: it loads the operator registry and exposes existing operators' source code as
a coding-style reference for the PipelineAgent. The `dataflow-dev` skill also assumes the
agent `cwd` (`workspace/_agent_root/`) is the DataFlow repo root, so the source must live
under `cwd`.

## Updating

Re-vendor from a newer upstream checkout and update the version/commit fields above:

```bash
rm -rf workspace/_agent_root/dataflow
cp -r <DataFlow>/dataflow workspace/_agent_root/dataflow
cp <DataFlow>/LICENSE workspace/_agent_root/dataflow/LICENSE
```
