# Recipe / Pipeline Agent runtime boundary

The evolution loop has three explicit stages:

1. `RecipeAgent` receives `RecipeContext`, owns external source discovery/download, and writes only `recipe.json`.
2. The deterministic framework `RecipeValidator` validates and canonicalizes that file. It does not call an LLM.
3. `PipelineAgent` receives the validated recipe, the parent implementation, and implementation-level feedback. It writes code but cannot discover or download new sources.

Claude runs both agents today, with phase-specific MCP profiles:

- `recipe`: `dataset-search` and optional `data-discovery`; no `dataflow-operators`.
- `pipeline`: `dataflow-operators`; no external-data MCP.

Both agents depend on `llm.agent_runtime.AgentRuntime.run(...)`, not directly on Claude SDK. The contract deliberately models one completed stage run and includes `cwd`, `phase`, `tool_profile`, system-prompt append, and an `AgentRunResult`.

## OpenCode ablation

OpenCode's official SDK is a client for an OpenCode server. Work is organized as sessions, prompts are sent through the client, and streaming events are exposed separately. It is therefore not a drop-in replacement for Claude Agent SDK's `query(...)` iterator.

An OpenCode adapter should own these runtime-specific details:

1. start or connect to an OpenCode server;
2. create a session for the requested phase;
3. map `tool_profile` to an OpenCode agent configuration with phase-specific MCP and permissions;
4. submit the prompt, consume events, and collect tool calls/session id;
5. map transport, timeout, authentication, and process errors to `AgentRunResult`.

RecipeContext construction, deterministic validation, prompt contracts, tree search, and PipelineAgent must remain unchanged. This makes `agent_sdk.runtime` the intended ablation switch once `OpenCodeAgentRuntime` is implemented and its dependency is installed.

Primary references consulted for this boundary:

- OpenCode SDK documentation: <https://opencode.ai/docs/sdk/>
- OpenCode agent and permission configuration: <https://opencode.ai/docs/agents/>
- OpenCode TypeScript SDK: <https://github.com/anomalyco/opencode-sdk-js>
- OpenCode Python SDK: <https://github.com/anomalyco/opencode-sdk-python>
