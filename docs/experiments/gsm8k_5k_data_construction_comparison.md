# GSM8K 5k Data-Construction Comparison

Status: configuration confirmed and frozen for the formal construction run.

## Scope

This experiment compares only data construction by DataEvolver and DataFlow-Evolver.
It does not run SFT, OpenCompass, GSM8K test inference, or any MATH preparation/eval.

The only raw source is:

```text
/vepfs-mlp2/c20250602/500050/lh/xqf/DataFlow-Evolver/data/gsm8k_train_full.jsonl
```

Sampling is performed exactly once with Python `random.Random(42)` and preserves the
selected order in a single master JSONL. Both framework copies must have the same SHA-256.

The frozen 5k file has SHA-256:

```text
613c85c37ea70fe7e365b5c059109e5a2e7649458393c6fb57639c32695faf21
```

## Model allocation

| Component | Runtime/model |
|---|---|
| DataEvolver understanding/orchestration/evolution/instantiation/Judge | `gpt-5.5` |
| DataEvolver runtime pipeline LLM operators | `gpt-4o` |
| DataFlow PipelineAgent | Codex CLI 0.118.0 / `gpt-5.4` / effort `xhigh` / Responses API |
| DataFlow generated-pipeline generation/reasoning | `gpt-4o` |
| DataFlow ReviewAgent | `gpt-5.5` |
| DAS embedding | local `model/Qwen3-Embedding-8B` |

The chat endpoint is OpenAI-compatible. DataEvolver stores the base URL ending in `/v1`;
DataFlow stores the full endpoint ending in `/v1/chat/completions`. No secret is written
to either repository.

## DataEvolver configuration

The effective tracked defaults for this run are in `DataEvolver/config/config.json`:

```json
{
  "llm": {
    "model": "gpt-5.5",
    "pipeline_model": "gpt-4o",
    "temperature": 0.1,
    "max_tokens": 8000,
    "api_url": "http://123.129.219.111:3000/v1"
  },
  "processing": {
    "timeout": 120,
    "max_retries": 3,
    "full_run_llm_max_records_per_step": null
  }
}
```

`null` means the full-run LLM operators process all input records. Trial remains capped
at 8 records. The API key is read from `OPENAI_API_KEY` (or legacy `API_KEY`).

The formal session id is `gsm8k_5k_seed5_gpt55_gpt4o_r1`. It deliberately does not
reuse the earlier partial `gsm8k_5k_seed5` workflow state.

Session and workflow commands:

```bash
cd /vepfs-mlp2/c20250602/500050/lh/xqf/DataEvolver
dataevolver session-start gsm8k_5k_seed5_gpt55_gpt4o_r1 \
  --raw tmp/samples/gsm8k_train_5k_seed42.jsonl \
  --seed tmp/samples/gsm8k_reasoning_seed_5.jsonl \
  --domain mathematics --task-type reasoning --language English
dataevolver workflow advance-all gsm8k_5k_seed5_gpt55_gpt4o_r1 --max-steps 32
dataevolver state gsm8k_5k_seed5_gpt55_gpt4o_r1
dataevolver tokens gsm8k_5k_seed5_gpt55_gpt4o_r1
dataevolver run gsm8k_5k_seed5_gpt55_gpt4o_r1
```

## DataFlow-Evolver configuration

Use `configs/gsm8k_5k_compare.yaml`. Important resolved values:

- `project.run_name=gsm8k_5k_seed42_df_gpt54_codex_cli_r3`
- `agent.backend=codex`, `agent.model=gpt-5.4`, `agent.effort=xhigh`
- Codex CLI pinned to the tested version `0.118.0`
- Codex provider `openai-custom`, `wire_api=responses`
- pipeline generation/reasoning model `gpt-4o`
- Review model `gpt-5.5`
- `llm.api.max_workers=100`
- `loop.max_iterations=5`
- `loop.execution_repairs=3`
- target schema `instruction/input/output`
- duplicate fields `input/output`
- DAS candidate `input/output`, proxy `question/response`
- DAS sample size and seed `1000/42`

Required secret environment variables:

```bash
export DF_API_KEY='<configured outside the repository>'
export DF_AGENT_API_KEY="$DF_API_KEY"
export OPENAI_API_KEY="$DF_API_KEY"
```

The configured non-secret endpoints are:

```bash
export DF_API_URL=http://123.129.219.111:3000/v1/chat/completions
export DF_AGENT_BASE_URL=http://123.129.219.111:3000/v1
export CODEX_BIN=/path/to/codex-0.118.0
```

Run command, only after confirmation and after the local embedding endpoint is healthy:

```bash
cd /vepfs-mlp2/c20250602/500050/lh/xqf/DataFlow-Evolver
/vepfs-mlp2/c20250602/500050/lh/xqf/miniconda3/envs/dfe-review/bin/python \
  -m dataflow_evolver.main \
  --config configs/gsm8k_5k_compare.yaml \
  --task 'Transform only the supplied GSM8K 5k raw records into exactly 5,000 SFT records. Do not download or synthesize replacement problems. Each record must contain a fixed non-empty instruction, the concrete source problem in input, and ordered reasoning in output ending with the GSM8K convention #### <final answer>. The final answer must be mathematically equivalent to the source answer. Do not read GSM8K test or any external training questions.'
```

For the recorded server run, prefer `scripts/run_gsm8k_5k_compare.sh` after exporting
the same environment variables. The script stores the exact multiline task text without
putting secrets in the repository and avoids SSH/tmux command-line quoting changes.

The DataFlow result is always selected from `manifest.json -> best.dataset_path`, not
from the last iteration.

## Stop conditions

DataFlow-Evolver evaluates at most five candidates, including the initial pipeline.
A passing review does not stop the loop early. It stops after candidate five, or earlier
only when there is no reviewed incumbent, the PipelineAgent cannot produce the next
pipeline, or an unrecoverable infrastructure/execution failure occurs. Each candidate
allows at most three execution-repair attempts. The final result remains the best-so-far
incumbent.

DataEvolver `advance-all` stops successfully when quality check passes and the workflow
sets `ready_for_full_run=true`. It stops unsuccessfully when 32 workflow steps are
consumed, the same state repeats twice, an internal DAG-evolution guard is exceeded, or
a step raises an error. The full run is launched only after successful `advance-all` and
then stops after the generated pipeline processes the complete 5,000-row input or raises
an execution error. `full_run_llm_max_records_per_step=null` means no LLM-step row cap.

## Preflight evidence

- DataFlow-Evolver tests: `137 passed`.
- DataEvolver comparison-runtime tests: `5 passed`.
- Standalone Codex CLI health check: `gpt-5.4/xhigh`, Responses API, `CODEX_OK`.
- Chat health checks: `gpt-5.5` HTTP 200 and `gpt-4o` HTTP 200.
- DAS proxy: local `data/ODA-Math-460k` parquet dataset.
- Embedding model: local `model/Qwen3-Embedding-8B`.

## Evidence to retain

- master/copy SHA-256 and row/schema validation;
- exact resolved configuration with key status only;
- complete commands and wall-clock timestamps;
- DataEvolver workflow state, token ledger, run report, and raw output;
- DataFlow manifest, per-iteration token files, review evidence, and best raw output;
- chat token namespaces kept separate from embedding request/token/wall-time evidence;
- all failures, retries, missing usage, output yield, and uncovered calls.
