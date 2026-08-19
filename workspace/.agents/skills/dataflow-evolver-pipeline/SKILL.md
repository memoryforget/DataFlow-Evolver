---
name: dataflow-evolver-pipeline
description: Generate one bounded custom DataFlow-Evolver pipeline using open-dataflow 1.0.10.
---

# DataFlow-Evolver Pipeline Contract

Use this skill for Pipeline Agent generation and self-correction. It is the
complete API contract for this task. Keep inspection scoped to the current
iteration artifacts and the dynamic task context.

## Imports and runtime

- Runtime distribution: `open-dataflow==1.0.10`; Python imports use `dataflow.*`.
- Import `OperatorABC` from `dataflow.core`, `PipelineABC` from
  `dataflow.pipeline`, and `FileStorage` from `dataflow.utils.storage`.
- API-backed operators import `EvolverAPILLMServing` from
  `dataflow_evolver.compat`.
- Generated operators and pipelines are local custom code. Use the dynamic URL,
  model, API-key environment-variable name, worker count, and serving kwargs
  supplied by the task prompt.

## FileStorage contract

Construct one sequential storage object:

```python
self.storage = FileStorage(
    first_entry_file_name=ENTRY_PATH,
    cache_path=CACHE_DIR,
    file_name_prefix="pipeline_step",
    cache_type="jsonl",
)
```

`FileStorage` is step-based, not key-based:

- `storage.step()` takes no arguments. It advances the pipeline step and returns
  the storage view passed to one operator.
- `storage.read("dict")` returns the current step as `list[dict]`.
- `storage.read()` returns a pandas DataFrame.
- `storage.write(rows)` writes the next step and returns its file path.
- Do not pass logical input/output keys to `step`, `read`, or `write`.

## Operator contract

Each operator subclasses `OperatorABC`, calls `super().__init__()`, and implements
`run`. A standard operator is:

```python
class CustomOperator(OperatorABC):
    def __init__(self, ...):
        super().__init__()

    def run(self, storage: FileStorage) -> None:
        rows = storage.read("dict")
        output_rows = []
        for row in rows:
            # Preserve original fields and apply this step's bounded transform.
            output_rows.append(row)
        storage.write(output_rows)
```

An LLM operator owns one `EvolverAPILLMServing` instance. Its generated responses
remain aligned with inputs; filter `None` or empty responses explicitly before
writing. Reuse valid parent operator code and intermediate step JSONL files when
the task prompt supplies them, especially to avoid repeated LLM calls.

## Pipeline contract

Every concrete pipeline subclasses `PipelineABC`. Instantiate storage and each
operator as members in `__init__`; `PipelineABC.compile()` discovers only operator
members. `forward()` must contain only ordered operator calls:

```python
class Pipeline(PipelineABC):
    def __init__(self):
        super().__init__()
        self.storage = FileStorage(...)
        self.first_operator = FirstOperator(...)
        self.second_operator = SecondOperator(...)

    def forward(self) -> None:
        self.first_operator.run(storage=self.storage.step())
        self.second_operator.run(storage=self.storage.step())
```

`compile()` internally calls `forward()` once after wrapping operator members to
record the graph; operator bodies are not run during that graph-recording call.
Keep non-operator side effects out of `forward()`. The formal `forward()` writes
ordered step JSONL files beneath the configured cache directory, and the final
step JSONL is the dataset artifact.

Use exactly this entry point:

```python
if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.compile()
    if os.getenv("DF_COMPILE_ONLY") == "1":
        raise SystemExit(0)
    pipeline.forward()
```

Do not execute the pipeline yourself. Evolver performs artifact validation,
compile, formal execution, output discovery, and self-correction.

## Serving response policy

`EvolverAPILLMServing` is a lightweight `LLMServingABC` implementation:

- `response_mode="content"`: return assistant content; safe default.
- `response_mode="reasoning_and_content"`: compose reasoning and content with
  the configured `reasoning_template`.
- `response_mode="auto"`: compose when reasoning exists, otherwise use content.
- `reasoning_fields`, `reasoning_template`, `allowed_finish_reasons`,
  `require_reasoning`, and `require_content` come from experiment configuration.

For CoT SFT, the final assistant field must contain the serialized reasoning and
answer expected by the model template. For non-CoT tasks use content mode. Never
infer CoT behavior from the model name.

## Task discipline

- Treat the task prompt as authoritative for paths, schemas, parent artifacts,
  feedback, bad-case counts, and serving settings.
- Preserve the fixed input corpus and every original field unless the task
  explicitly transforms it.
- Keep domain strategy in custom operators rather than inventing APIs.
- Do not scan the full corpus, install packages, run git, execute the pipeline,
  or benchmark it during generation.
- Prioritize a coherent, complete `pipeline.py` and its local operators early.
  Inspect only the current iteration files needed to satisfy the artifact contract.
- After the first write, make any focused follow-up edits needed to leave coherent,
  runnable artifacts, then return control to Evolver for framework validation.
- On self-correction, use the supplied traceback and diagnostics and modify only
  the smallest failing region.
