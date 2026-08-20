---
name: dataflow-evolver-pipeline
description: Design and implement a high-quality custom DataFlow-Evolver pipeline using open-dataflow 1.0.10.
---

# DataFlow-Evolver Pipeline Contract

Use this skill when you generate or self-correct a DataFlow-Evolver pipeline. It is the
complete API contract for this task. Use the dynamic task context, Review feedback,
downstream results, and bad cases to make the strongest justified pipeline design.

## Imports and runtime

- Runtime distribution: `open-dataflow==1.0.10`; Python imports use `dataflow.*`.
- Import `OperatorABC` and `LLMServingABC` from `dataflow.core`,
  `PipelineABC` from `dataflow.pipeline`, and `FileStorage` from
  `dataflow.utils.storage`.
- Pipeline modules import `EvolverAPILLMServing` from
  `dataflow_evolver.compat`; operator modules depend only on an injected
  `LLMServingABC`.
- Generated operators and pipelines are local custom code. Available serving
  profiles are runtime capabilities, not prescribed operator roles. You decide whether, where, and how to use them from the task and feedback.
  When used, construct serving in the Pipeline from the dynamic configuration.

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

## Framework-owned prefix reuse

Step filenames are execution-relative, not operator-relative. If a prefix operator
is skipped, every downstream `pipeline_step_stepN.jsonl` number moves forward.
Never infer an artifact producer from its filename, inspect parent cache files to
choose one, or implement custom cache validation. Parseable JSONL is not proof of
producer identity.

Every pipeline must mechanically consume the framework decision:

```python
from dataflow_evolver.engine.step_cache import FrameworkStepCache

OPERATOR_NAMES = ["FirstOperator", "SecondOperator"]
self.framework_cache = FrameworkStepCache.from_env(
    raw_entry_path=ENTRY_PATH,
    operator_names=OPERATOR_NAMES,
)
self.storage = FileStorage(
    first_entry_file_name=self.framework_cache.entry_path,
    cache_path=CACHE_DIR,
    file_name_prefix="pipeline_step",
    cache_type="jsonl",
)
```

Use stable zero-based logical indexes in `forward()`:

```python
def forward(self) -> None:
    if self.framework_cache.should_run(0):
        self.first_operator.run(storage=self.storage.step())
    if self.framework_cache.should_run(1):
        self.second_operator.run(storage=self.storage.step())
```

The framework validates raw-input identity and the ordered semantic identity declared
by operator name, `decision.json.params`, and initialization calls against the parent's
`step_manifest.json`, then injects the entry path and reusable prefix count. Source bytes
and source-file hashes are intentionally not compared. Keeping the same identity declares
that behavior is unchanged and authorizes reuse. If an implementation or local dependency
changes behavior, change the operator class name, decision params (for example a semantic
revision), or initialization arguments so the framework executes the changed suffix. The pipeline must not read the manifest or parent cache itself. Without a
valid framework plan, the adapter uses the raw entry and runs every operator.
`decision.json.reason` may identify unchanged prefix operators and the first
changed operator, but must not claim a cache hit or select an artifact path.

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
            # Preserve original fields and apply this operator's coherent responsibility.
            output_rows.append(row)
        storage.write(output_rows)
```

## Serving ownership and dependency injection

The Pipeline owns serving construction. An API-backed operator receives an initialized
`LLMServingABC` through its constructor and stores only that interface:

```python
class LLMAssistedOperator(OperatorABC):
    def __init__(self, llm_serving: LLMServingABC, domain_option: str = "default"):
        super().__init__()
        self.llm_serving = llm_serving
        self.domain_option = domain_option

    def run(self, storage: FileStorage) -> None:
        rows = storage.read("dict")
        prompts = [self.build_prompt(row) for row in rows]
        responses = self.llm_serving.generate_from_input(prompts)
        output_rows = []
        for row, response in zip(rows, responses, strict=True):
            if response is None or not str(response).strip():
                continue
            output_rows.append(self.consume_response(row, str(response)))
        storage.write(output_rows)
```

- Do not construct `EvolverAPILLMServing` inside an operator module or `run()`.
- Do not make an operator read API-key variables or declare URL, model, worker,
  timeout, retry, reasoning-field, or response-format transport settings.
- Do not add operator-level `batch_size` or manually split requests merely to
  control API concurrency. Pass the aligned logical input list to serving and let
  its configured `max_workers` own transport concurrency. A domain-specific chunk
  size is acceptable only when chunking changes the data algorithm itself.
- Do not parse raw OpenAI response dictionaries or rebuild
  `reasoning + content` in an operator. Serving returns the configured formatted
  value. Operators handle missing values and domain validation only.
- Share one serving instance among operators with the same response contract.
  When the design requires different response contracts, construct separate
  Pipeline-owned instances from the available dynamic profile. The Skill does not
  prescribe what those calls do.
- Do not initialize or close serving inside `run()`; one execution may reuse it
  across multiple operator calls.

Generated responses remain aligned with inputs; filter `None` or empty responses
explicitly before writing. Copy unchanged parent operator source when useful, but
leave intermediate artifact selection, validation, and reuse entirely to the
framework-owned step manifest mechanism.

## Pipeline contract

Every concrete pipeline subclasses `PipelineABC`. Instantiate storage and each
operator as members in `__init__`; `PipelineABC.compile()` discovers only operator
members. `forward()` must contain only ordered operator calls:

```python
class Pipeline(PipelineABC):
    def __init__(self):
        super().__init__()
        self.storage = FileStorage(...)
        self.reasoning_serving = EvolverAPILLMServing(
            api_url=PIPELINE_API_URL,
            key_name_of_api_key=PIPELINE_API_KEY_ENV,
            model_name=PIPELINE_MODEL,
            max_workers=PIPELINE_MAX_WORKERS,
            **PIPELINE_REASONING_SERVING_KWARGS,
        )
        self.first_operator = FirstOperator(...)
        self.llm_operator = LLMAssistedOperator(
            llm_serving=self.reasoning_serving,
            domain_option="task-specific",
        )

    def forward(self) -> None:
        self.first_operator.run(storage=self.storage.step())
        self.llm_operator.run(storage=self.storage.step())
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

A dynamic serving profile is an available runtime contract, not a required use
case. You may use no LLM, one serving, or multiple Pipeline-owned
servings with deliberately selected response contracts. Inject any serving into
operators and do not copy compatibility rules into operator response parsers.
Whenever a formatted LLM response becomes the final assistant training field,
preserve the representation required by the task. Never infer response behavior
from the model name.

## Task discipline

- Treat the task prompt as authoritative for paths, schemas, parent artifacts,
  feedback, bad-case counts, and serving settings.
- Optimize for resolving Review, downstream-evaluation, and bad-case evidence, not for
  minimizing code, files, or operator count.
- Choose operator boundaries by coherent responsibility and diagnostic value. Freely add,
  remove, split, merge, or replace operators when the evidence justifies it.
- Do not preserve the parent architecture merely to obtain cache reuse. Preserve an
  operator identity only when its behavior and relevant dependencies are truly unchanged.
- Serving behavior is part of an LLM operator's semantic identity even though serving is
  injected. Record the relevant serving profile or semantic revision in
  `decision.json.params` so model or response-contract changes cannot reuse an
  incompatible cached step. Do not put secret values in `decision.json`.
- Preserve the fixed input corpus and every original field unless the task
  explicitly transforms it.
- Keep domain strategy in custom operators rather than inventing APIs.
- Do not scan the full corpus, install packages, run git, execute the pipeline,
  or benchmark it during generation.
- Produce a coherent, complete `pipeline.py` and local operators, spending the effort
  needed for a high-quality response to the supplied evidence. Inspect the current
  iteration files needed to satisfy the artifact contract.
- Iterate on the implementation as needed to leave coherent, runnable, well-structured
  artifacts, then return control to Evolver for framework validation.
- On self-correction, use the supplied traceback and diagnostics and modify only
  the smallest failing region.
