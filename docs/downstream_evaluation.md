# Periodic downstream evaluation

Periodic downstream evaluation adds an expensive, task-level signal without importing a
training or benchmark stack into DataFlow Evolver. Static Review still runs on every
candidate. After every configured number of completed iterations, the current
best-so-far incumbent dataset is exported to an external command for SFT and evaluation:

```text
checkpoint 0 -> evaluate the untrained downstream model on the fixed diagnostic split
             -> PipelineAgent receives baseline scores and bad cases
iteration 1 -> static Review -> incumbent update
iteration 2 -> static Review -> incumbent update
            -> external SFT + benchmark evaluation of that incumbent
            -> validated scores and complete bad cases
iteration 3 -> PipelineAgent receives static Review + downstream feedback
            -> static Review -> incumbent update
iteration 4 -> static Review -> incumbent update
            -> external SFT + benchmark evaluation of that incumbent
iteration 5 -> PipelineAgent receives the latest downstream feedback
            -> final static Review and incumbent update
```

The downstream score is diagnostic feedback. It does not replace static `review_score`
and does not retroactively change incumbent selection. This keeps candidate selection
deterministic and makes it clear which signal caused each later pipeline change.

## Configuration

The external command is an argv list, not a shell expression. It can point to a Python
binary in a separate Conda environment, so MS-Swift, OpenCompass, model weights, CUDA
packages, and benchmark dependencies remain outside this project's environment.

```yaml
downstream_eval:
  enabled: true
  interval: 2
  command:
    - /opt/conda/envs/downstream/bin/python
    - /absolute/path/to/run_sft_and_eval.py
  benchmarks: [gsm8k, math, math500, aime]
  baseline:
    enabled: true
  bad_cases_per_benchmark: 3  # maximum retained per benchmark
  timeout_sec: 86400
```

DataFlow Evolver appends these arguments:

```text
--request <checkpoint_dir>/request.json --result <checkpoint_dir>/result.json
```

The child process inherits the launch environment. Provider keys and training settings
should be supplied to the external wrapper through that environment or its own config;
they are not copied into DataFlow Evolver's YAML.

## Request contract

`request.json` contains protocol version 3, the checkpoint stage, task contract,
expected benchmark names, and required bad-case count.

At checkpoint 0, `stage` is `baseline`, `training_performed` is false, and incumbent,
dataset, and pipeline paths are null. The runner must skip SFT and directly evaluate its
configured untrained model. Periodic requests use `stage=periodic`, set
`training_performed=true`, and provide absolute incumbent dataset and pipeline paths. The
runner must train from exactly the supplied `dataset_path`; it must not select the last
iteration by convention.

Diagnostic datasets must be materialized before the DataFlow Evolver run. A single
external command may dispatch to a different evaluator for every benchmark, following
the same pattern as Qwen2.5-Math. Dataset paths, prompts, chat templates, decoding
parameters, sample counts, answer extractors, graders, and metric names belong to that
external configuration rather than this request protocol. The runner must reuse the
same prepared diagnostic files at every checkpoint and should write their manifest and
evaluation-config paths into `artifacts` for auditability.

## Result contract

The command exits with code 0 and writes one JSON object to `result.json`:

```json
{
  "model": "Qwen2.5-4B-Base",
  "benchmarks": {
    "gsm8k": {
      "score": 0.71,
      "metric": "accuracy",
      "incorrect_count": 17,
      "bad_cases": [
        {
          "question": "complete benchmark question",
          "model_response": "complete raw model response",
          "parsed_answer": "answer extracted by the benchmark evaluator",
          "correct_answer": "complete reference answer"
        },
        {
          "question": "...",
          "model_response": "...",
          "parsed_answer": "...",
          "correct_answer": "..."
        },
        {
          "question": "...",
          "model_response": "...",
          "parsed_answer": "...",
          "correct_answer": "..."
        }
      ]
    }
  },
  "summary": "optional cross-benchmark diagnosis",
  "artifacts": {
    "checkpoint": "/absolute/path/to/model/checkpoint",
    "diagnostic_manifest": "/absolute/path/to/diagnostic_manifest.json",
    "evaluation_config": "/absolute/path/to/evaluation_config.yaml"
  }
}
```

Every configured benchmark must appear. `score` must be finite, `metric`, `question`,
`model_response`, and `correct_answer` must be non-empty strings, and `incorrect_count`
must be a non-negative integer. `parsed_answer` must be a string but may be empty when
the benchmark evaluator could not extract an answer.
Each benchmark is validated independently and must contain exactly
`min(bad_cases_per_benchmark, incorrect_count)` bad cases. This distinguishes a nearly
perfect model with fewer available failures from an incomplete collector. DataFlow Evolver
never truncates question, raw model response, parsed answer, or correct answer in the
auditable `result.json`. To keep Agent context bounded, the prompt contains response
length/termination metadata plus deterministic head/tail excerpts and the full result
path; PipelineAgent can read the complete response from that file when needed. Runners
may additionally return `response_token_count`, `finish_reason`, and
`length_truncated`; these fields identify outputs that exhausted the generation budget
without clipping the retained evidence. The protocol intentionally omits a per-case `score` (every
retained case is already incorrect) and `format_compliant` (the Qwen2.5-Math parser has no
single format-compliance contract across benchmarks).

Each checkpoint is auditable under:

```text
workspace/runs/<run_name>/downstream_eval/checkpoint_NNN/
  request.json
  result.json
  stdout.log
  stderr.log
```

Validated records are also appended to `downstream_evaluations.jsonl` and included in
`manifest.json`. From the second successful checkpoint onward, the framework records and
prompts with per-benchmark score deltas against the previous successful checkpoint. When
a baseline exists, every periodic result also records deltas against the untrained
baseline. A
timeout, non-zero exit, missing file, malformed JSON, or contract
violation becomes a structured failed evaluation. The static incumbent remains valid and
evolution continues; the prompt explicitly identifies the signal as unavailable instead
of treating an infrastructure failure as a data-quality failure.

## Evaluation hygiene

Bad cases are marked as diagnostic evidence and PipelineAgent is forbidden from copying
or hard-coding them into generated training data. Even with that guard, repeatedly using
the final test split for adaptive pipeline evolution is benchmark leakage at the system
selection level. For a defensible experiment, use fixed development/feedback splits at
periodic checkpoints and run untouched final splits only once after evolution. If the
same split is intentionally used for both, the full-set score may be reported only as a
diagnostic-inclusive secondary result, not as an unbiased held-out result. Exact or
semantic decontamination of generated SFT records does not remove this adaptive model-
selection leakage.

Metrics may differ by benchmark. For example, an external runner may use greedy accuracy
for one task and pass@k for another. DataFlow Evolver preserves each returned `metric`
name and score without forcing a global decoding policy. The external configuration must
define what `incorrect_count` means under that metric and must retain one deterministically
selected, complete failed trajectory per bad case rather than concatenating every sampled
output into the PipelineAgent prompt.
