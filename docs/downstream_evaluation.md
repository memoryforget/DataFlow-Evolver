# Periodic downstream evaluation

Periodic downstream evaluation adds an expensive, task-level signal without importing a
training or benchmark stack into DataFlow Evolver. Static Review still runs on every
candidate. After every configured number of completed iterations, the current
best-so-far incumbent dataset is exported to an external command for SFT and evaluation:

```text
iteration 1 -> static Review -> incumbent update
iteration 2 -> static Review -> incumbent update
iteration 3 -> static Review -> incumbent update
            -> external SFT + benchmark evaluation of that incumbent
            -> validated scores and complete bad cases
iteration 4 -> PipelineAgent receives static Review + downstream feedback
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
  interval: 3
  command:
    - /opt/conda/envs/downstream/bin/python
    - /absolute/path/to/run_sft_and_eval.py
  benchmarks: [gsm8k, math, math500, aime]
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

`request.json` contains protocol version 1, the one-based checkpoint and incumbent
iterations, absolute candidate dataset and pipeline paths, the task contract, expected
benchmark names, and the required bad-case count. The runner must train from exactly the
supplied `dataset_path`; it must not select the last iteration by convention.

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
          "model_output": "complete model output",
          "correct_answer": "complete reference answer"
        },
        {
          "question": "...",
          "model_output": "...",
          "correct_answer": "..."
        },
        {
          "question": "...",
          "model_output": "...",
          "correct_answer": "..."
        }
      ]
    }
  },
  "summary": "optional cross-benchmark diagnosis",
  "artifacts": {
    "checkpoint": "/absolute/path/to/model/checkpoint"
  }
}
```

Every configured benchmark must appear. `score` must be finite, `metric` and all bad-case
fields must be non-empty strings, and `incorrect_count` must be a non-negative integer.
Each benchmark must contain exactly
`min(bad_cases_per_benchmark, incorrect_count)` bad cases. This distinguishes a nearly
perfect model with fewer available failures from an incomplete collector. DataFlow Evolver
never truncates question, model output, or correct answer text before giving it to
PipelineAgent.

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
prompts with per-benchmark score deltas against the previous successful checkpoint. A
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
same split is intentionally used for both, report it as an optimization-set result rather
than an unbiased final evaluation.
