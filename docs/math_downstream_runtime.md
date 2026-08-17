# Qwen2.5-7B math downstream runtime

This runtime implements the process boundary documented in
`docs/downstream_evaluation.md`. DataFlow Evolver writes `request.json`; this runtime
optionally performs full-parameter SFT, evaluates a fixed diagnostic suite, and writes
the validated `result.json`. LlamaFactory, vLLM, and benchmark dependencies stay outside
the DataFlow Evolver package environment.

## Confirmed assets and training choice

The target directory must contain Qwen2.5-7B **Base** (`Qwen2ForCausalLM`), not an
Instruct checkpoint. For the current 5,000-row decontaminated math mixture, Qwen chat
template lengths are: median 159, p95 769, p99 1,180, and maximum 2,527 tokens. Therefore
`cutoff_len: 4096` retains every current sample. Increasing it to 8K or 32K only raises
activation cost for this dataset.

Before every periodic SFT run, the runtime repeats this tokenizer audit on the actual
incumbent. It records length quantiles, rows above the configured cutoff, and retained
token fraction in `normalization_report.json`. Set `sft.fail_on_truncation: true` if an
experiment should reject rather than record any newly overlong incumbent.

The example uses bf16 full SFT, SDPA, gradient checkpointing, and DeepSpeed ZeRO-2.
Seven-billion-parameter training fits on 80 GB A100s without parameter sharding, so
ZeRO-3 is unnecessary here. ZeRO-2 still shards optimizer and gradient state. The
reference schedule is retained: three epochs, learning rate `1e-5`, cosine decay, and a
0.1 warmup ratio. Sequence packing is disabled to avoid cross-example attention and to
keep the data contract transparent.

## Environments

Create independent training and evaluation environments. The setup script clones an
existing vLLM-compatible environment, then installs only the role-specific additions:

```bash
bash scripts/downstream/setup_math_envs.sh \
  /path/to/miniconda3 \
  /path/to/LlamaFactory \
  /path/to/miniconda3/envs/existing-vllm-env
```

It creates `dfe-sft` and `dfe-math-eval`. Do not install these packages into the
DataFlow Evolver Agent/Review environment. The LlamaFactory checkout is needed only by
this one-time setup command. The script performs a regular installation (not an editable
install), and the runtime subsequently invokes the installed `llamafactory-cli`; it does
not depend on the source checkout. The project also carries its own ZeRO-2 configuration
under `configs/deepspeed/`.

## Fixed diagnostic materialization

Materialize diagnostics once before evolution:

```bash
/path/to/python scripts/downstream/prepare_math_diagnostics.py \
  --benchmark-root /path/to/benchmark_data \
  --output-dir /path/to/DataFlow-Evolver/artifacts/downstream_eval/math_diagnostics_seed42 \
  --fraction 0.1 \
  --seed 42
```

The manifest records source paths, source counts, selected indices, output hashes, and
the parser name for each benchmark. Evaluator-ready AIME24, Minerva Math,
Gaokao2024-Mix, and OlympiadBench source files live under
`benchmark_data/qwen25_math_eval_data/`; benchmark data is not committed to this
repository. AIME-2025 uses its own parser name in the vendored evaluator.
The local `benchmark_data/AIME-2024` parquet contains 960 training-style rows rather than
the official 30-question evaluation set; the preparation script deliberately uses the
Qwen2.5-Math 30-question AIME-2024 file.

The evaluator code is vendored under `scripts/downstream/vendor/qwen25_math` from the
MIT-licensed Qwen2.5-Math evaluator, with its upstream commit and local changes recorded
in `UPSTREAM.md`; no external Qwen2.5-Math checkout is required at runtime.

The suite contains GSM8K, MATH, AMC23, AIME24, AIME25, Minerva Math,
Gaokao2024-Mix, and the Qwen OlympiadBench subset. Core benchmarks use deterministic
greedy accuracy. Small competition sets use `pass@32` with temperature 0.6. Every group
uses `max_tokens_per_call=16384`. The `qwen25-math-cot` prompt already contains Qwen
ChatML markers, so `apply_chat_template` is false; enabling it would double-wrap the
prompt.

## Runtime configuration

Copy `configs/downstream_math_runtime.example.yaml` to an ignored local path and export:

```bash
export DFE_BASE_MODEL=/path/to/Qwen2.5-7B
export DFE_REPO=/path/to/DataFlow-Evolver
export DFE_SFT_CONDA=/path/to/miniconda3/envs/dfe-sft
export DFE_EVAL_CONDA=/path/to/miniconda3/envs/dfe-math-eval
export DFE_DIAGNOSTIC_ROOT=/path/to/math_diagnostics_seed42
```

Then configure DataFlow Evolver:

```yaml
downstream_eval:
  enabled: true
  interval: 2
  command:
    - /path/to/miniconda3/envs/dfe-math-eval/bin/python
    - /path/to/DataFlow-Evolver/scripts/downstream/run_math_downstream.py
    - --config
    - /path/to/DataFlow-Evolver/configs/downstream_math_runtime.local.yaml
  benchmarks: [gsm8k, math, amc23, aime24, aime25, minerva_math, gaokao2024_mix, olympiadbench]
  baseline:
    enabled: true
  bad_cases_per_benchmark: 3
  timeout_sec: 86400
```

Checkpoint 0 evaluates the Base model without training. Periodic checkpoints normalize
only the configured `instruction` and `output` fields from the incumbent dataset, train
from that exact path, and evaluate the resulting checkpoint. Runtime checkpoints,
diagnostics, local configuration, model outputs, and logs belong under `artifacts/` and
remain ignored by Git.

If an embedding server occupies one GPU during evolution, exclude that GPU from
`sft.cuda_visible_devices`. The default example lists all eight GPUs and assumes they are
free; the external runner does not stop or restart unrelated GPU services.
