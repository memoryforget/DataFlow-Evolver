# `data/` in the PTB Official Default Profile

This directory stores local benchmark mirrors and reference files. The default
PTB official profile does **not** load these files for scoring; official scores
come from PostTrainBench `evaluate.py`. Other future evaluator backends may define
their own data contract.

## Data Roles

| Role | Current PTB default behavior |
| --- | --- |
| Training input | None. `project.input_dir` is empty by default, so the agent synthesizes from scratch. |
| Per-node validation | Official PTB evaluator with `--limit ${PTB_DEV_LIMIT:-150}`. |
| Final test | Official PTB evaluator with `--limit -1`. |
| Decontamination reference | Local benchmark test mirrors, defaulting to GSM8K. |

## Files

| Path | Role |
| --- | --- |
| `benchmarks/limo_eval/gsm8k/test.jsonl` | Default GSM8K test mirror for n-gram decontamination. |
| `benchmarks/limo_eval/*/test.jsonl` | Historical/local benchmark mirrors; not used by PTB official scoring. |
| `gsm8k_dev.jsonl`, `math_train.jsonl` | Legacy dev sources for the removed custom math evaluator. |
| `gsm8k_train_full.jsonl`, `gsm8k_train_pool.jsonl` | Legacy input/dev split artifacts; not used by the current PTB default unless explicitly configured. |

## PTB Validation

`trainer.dev_set.source` is intentionally absent from `configs/default.yaml`.
PostTrainBench task evaluators do not accept an external dev file; they read the
official benchmark pool internally.

The bridge calls:

```bash
python scripts/posttrainbench_sft_eval.py --mode dev \
  --dev-limit ${PTB_DEV_LIMIT:-150} \
  --benchmark ${PTB_BENCHMARK:-gsm8k}
```

and final evaluation calls:

```bash
python scripts/posttrainbench_sft_eval.py --mode test \
  --benchmark ${PTB_BENCHMARK:-gsm8k}
```

The official evaluator writes aggregate metrics only. The framework does not
parse per-sample benchmark logs into `bad_cases`, to avoid feeding benchmark
content back into MCTS.

## Hugging Face Access

For Hugging Face based inspect_ai tasks, `scripts/posttrainbench_sft_eval.py`
defaults to:

```bash
HF_ENDPOINT=https://hf-mirror.com
```

unless the caller already set `HF_ENDPOINT`. A local HF cache or direct HF access
also works; `mihomo`/proxy is only a fallback when neither cache nor mirror is
reachable.

## Decontamination

PTB forbids using benchmark test questions or answers for training. Local test
mirrors in this directory may be used for decontamination checks only.

Default config:

```yaml
review:
  decontamination:
    reference_files:
      - ${PTB_DECONTAM_REF:data/benchmarks/limo_eval/gsm8k/test.jsonl}
    ngram: 13
    max_rate: 0.02
```

When changing `PTB_BENCHMARK`, also change `PTB_DECONTAM_REF` to the matching
benchmark test mirror. Otherwise the run will have a false sense of protection.

## Legacy Data

Legacy local dev/input files are kept only for reproducibility and old reports.
They are not part of the current PTB official default path. Re-enable them only
for an explicit ablation or another downstream backend with a documented
data-use contract.
