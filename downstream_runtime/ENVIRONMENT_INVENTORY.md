# Reproduction environment inventory (updated 2026-08-23)

This tracked document records the exact source-server layout. It contains no
credentials. Paths may be replaced through the portable config template when the
repository is migrated.

## Source

- Repository: `/jizhicfs/xiaqifeng/DataFlow-Evolver`


## Hardware

- 8 x NVIDIA H20, 97,871 MiB each
- Driver 535.247.01; host reports CUDA 12.9
- Intended allocation: GPU 7 embedding; GPUs 0-6 SFT; GPUs 0-3 evaluation

## Models and proxy

- Qwen2.5-7B Base: `/jizhicfs/xiaqifeng/models/Qwen2.5-7B`
- Qwen3-Embedding-8B: `/jizhicfs/xiaqifeng/DataFlow-Evolver/model/Qwen3-Embedding-8B`
- ODA-Math-460k: `/jizhicfs/xiaqifeng/DataFlow-Evolver/data/ODA-Math-460k`

## Input data

- 24,969-row DataFlow-Evolver input:
  `data/math_reasoning_mix_v3/gsm8k_math_dapo3333_bigmath3333_numina3333_train_full_seed42_decontam.jsonl`
- 5,000-row comparison input for methods unable to select 25k -> 5k:
  `data/math_reasoning_mix_v3/gsm8k_math_dapo3333_bigmath3333_numina3333_train_5k_seed42_decontam.jsonl`
- The current 5k file is sampled with seed 42 from the full pool after exact normalized 13-gram decontamination against all configured benchmark test files. Its SHA256 is `8aa4c5cf21747b9b6137f172e329ba3f6084a6b1f099bd485f73816ee87eb7a0`.

## Benchmarks

- Full Qwen2.5-Math evaluator data:
  `/jizhicfs/xiaqifeng/Qwen2.5-Math/evaluation/data`
- Fixed adaptive-feedback diagnostics:
  `artifacts/downstream_eval/math_diagnostics_seed42`

## Environments

- Orchestration/review: `/jizhicfs/xiaqifeng/miniconda3/envs/dfe-review`
- Full SFT: `/jizhicfs/xiaqifeng/miniconda3/envs/dfe-sft`
  - Python 3.11.15
  - torch 2.7.0+cu126
  - LLaMA-Factory 0.9.3 (source tag v0.9.3, commit ca75f1e)
  - DeepSpeed 0.16.9
- Evaluation: `/jizhicfs/xiaqifeng/miniconda3/envs/dfe-math-eval`
  - Alias of existing `qwen25-math-eval`
  - Python 3.11.15
  - torch 2.7.0+cu126
  - vLLM 0.9.2
  - transformers 4.52.4

## Local configuration

- Downstream runtime: `configs/downstream_math_runtime.local.yaml`
- DataFlow experiment config: `downstream_runtime/dataflow_math_periodic.local.yaml`
- Frozen task text: `downstream_runtime/task_math_mix_v3.txt`
- Embedding launcher: `downstream_runtime/start_embedding_server.sh`
- Main launcher: `downstream_runtime/run_math_periodic.sh`
- Formal experiment runbook and all run-specific parameters:
  `downstream_runtime/math_mix_v3_periodic_seed42_formal_v1.md`

## Current services

- Pipeline Agent and ReviewAgent: `gpt-5.4` through the relay.
- Pipeline teacher backend: Qwen3.8-27B at
  `http://28.59.7.41:8000/v1/chat/completions`.
- Teacher service aliases: `teacher` and `Qwen3.8-27B`; experiments use the actual model name
  `Qwen3.8-27B`.
- Verified response contract: the backend returns both message.reasoning and message.content; formal training uses only the complete clean message.content with finish_reason=stop. Hidden reasoning is not training text.
- Embedding service: Qwen3-Embedding-8B on GPU 7 at `http://127.0.0.1:8000/v1`.

## Migration boundary

Git contains the orchestration code, launchers, exact local configuration, portable
configuration template, deterministic diagnostic manifest, and runbooks. The tracked
`scripts/downstream/prepare_math_diagnostics.py` recreates diagnostic rows from a
lawfully installed full benchmark dataset.
It intentionally does not contain model weights, source corpora, workspace runs,
checkpoints, API keys, caches, or large full-benchmark datasets.

On a new server, restore or remap the following external assets before launch:

- Qwen2.5-7B Base and Qwen3-Embedding-8B model directories;
- DataFlow-Instruct-10K, MATH/GSM8K train files, ODA-Math-460k, and full benchmark data;
- the `dfe-review`, `dfe-sft`, and `dfe-math-eval` environments;
- Qwen3.8-27B and embedding API services;
- environment-only relay credentials.

Use `configs/downstream_math_runtime.example.yaml` for a different filesystem root.
The tracked `configs/downstream_math_runtime.local.yaml` is the exact source-server
record and is retained for provenance.
