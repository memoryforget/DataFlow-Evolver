# DataFlow-Instruct Math-3K Qwen control

This directory is the only maintained DataFlow comparison implementation.
Historical 1,500-row baseline scripts are preserved only in the immutable archive:

`/jizhicfs/xiaqifeng/dfe_archive/dataflow_math_baseline_input1500_skip_qgen_seed42_20260823`

The experiment performs one reproducible flow:

1. Extract exactly 1,500 MATH-train and 1,500 GSM8K-train rows from
   DataFlow-Instruct-10K by exact normalized question matching.
2. Rewrite the solutions with Qwen3.8-27B using the archived official
   DataFlow Math pipeline with `ReasoningQuestionGenerator` removed.
3. Keep the final pipeline survivors with non-empty `generated_cot`; require at
   least 1,000 unique records.
4. Full-parameter SFT Qwen2.5-7B for one epoch with the repository's canonical
   downstream runtime.
5. Evaluate all eight full math benchmarks. Competition benchmarks use avg@4.

`reasoning_math_pipeline_skip_qgen.py` is copied byte-for-byte from the archived
executed pipeline and must have SHA256:

`f0fcf3467a928770731082f9d90da8cc8b4e9e45748dcba87fe13bbd6e626673`

The file retains two unused imports from the official source for byte identity,
but it neither constructs nor runs `ReasoningQuestionGenerator`.

Run:

```bash
bash downstream_runtime/comparisons/dataflow_instruct_math3k_qwen38_skip_qgen/run_dataflow_instruct_math3k_qwen_control.sh
```

Default run directory:

`workspace/runs/dataflow_instruct_math3k_qwen38_skip_qgen_seed42`

## Migration prerequisites

The scripts and protocol are tracked, while datasets, model weights, checkpoints and
runtime logs remain outside Git. Before running on another server, provide the paths
recorded in `downstream_runtime/ENVIRONMENT_INVENTORY.md` or update the launcher path
variables. API credentials remain environment-only.
