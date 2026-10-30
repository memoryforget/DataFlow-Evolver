# DataFlow-Instruct original Math-3K control

This comparison trains directly on the exact 3,000 MATH/GSM8K rows extracted
from DataFlow-Instruct-10K, without rewriting their original CoT answers.

It is the matched control for:

- `dataflow_instruct_math3k_qwen38_skip_qgen`: 2,572 retained rewritten rows;
- `dataflow_instruct_9572_qwen38_math_rewrite`: those 2,572 rows plus the
  untouched 7,000-row remainder.

The corpus contains 1,500 MATH and 1,500 GSM8K rows. The source file must have
SHA256 `0c40ee2ec56cc2071f2e6dcc7d2503f77a790d40b68fc44ecd7163f1086ae814`.

Run immediately:

```bash
bash downstream_runtime/comparisons/dataflow_instruct_math3k_original/run_dataflow_instruct_math3k_original_sft_eval.sh
```

Queue it after the 9,572-row comparison:

```bash
bash downstream_runtime/comparisons/dataflow_instruct_math3k_original/wait_then_run_after_9572.sh
```

The dependency wrapper starts this control only if the upstream result reports
`status=ok` and `stage=completed`. It aborts if the upstream task exits without
a successful terminal status.
