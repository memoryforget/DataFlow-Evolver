# DataFlow-Instruct-9572 with Qwen3.8 Math rewrite

This comparison measures the effect of the retained Qwen3.8-27B Math CoT
rewrite in the context of the rest of DataFlow-Instruct-10K.

The training corpus contains exactly:

- 7,000 original DataFlow-Instruct rows that did not match the selected
  MATH/GSM8K subset; and
- 2,572 survivors from the Qwen3.8-27B DataFlow Math pipeline.

The 428 selected Math rows rejected by the pipeline are not restored from the
original corpus. The final 9,572 rows preserve the original 10K source order.
Selection and replacement use the recorded one-based `source_line_number`, so
the two portions are disjoint and deterministic.

The source rewrite run is:

`workspace/runs/dataflow_instruct_math3k_qwen38_skip_qgen_seed42`

Run from the repository root:

```bash
bash downstream_runtime/comparisons/dataflow_instruct_9572_qwen38_math_rewrite/run_dataflow_instruct_9572_sft_eval.sh
```

The script performs full-parameter SFT of Qwen2.5-7B for one epoch and then
runs the same eight-benchmark evaluation protocol used by the 2,572-row
comparison. It refuses to reuse non-empty run or record directories.

Expected immutable inputs:

- DataFlow-Instruct-10K SHA256: `76ea3f877b52b8af900adaf51e468ec5dfc473b7f34d53dd4d595cc5f1baa616`
- selected Math-3K SHA256: `0c40ee2ec56cc2071f2e6dcc7d2503f77a790d40b68fc44ecd7163f1086ae814`
- rewritten Math-2572 SHA256: `89efa9195f168284c9d2274341ba98c5feafd66b0117f14601b529c72b37820d`
- pipeline SHA256: `f0fcf3467a928770731082f9d90da8cc8b4e9e45748dcba87fe13bbd6e626673`
