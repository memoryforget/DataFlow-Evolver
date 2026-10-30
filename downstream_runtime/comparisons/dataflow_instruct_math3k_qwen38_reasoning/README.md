# DataFlow-Instruct Math-3K with Qwen3.8 reasoning

Uncommitted comparison for testing the Qwen3.8-27B reasoning field in SFT.

The official skip-question-generation Math pipeline is reused byte-for-byte.
Only the `ReasoningAnswerGenerator` receives a separate serving instance whose
response formatter accepts either `message.reasoning_content` or
`message.reasoning` and produces:

```text
<think>{reasoning}</think>
<answer>{content}</answer>
```

All filter and classifier operators keep the original serving formatter so
their JSON responses are not wrapped. Before SFT, every retained row must pass
a strict non-empty think/answer contract and contain a boxed answer.

Run from the repository root:

```bash
bash downstream_runtime/comparisons/dataflow_instruct_math3k_qwen38_reasoning/run_reasoning_math3k_control.sh
```

The experiment uses the same exact Math-3K extraction, Qwen2.5-7B SFT config,
and eight-benchmark evaluation protocol as the previous comparisons.
