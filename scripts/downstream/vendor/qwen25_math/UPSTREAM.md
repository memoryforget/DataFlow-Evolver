# Vendored Qwen2.5-Math evaluator

This directory contains the minimal math evaluation runtime copied from
<https://github.com/QwenLM/Qwen2.5-Math> at commit
`a45202bd16f1ec06f433442dc1152d0074773465`.

The upstream evaluator is MIT licensed; its license is preserved in `LICENSE`.
The bundled `latex2sympy` component retains its own `LICENSE.txt`.

Local changes are intentionally small:

- pass the CLI `--seed` value explicitly to `vllm.LLM`;
- recognize `aime25` as a distinct benchmark name while reusing the MATH prompt and
  answer parser;
- omit upstream datasets, caches, generated outputs, shell launchers, and tests.
