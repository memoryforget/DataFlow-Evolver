# `data/` —— 评测用数据，不入库

这里放的是**基准数据**，不是框架代码：跑哪个 benchmark 就取哪份，各人各机器不同，
所以除本说明和 `benchmarks/` 外一律不进 git（见根目录 `.gitignore`）。

框架本身对这些文件没有编译期依赖——路径全部经配置传入，缺了会在**启动时**报错并指到这里，
而不是跑到一半才炸。

## 数据在框架里的三个角色

搞清角色比记住路径重要，三者互不替代：

| 角色 | 配置项 | 说明 |
|---|---|---|
| 合成原料 | `project.input_dir`（`input/`） | 可选。有语料 = corpus-grounded 合成，空 = 从零合成 |
| dev 题库 | `trainer.dev_set.source` | MCTS 的**考卷**。框架启动时物化成 `workspace/eval/dev.jsonl` |
| test 集 | `trainer.test_eval_cmd` 内部路径 | 终评，全程对搜索不可见 |

注意 dev 题库**不是训练语料**——训练数据是每个节点现合成的，我们没有可切的 train split。
它只用来出题打分，所以必须与 test 集不相交（见 `docs/design-decisions.md` §4.1）。

## `math_train.jsonl`（dev 题库，MATH 路径）

MATH competition 的 **train** split，7500 条，公开竞赛训练题，不是 benchmark test set。

获取：从 [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math) 仓库取
`evaluation/data/math/train.jsonl`，放到 `data/math_train.jsonl`。

需要的字段（`scripts/math_benchmarks/math_train.py` 按此读取）：

```json
{"problem": "题面", "solution": "含 \\boxed{} 的解答", "answer": "最终答案（可选）", "level": "Level 3", "subject": "Algebra"}
```

`answer` 缺失时从 `solution` 的 `\boxed{}` 里抽。放在别处的话改
`configs/*.yaml` 的 `trainer.dev_set.source` 即可，不必落在这个路径。

## `benchmarks/limo_eval/*/test.jsonl`（终评集）

11 个数学/推理 benchmark 的 test split，由 `scripts/math_benchmarks/additional_benchmarks.py`
按固定相对路径加载（`gsm8k/test.jsonl`、`aime/test.jsonl` 等，映射表见该文件的
`_DEFAULT_LOCAL_BENCHMARK_PATHS`）。

这批目前仍在库里（约 2.3 MB）。它们是**终评**的输入，全程不参与搜索循环。

## PostTrainBench 路径

不用这个目录。dev/test 都由 `scripts/posttrainbench_sft_eval.py` 转交 PostTrainBench
官方 `evaluate.py`，题目来自 `PTB_TASK_DIR` 指向的任务目录。
