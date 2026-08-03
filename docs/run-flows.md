# 两条链路的跑通流程

本仓库有**两条**能端到端跑的链路，区别只在一个开关 `search.reward_source`：

| | 链路 A：数学 SFT | 链路 B：PostTrainBench |
|---|---|---|
| 配置 | `configs/default.yaml` | `configs/posttrainbench_gsm8k.yaml` |
| `reward_source` | `train`（默认） | `review` |
| 每节点 reward | 真实 SFT + dev 评测的分数 | ReviewAgent 五维复合分（不占 GPU） |
| 搜索期 GPU | 每个节点都要训 | 完全不训 |
| 终选 | 无（reward 已是真分） | top-K 放大重产 + 真实 SFT，选最优 |
| 终评 | 手动跑 `--mode test` | 编排器自动跑官方 `evaluate.py --limit -1` |
| 评测题集 | 框架物化的 dev（与 test 不相交） | 官方题池，**dev ⊂ test**（见 §3.4） |

选哪条：有充足 GPU、想让每个节点的 reward 就是下游真分 → A；GPU 紧张或要搜得深、能接受用代理
分驱动搜索 → B。

---

## 1. 共同前置

### 1.1 三个隔离环境

编排、训练、评测的依赖互相冲突（`claude-agent-sdk` 与 `llamafactory` 争 pydantic；
`llamafactory+deepspeed` 与 `vllm+xformers` 争 torch 版本线），因此必须分环境，安装见
[README 的「安装」](../README.md#安装)：

- **df-agent**：跑 `python -m dataflow_evolver.main`，即编排器本身。
- **df-train**：全量 SFT。由 `trainer.train_python` 指到。
- **df-eval**：vLLM / inspect 评测。由 `trainer.eval_python` 指到，内部以 `SFT_EVAL_PYTHON`
  传给评测子进程。

> 别把评测依赖装进 df-agent。`grader.py` 需要的 `latex2sympy2` 会把 antlr 降到 4.7.2，
> 打断 df-agent 侧 DataFlow 算子用的 `math-verify`。两边分居是设计使然。

### 1.2 凭据

| 变量 | 用途 | 缺了会怎样 |
|---|---|---|
| `ANTHROPIC_API_KEY` | PipelineAgent 的 Claude Agent SDK | 生成不出流水线，链路走不动 |
| `DF_API_KEY`（配合 `DF_API_URL`） | Profiler / ReviewAgent / Diagnostic | 画像与评审失败 |

算子运行时合成数据用的 LLM 走 `llm.pipeline_llm`，默认回退到同一个 `llm.api` 段。

### 1.3 Agent 工作目录

`workspace/_agent_root/`（技能 + vendored DataFlow 源码）**已随仓库提交**，clone 后开箱即用，
不需要手动准备。

---

## 2. 链路 A：数学 SFT（`configs/default.yaml`）

### 2.1 需要准备的数据

| 路径 | 角色 | 是否入库 |
|---|---|---|
| `data/math_train.jsonl` | dev 题集来源（MATH train，公开训练语料，非 benchmark） | 否，见 [`data/README.md`](../data/README.md) |
| `data/benchmarks/limo_eval/` | 终评的 11 个 benchmark test set | 是（量具，需逐字节稳定） |
| `input/` | 可选输入语料。放了 = grounded 合成；空 = 从零合成 | 内容不入库 |

### 2.2 启动

```bash
python -m dataflow_evolver.main \
  --config configs/default.yaml \
  --task "构建一个高质量的数学问答 SFT 数据集" \
  --log-level INFO 2>&1 | tee workspace/run_$(date +%m%d_%H%M).log
```

### 2.3 启动时框架做的事

1. **TaskProfiler** 把任务文案解析成结构化画像。`trainer.dataset_schema`（`instruction`/`output`）
   会**覆盖** LLM 推断出的 `target_schema`——下游脚本认哪些字段是权威契约，不让模型现编。
2. **物化 dev 题集**：按 `(source_sha, seed, size)` 从 `trainer.dev_set.source` 抽 500 条，
   固定成 `workspace/eval/dev.jsonl` + manifest，之后以 `--dev-set` 显式下传。所有节点因此评在
   同一批题上。口径进 checkpoint，续跑对不上直接抛 `DevSetChanged` 而不是静默改写。
   评测脚本这边不留退路：`--mode dev` 拿不到 `--dev-set` 会在训练开始前直接非零退出。
3. **Init node（node_0）** 与其余节点同口径评分，没有特殊地位。三档启动方式见 `init` 配置。

### 2.4 每一轮（`max_iterations: 10`）

```
UCB1 选节点 → PipelineAgent 写 pipeline.py → 子进程执行产出数据集
   → ReviewAgent Phase-1 评审（不过门槛就自修正重来，上限 self_correction_retries）
   → CappedTrainer 把数据切齐到 target_samples（800）
   → dev_eval_cmd：全量 SFT + 在物化 dev 上评测 → 真分
   → 分数回传 MCTS 做 backprop；bad case 交 DiagnosticAgent 产出改进建议
   → 落 checkpoint
```

切齐这一步不是可有可无的：训练配方在框架外冻结，总优化步数正比于数据条数，不切齐则多产数据的
节点白拿梯度步数，分数不再只反映数据质量。想做「更多数据是否更好」的消融时才把
`generation.enforce_target_samples` 关掉。

### 2.5 终评

train 模式下终评**不由编排器驱动**，手动跑：

```bash
python scripts/llamafactory_math_sft_eval.py --mode test \
  --model-dir <out_dir>/sft_model --output <eval_out>
```

跑完整套 11 个 benchmark（GSM8K / MATH500 / AIME2024 / AMC23 / OlympiadBench / CHMath /
Gaokao / Kaoyan / GradeSchool / MinervaMath / GPQA）。dev 来自 MATH train、test 来自这些
benchmark 的 test split，两者**不相交**。

### 2.6 产物

```
workspace/
├── node_<id>/            # 每个节点的 pipeline.py 与产出数据
├── eval/dev.jsonl        # 物化的 dev 题集 + manifest
├── checkpoint.json       # 树 + 画像 + dev 口径，可续跑
└── run.log
```

---

## 3. 链路 B：PostTrainBench（`configs/posttrainbench_gsm8k.yaml`）

### 3.1 需要准备的东西

- **base model** 本地路径。
- **PTB 任务目录**：`<PostTrainBench>/src/eval/tasks/<bench>/`。里面只有 `evaluate.py` 和
  `benchmark.txt`——**PTB 不提供任何训练数据**，训练集 100% 由流水线现场合成。
- **df-eval 里装 inspect**：

```bash
uv pip install --python <df-eval-python> \
  inspect-ai==0.3.150 \
  "inspect_evals @ git+https://github.com/UKGovernmentBEIS/inspect_evals.git@06001a83e6d7c709c2ede0570dce7f1031a0bad8"
```

- 评测时 `inspect_evals` 会从 HuggingFace 拉题集，离线环境需预置 HF 缓存或 `HF_HOME`。
- healthbench / arenahardwriting 另需 `OPENAI_API_KEY`（judge）。

### 3.2 启动

```bash
export DF_API_KEY=...
export PTB_TIME_BUDGET_SEC=1200          # 快速试；默认 28800（8h）
scripts/run_posttrainbench_local.sh gsm8k /path/to/Qwen3-1.7B
```

runner 查表得出派生的 `--task` 与 dev-limit，export 一组 `PTB_*`，再用 df-agent 起
`main.py --config configs/posttrainbench_gsm8k.yaml`。换 benchmark 只改第一个参数
（`bfcl` / `humaneval` / `aime2025` / `gpqamain` / `healthbench` / `arenahardwriting`）。

### 3.3 三个阶段

**搜索期**（30 轮与 8h 预算谁先到谁停，全程不占 GPU、不接触 benchmark）：

```
UCB1 选节点 → PipelineAgent 写 pipeline.py → 产出 probe_samples（200）条探针数据
   → ReviewAgent 五维打分（correctness .35 / relevance .20 / difficulty .15
      / diversity .15 / schema .15）× 硬指标惩罚 = tier1
   → tier1 即 MCTS reward
```

节点之间唯一的反馈通道是 global memory：把 top-4 节点的**完整** issues 带给下一个节点
（本模式下没有下游真分可用，所以不限条数、不截断）。

**终选**（`tier2_top_k: 3`）：

```
按 tier1 取 top-3 → 各自用同一份 pipeline.py 重产 4000 条（env DF_TARGET_SAMPLES 注入）
   → 塌陷安全阀：重跑 review，dup>0.5 / null>0.3 / parse<0.7 / tier1 掉到 60% 以下 → 跳过换次优
   → 存活候选各跑一次单卡全量 SFT + 官方 evaluate.py --limit 150 → dev 真分
   → 最高者的 sft_model 落为 workspace/final_model
```

**终评**：对 `final_model` 跑一次官方 `evaluate.py --limit -1`（gsm8k 全量 1319 题），只评不训。

时间预算会为「K 次 Tier-2 + 终评」预留；不够就自动缩小 K，极端到 K=1。

### 3.4 两个必须知道的口径问题

**dev ⊂ test**。官方没有独立 dev 池，dev 与终评都评在同一份题池上，而 `--limit N` 是确定性取前
N 条，所以 dev 是 test 的前缀子集。搜索期不受影响（reward 是 tier1，不碰 benchmark），重叠只作用
在「3 选 1」这一步，后果是终评分偏乐观。对外报数须一并说明。完整分析与两种改法见
[`posttrainbench_adaptation.md` §6.1](./posttrainbench_adaptation.md)。

**外部数据默认关闭**。PTB 配置里没有 `agent_sdk.external_data`，而 `load_config` 不做跨文件继承，
所以 Agent 拿不到 `dataset-search` / `data-discovery` 两个 MCP，只能从零合成。要开必须同时配去污染，
理由与配置片段见 [§6.2](./posttrainbench_adaptation.md)。

---

## 4. 配置不继承，这是个坑

`utils/config.py` 的 `load_config` 只读**单份** YAML，不与 `default.yaml` 合并。新配置项加进
`default.yaml` 后，`posttrainbench_gsm8k.yaml` 不会自动获得——代码走 `.get(key, 默认值)`，静默地
落在旧行为上。加新开关时记得两份都过一遍，或明确决定某份不启用并写明理由。

---

## 5. 常见断点

| 现象 | 多半是 |
|---|---|
| 起不来，Agent SDK 报鉴权 | `ANTHROPIC_API_KEY` 没配 |
| 画像/评审失败 | `DF_API_KEY` / `DF_API_URL` 没配 |
| 流水线自检过了、正式执行缺包 | 解释器不一致；见 README「解释器一致性」 |
| 训练脚本非零退出、看不出根因 | 多半是字段对不上。配好 `trainer.dataset_schema` 后会在评审阶段（花 GPU 之前）就拦下 |
| 续跑报 `DevSetChanged` | dev 口径（source/size/seed）变了。这是有意报错，不是 bug |
| 训练脚本立刻报「`--mode dev` 需要 `--dev-set`」 | `trainer.dev_set.source` 留空了。配上它；脚本按契约不自行挑题 |
| PTB 评测拉不到题集 | 离线环境需预置 HF 缓存或 `HF_HOME` |
