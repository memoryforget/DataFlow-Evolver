# DataFlow-Evolve

多智能体、MCTS 驱动的自动数据工程框架（双反馈环）。

根据高层任务描述，自动生成 / 变异 DataFlow 流水线 DAG，并通过 **双反馈环**——
快速的"左移"数据评审（Phase-1）+ 真实的下游模型训练评估（Phase-2，产出可执行诊断报告）——
不断自我修正，最终产出高质量训练数据集。

> 设计蓝图见 [`TODO.md`](./TODO.md)。

---

## 架构一览

```
TaskProfiler ──> TaskProfile
                     │
        ┌────────────▼─────────────  MCTS 主循环（main.py）  ───────────┐
        │  select(UCB1) → PipelineAgent(Agent SDK 写流水线) → 执行       │
        │     → Phase-1 ReviewAgent 快评审 → Phase-2 训练+DiagnosticAgent │
        │     → backpropagate(真实得分) → GlobalMemory 更新              │
        └──────────────────────────────────────────────────────────────┘
```

- **PipelineAgent** 不是普通的 chat API 调用，而是驱动 **Claude Agent SDK**（无头），
  让 SDK 内的 Claude 借助本地技能 `dataflow-dev`
  以及 **MCP 算子源码参考**（`dataflow-operators` in-process server）
  编写自定义算子并构建完整可运行的 DataFlow 流水线 `.py` 到节点工作目录。
- **DownstreamTrainerWrapper** 是 **可插拔外部脚本桥接**：框架不内置训练逻辑，
  而是按 I/O 契约子进程调用你提供的"训练+评估"脚本。

---

## 安装

本项目将 **编排层（Agent）** 和 **训练层（Train）** 的依赖分离到不同的 optional extras，
因为 `claude-agent-sdk` 依赖的 `pydantic>=2.11` 与 `llamafactory` 要求的 `pydantic<=2.10.6` 冲突，
**不能共存于同一环境**。实际部署需要两个 conda 环境。

### 前置：安装 uv（全局，一次性）

```bash
# 推荐：独立安装（不污染任何 Python 环境）
curl -LsSf https://astral.sh/uv/install.sh | sh
# 或：pipx install uv
```

### 环境 1：Agent（编排主进程）

```bash
conda create -n df-agent python=3.11 -y
conda activate df-agent

uv pip install -e ".[agent]"     # openai / pyyaml / pydantic / claude-agent-sdk
uv pip install -e ".[dev]"       # pytest（可选）
```

### 环境 2：Train（训练/推理子进程）

```bash
conda create -n df-train python=3.11 -y
conda activate df-train

uv pip install -e ".[train]" \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  --index-strategy unsafe-best-match
```

> **CUDA 兼容性**：`torch>=2.5,<2.6` 使用 cu121 wheel，兼容驱动 CUDA 12.1+（向下兼容）。
> 若机器 CUDA < 12.1，切换到对应的 PyTorch index（如 `cu118`）。

### 跨环境桥接

训练子进程由主进程 `subprocess` 调用，通过配置指定 train 环境的 python 路径：

```yaml
# configs/default.yaml
trainer:
  train_python: /path/to/miniconda3/envs/df-train/bin/python
```

或通过环境变量：

```bash
export TRAIN_PYTHON=/path/to/miniconda3/envs/df-train/bin/python
```

留空则使用当前环境的 `python`（适用于单环境开发/测试场景）。

### 一次性准备 Agent 工作目录 `workspace/_agent_root/`

Agent SDK 的 `cwd` 指向共享的 `workspace/_agent_root/`，其中需并存技能与 DataFlow 源码副本
（`dataflow-dev` 技能假设 cwd 即 DataFlow 仓库根）。

**技能文件已随仓库提交**（`workspace/_agent_root/.claude/skills/` 下），无需额外拷贝。

**需要手动准备的只有 DataFlow 源码副本**（二选一）：

```bash
# 方式 A（推荐）：直接拷贝源码目录到 _agent_root 下
# MCP server 通过 sys.path 自动 import，无需额外安装
cp -r <DataFlow>/dataflow workspace/_agent_root/dataflow

# 方式 B：editable install（同时满足 import 和文件浏览需求，但 agent 浏览源码路径不在 cwd 下）
cd <DataFlow> && pip install -e .
```

> `_agent_root/dataflow/` 是上游 DataFlow 的拷贝，会随上游更新而变旧，必要时手动重拷同步。
> MCP server（`llm/mcp_operators.py`）启动时动态加载 registry，提供已有算子源码供 PipelineAgent 作为编码风格参考。

---

## 配置

见 [`configs/default.yaml`](./configs/default.yaml)。支持 `${ENV}` / `${ENV:default}` 环境变量展开。

关键环境变量：

| 变量 | 用途 |
|---|---|
| `ANTHROPIC_API_KEY` | **PipelineAgent 的 Agent SDK 运行所必需**（未设置则无法实际生成流水线） |
| `DF_API_KEY` | 辅助 Agent（Profiler/Review/Diagnostic）调用的 OpenAI-compatible API 密钥 |
| `DF_API_URL` / `DF_MODEL` | 辅助 Agent 的 API 地址 / 模型名（可选，有默认值） |
| `CLAUDE_MODEL` | Agent SDK 使用的 Claude 模型（默认 `claude-opus-4-6`） |
| `TRAIN_PYTHON` | Train 环境的 python 路径（编排主进程用它 subprocess 调训练脚本） |

---

## 运行
export IS_SANDBOX=1
export ANTHROPIC_BASE_URL=
export ANTHROPIC_API_KEY=
export DF_API_KEY=
```bash
python -m dataflow_evolve.main \
  --config configs/default.yaml \
  --task "构建一个高质量的数学问答 SFT 数据集" \
  --log-level INFO
```

输出：最优流水线所在节点目录、`field_flow` 与平均奖励。

---

## 外部训练脚本契约（Phase-2）

`DownstreamTrainerWrapper` 按如下契约子进程调用你的脚本：

```
<train_python> <eval_cmd> --dataset <ds.jsonl> --output <out_dir>
```

其中 `<train_python>` 由 `trainer.train_python` 配置（留空则用当前环境 python），
`<eval_cmd>` 对应 `trainer.dev_eval_cmd`（MCTS 迭代）或 `trainer.test_eval_cmd`（终评）。

脚本必须在 `<out_dir>/results.json` 回写：

```json
{
  "score": 0.0,
  "metric_name": "accuracy",
  "bad_cases": [
    {"input": "...", "prediction": "...", "ground_truth": "...", "error_type": "..."}
  ]
}
```

- `score` 作为该节点回传给 MCTS 的奖励。
- `bad_cases` 交给 DiagnosticAgent 归因，产出下一轮变异的可执行建议。
- 超时 / 非零退出 / 缺 `results.json` 均被降级为 `TrainResult(failed=True)`，**不会拖垮主循环**。

参考实现与最小可跑示例见 [`scripts/reference_train_eval.py`](./scripts/reference_train_eval.py)。

---

## 数学 SFT 下游链路（LlamaFactory + Math Benchmark）

当前已配置的 Phase-2 真实链路：**Full SFT + 三 Math Bench 评测**。

### 安装训练/评测依赖

```bash
# 使用独立的 train 环境（见上方「安装」章节）
conda activate df-train
uv pip install -e ".[train]" \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  --index-strategy unsafe-best-match
```

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SFT_BASE_MODEL` | `Qwen/Qwen3.5-9B` | 基座模型（HuggingFace ID 或本地路径） |
| `SFT_NUM_EPOCHS` | `3` | 训练 epoch 数 |
| `SFT_LEARNING_RATE` | `2e-5` | 学习率 |
| `SFT_BATCH_SIZE` | `4` | per-device batch size |
| `SFT_GRAD_ACCUM` | `4` | 梯度累积步数 |
| `SFT_NUM_GPUS` | `8` | 训练 GPU 数（torchrun nproc） |
| `EVAL_TP_SIZE` | `1` | vLLM 推理 tensor parallel size |
| `EVAL_DP_SIZE` | `8` | vLLM 推理 data parallel size |
| `EVAL_MAX_MODEL_LEN` | `4096` | vLLM 最大序列长度 |
| `EVAL_MAX_NEW_TOKENS` | `2048` | 评测推理最大生成 token 数 |
| `MATH_TRAIN_PATH` | `data/math_train.jsonl` | dev 评测题库路径（MATH competition train 7500 条） |
| `HF_CACHE_DIR` | `None` | HuggingFace datasets 缓存目录 |

### 运行方式

脚本遵循 Phase-2 契约，由框架自动调用（`trainer.dev_eval_cmd`）：

```bash
# dev 模式（MCTS 迭代，框架自动调用）：MATH train 全量 7500 题
python scripts/llamafactory_math_sft_eval.py --mode dev --dataset <ds.jsonl> --output <out_dir>

# test 模式（终评，手动调用一次）：跑 GSM8K test + MATH-500 + AIME 2024
python scripts/llamafactory_math_sft_eval.py --mode test --dataset <ds.jsonl> --output <out_dir>
```

内部流程：
1. **格式转换**：DataFlow JSONL (`instruction` + `generated_cot`) → LlamaFactory alpaca 格式
2. **Full SFT**：Qwen3.5-9B + DeepSpeed ZeRO-2（8 GPU），全量微调
3. **评测**：vLLM 加载 fine-tuned 模型推理+判分（Qwen2.5-Math prompt 格式）
   - dev：MATH train 全量（7500 条公开竞赛训练题，非 benchmark test set）
   - test：GSM8K test (1319) + MATH-500 (500) + AIME 2024 (30)
4. **输出**：bench 平均 accuracy 作为 `score` 写入 `results.json`

### 双机训练部署（`dispatch: queue`）

LIMO 配置的 Qwen2.5-32B Full SFT 需**双机 16 卡**。但 MCTS 编排器是 node0 上的单进程，
无法直接驱动双机 torchrun（node1 上没有进程会合，会卡在 rendezvous 直到超时）。

解决方式：**文件队列 + 常驻 worker 解耦**。两机各起一个常驻 worker 守护进程，轮询共享 FS 上的
队列目录；编排器只把训练任务投递到队列并轮询结果。要求两机之间有**共享文件系统**（NFS/Lustre/GPFS
等），`workspace` 对两机可见。

**1. 打开 queue 调度**（`configs/default.yaml`）：

```yaml
trainer:
  dispatch: queue                      # 默认 local（单机）；改为 queue 启用双机解耦
  queue_dir: ./workspace/train_queue   # 须在共享 FS 上，对两机可见
```

**2. 两机各起一个 worker**（`MASTER_ADDR` 填 node0 对 node1 可路由的 IP）：

```bash
# node0（主，rank0：负责认领任务 + 写完成标记）
python scripts/dual_node_worker.py --rank 0 --nnodes 2 \
    --master-addr <IP0> --master-port 29500 --gpus 8 \
    --queue-dir <shared>/workspace/train_queue --python <train_env_python>

# node1（从，rank1：等 running 标记出现即启动同一命令，靠 torchrun 会合）
python scripts/dual_node_worker.py --rank 1 --nnodes 2 \
    --master-addr <IP0> --master-port 29500 --gpus 8 \
    --queue-dir <shared>/workspace/train_queue --python <train_env_python>
```

`--gpus` 是**本机**卡数（如 8）；worker 内部换算 `SFT_NUM_GPUS = gpus × nnodes`（=16 集群总卡数），
并为训练子进程设好 `NODE_RANK` / `MASTER_ADDR` / `NNODES` 等分布式 env（修掉了 eval 脚本原本
`NODE_RANK`/`MASTER_ADDR` 缺失的问题）。

**3. 在 node0 起编排器**（照常，`python -m dataflow_evolve.main ...`）。编排器投递任务后阻塞轮询
`queue_dir/done/`；两 worker 拉起双机训练，rank0 训练结束写 done，编排器读回 `results.json`。

要点与容错：
- 双机需 100G+ RoCE/InfiniBand 互连，否则 ZeRO-3 通信成瓶颈（脚本 docstring 已注明）。
- 编排器超时会写 `cancel` 标记，worker 检测后杀训练进程树，避免僵尸 GPU 进程。
- worker 重启后读 `queue_dir/worker_state/<rank>.seen` 跳过已完成任务。
- `dispatch: local`（默认）时行为与旧版完全一致，单机直接拉起训练子进程。

### timeout 说明

`configs/default.yaml` 的 trainer 块有两个独立超时：

- `timeout_sec: 3600`（1h）——**流水线执行**（纯数据合成）超时，由执行器使用；数据合成通常几分钟完成。
- `train_timeout_sec: 25200`（7h）——**dev/test 评测**（含内层 Full SFT 训练 + 三 bench 推理）超时。

内层 `llamafactory_math_sft_eval.py` 的 SFT 训练子进程有 6h 超时（脚本内 `torchrun`），故外层 7h > 内层 6h + 评测，训练能完整跑完。每轮 MCTS 迭代调用一次 `dev_eval_cmd`；超过 `train_timeout_sec` 则强制终止进程树并标记该节点 `failed`，不阻塞主循环。

### 评测 Benchmark 详情

| Benchmark | 题数 | 难度 | 答案格式 | 判分方式 |
|-----------|------|------|----------|----------|
| GSM8K | 1319 | 小学-初中 | `#### <number>` | 数值精确匹配 |
| MATH-500 | 500 | 高中竞赛 | `\boxed{...}` | LaTeX 语义等价（math_verify） |
| AIME 2024 | 30 | 竞赛级 | 整数 0-999 | 整数精确匹配 |

最终 `score = mean(gsm8k_acc, math500_acc, aime_acc)`。

---

## 防数据泄露：dev / test 评测纪律

MCTS 反复用同一 benchmark 当奖励会**对它过拟合**；若再用同一 benchmark 报告结果，分数虚高、不可泛化。
因此评测必须分两套（设计见 [`docs/design-decisions.md`](./docs/design-decisions.md) §3）：

- **`dev`**（held-in）：**只**给 MCTS 当奖励信号，允许被「刷」——配 `trainer.dev_eval_cmd`。
- **`test`**（held-out）：**全程对 loop 不可见**，仅在选出最优 recipe 后**手动跑一次**用于最终报告——配 `trainer.test_eval_cmd`。
- **dev/test 切分由你的训练脚本内部负责**，框架把 benchmark 当黑箱、只决定调哪条命令（不传 split 路径、不介入切分）。
- **去污染（decontamination）**：生成数据的种子/源语料须与 `test` 题目做 n-gram 重叠过滤，剔除重叠项（堵内容泄露）。

> ReviewAgent（Phase-1）只查表面 schema/相关性，**不能**单独替代 Phase-2 真实反馈——
> 正解是保留 Phase-2 但评测放 dev、test 严格隔离。

---

## 数据池（可选副产品，默认关闭）

`configs/default.yaml` 的 `data_pool` 段可开启跨节点高质量数据的累积池。定位见
[`docs/design-decisions.md`](./docs/design-decisions.md)：

- **交付主线是「最优 pipeline」**；池是可选成品，**与 MCTS 奖励解耦**（奖励保持 per-node）。
- 池价值 = 可直接用的成品 + 回收已花算力 + 承载诊断驱动的累积覆盖（单节点结构上做不到）。
- ⚠️ **脚手架阶段**：`core/data_pool.py` 已有接口/去重/上限，但**尚未接入主循环**；
  「池 vs 单 pipeline 谁更强」是**待受控实验检验的假设**（须等量对比 + 多样性量化），**非既定结论**。

---

## 测试

```bash
# 单元测试只需 agent 环境（不依赖 torch/llamafactory）
conda activate df-agent
python -m pytest tests/ -q
```

> **Windows 提示**：用 `conda run` 跑会 spawn 子进程的命令可能崩溃；
> 请直接用环境内解释器：`"<conda>/envs/df-agent/python.exe" -m pytest tests/ -q`。

覆盖：配置/解析工具、UCB1 树搜索、辅助 Agent（桩后端）、执行与训练引擎、
以及不依赖 API 密钥的主事件环集成冒烟测试。

---

## 目录结构

```
dataflow_evolve/
├── agents/      # base / task_profiler / pipeline_agent(SDK 驱动) / review / diagnostic
├── core/        # schemas / tree_node / tree_manager / global_memory
├── engine/      # execution_wrapper / trainer_wrapper
├── llm/         # serving(API/vLLM) / agent_sdk / mcp_operators / prompts
├── utils/       # config / logging / parsing
└── main.py      # 双反馈事件环编排器
configs/         # default.yaml
scripts/         # reference_train_eval.py（外部训练脚本契约示例）
workspace/       # 运行产物 + _agent_root（Agent SDK cwd）
tests/
```
