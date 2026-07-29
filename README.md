# DataFlow-Evolver

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

本项目将 **编排层（Agent）**、**SFT 训练层（df-train）**、**vLLM 评测层（df-eval）**
分离到不同的 optional extras：

- `claude-agent-sdk` 与 `llamafactory` 的 `pydantic` 约束冲突，不能和 Agent 混装。
- `llamafactory + deepspeed` 与 `vllm + xformers` 依赖的 PyTorch 版本线不同，
  不建议再放进同一个环境。

推荐部署为三个 conda 环境；如果你只做其中一个阶段，也可以只创建对应环境。

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

# 运行 vendored DataFlow 流水线所需（流水线子进程用的就是本环境的解释器）。
# torch 仅为满足 dataflow.serving 的顶层导入，df-agent 不做 GPU 计算 —— CPU-only 轮子即可（~200MB）。
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e ".[agent-runtime]"

# 可选但推荐：解锁全部内置算子（202/202 可导入，MCP 目录 124 → 197）。
# 不装则 MCP 的 CATEGORY_GUIDE 会点名推荐若干不可导入的算子，Claude 用了就撞
# ModuleNotFoundError，而缺依赖的崩溃自纠循环修不好，只会白烧自修正次数。
uv pip install -e ".[agent-operators]"

uv pip install -e ".[dev]"       # pytest（可选）
```

### 环境 2：df-train（SFT 训练子进程）

```bash
conda create -n df-train python=3.11 -y
conda activate df-train

uv pip install -e ".[df-train]" \
  --index-url https://mirrors.aliyun.com/pypi/simple/ \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  --index-strategy unsafe-best-match
```

> **CUDA 兼容性**：`df-train` 的 `torch>=2.5,<2.6` 使用 cu121 wheel，兼容驱动 CUDA 12.1+（向下兼容）。
> 若机器 CUDA < 12.1，切换到对应的 PyTorch index（如 `cu118`）。

### 环境 3：df-eval（vLLM 推理 / benchmark 评测）

```bash
conda create -n df-eval python=3.11 -y
conda activate df-eval

uv pip install -e ".[df-eval]"   --index-url https://mirrors.aliyun.com/pypi/simple/   --extra-index-url https://download.pytorch.org/whl/cu128   --index-strategy unsafe-best-match
```

> **CUDA 兼容性**：`df-eval` 使用 **vLLM 0.19.1 + transformers 4.57.x + torch 2.10.0**，
> 推荐使用 PyTorch 官方 CUDA 12.8 wheel（兼容 CUDA 12.1+ 驱动）。
> 
> **重要**：不要使用 transformers 5.x，它删除了 `all_special_tokens_extended` 属性导致 vLLM 报错。
> - 需设置环境变量 `VLLM_WORKER_MULTIPROC_METHOD=spawn` 避免 CUDA fork 错误（框架已自动设置）
> - DP>1 时独立脚本自动启动 `vllm serve` + OpenAI API 模式推理（不再使用 external_launcher）
> 
> 评测配置：默认使用 TP=4（Tensor Parallel）+ DP=2（Data Parallel），共 8 张 GPU。
> 系统 CUDA 驱动需 >= 12.1。

### 跨环境桥接

训练子进程由主进程 `subprocess` 调用。SFT 与 benchmark 评测可分别指定不同环境的 python 路径：

```yaml
# configs/default.yaml
trainer:
  train_python: /path/to/miniconda3/envs/df-train/bin/python
  eval_python: /path/to/miniconda3/envs/df-eval/bin/python
```

> 这两项**只识别 YAML 键**，不是直接读取的环境变量。若想用环境变量注入，请在 YAML 里写成
> `train_python: ${TRAIN_PYTHON:}` / `eval_python: ${EVAL_PYTHON:}`（`utils/config.py` 会做 `${ENV:default}` 展开）。
> `eval_python` 在内部会以 `SFT_EVAL_PYTHON` 传给评测子进程。

留空则使用当前环境的 `python`。通常只建议在单环境开发/测试场景下这样做。

### 一次性准备 Agent 工作目录 `workspace/_agent_root/`

Agent SDK 的 `cwd` 指向共享的 `workspace/_agent_root/`，其中并存技能与 DataFlow 源码副本
（`dataflow-dev` 技能假设 cwd 即 DataFlow 仓库根）。

**这些都已随仓库提交，`git clone` 后开箱即用，无需手动准备：**

- **技能文件**：`workspace/_agent_root/.claude/skills/` 下。
- **DataFlow 源码副本**：vendored 在 `workspace/_agent_root/dataflow/`（Apache-2.0，署名与版本见该目录下
  `LICENSE` / `VENDOR.md`）。

编排/agent 环境依赖 DataFlow 的地方有**两处**：

1. **MCP server**（`llm/mcp_operators.py` 加载算子 registry，把已有算子源码作为 PipelineAgent 的编码风格参考）——只需能 `import dataflow`。
2. **流水线执行**（`SubprocessExecutionWrapper` 用 `sys.executable` 起子进程跑生成的 `pipeline.py`）——需要其第三方运行时依赖，即 `.[agent-runtime]`。

把源码放在 `_agent_root/` 下同时满足两件事：① MCP server 经 `sys.path` 注入即可 `import dataflow`；
② 源码就在 agent 的 `cwd` 下，`dataflow-dev` 技能“cwd 即 DataFlow 根目录”的假设成立、agent 可直接浏览源码。

> **解释器一致性**：Claude 会用裸 `python` 跑 `DF_COMPILE_ONLY=1 python <node_dir>/pipeline.py` 自检，
> 而正式执行用的是编排器的 `sys.executable`。二者若解析到不同环境，自检会在依赖齐备的那个环境里通过、
> 正式执行却因缺包崩溃。`llm/agent_sdk.py::_agent_env()` 通过 `ClaudeAgentOptions.env`
> 把 `sys.executable` 所在目录前置到 SDK 子进程的 `PATH`，使两者始终一致。

> vendored 副本会随上游更新而变旧；需要同步时按 `workspace/_agent_root/dataflow/VENDOR.md` 的步骤重拷并更新版本/commit 标记。

---

## 配置

见 [`configs/default.yaml`](./configs/default.yaml)。支持 `${ENV}` / `${ENV:default}` 环境变量展开。

关键环境变量：

| 变量 | 用途 |
|---|---|
| `ANTHROPIC_API_KEY` | **PipelineAgent 的 Agent SDK 运行所必需**（未设置则无法实际生成流水线） |
| `DF_API_KEY` | 辅助 Agent（Profiler/Review/Diagnostic）调用的 OpenAI-compatible API 密钥；API 地址/模型名在 `llm.api` 段配置（非环境变量） |

> Claude 模型不由环境变量控制，而是 `configs/default.yaml` 的 `agent_sdk.model`（当前默认 `claude-opus-4-8`）。
> df-train / df-eval 的 python 路径同样是 YAML 键 `trainer.train_python` / `trainer.eval_python`（见上"跨环境桥接"），
> 不是直接读取的 `TRAIN_PYTHON` / `EVAL_PYTHON` 环境变量。

---

## 运行

```bash
export IS_SANDBOX=1
export HF_HOME="/vepfs-mlp2/c20250602/500050/lh/xqf/DataFlow-Evolver/huggingface"
export ANTHROPIC_BASE_URL=http://123.129.219.111:3000
export ANTHROPIC_API_KEY=<你的 key>
export DF_API_KEY=<你的 key>

# 多机训练（dispatch: ssh）必需：把 NCCL/GLOO 钉到跨机可达的网卡，
# 需与 trainer.ssh.master_addr 所在网卡一致（本机 master_addr=198.19.158.245 → eth5）；
# 不设会导致 NCCL 自动选到管理网卡(如 eth0)而跨机连不通。单机训练可忽略。
export NCCL_SOCKET_IFNAME=eth5
export GLOO_SOCKET_IFNAME=eth5

python -m dataflow_evolver.main \
  --config configs/default.yaml \
  --task "构建一个高质量的数学问答 SFT 数据集" \
  --log-level INFO 2>&1 | tee workspace/run_$(date +%m%d_%H%M).log
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
如果同时配置了 `trainer.eval_python`，脚本会在 rank0 训练完成后通过环境变量 `SFT_EVAL_PYTHON`
重新拉起同一个脚本的 `--model-dir` 纯评测分支，并在 `df-eval` 环境内执行 vLLM benchmark。

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

当前已配置的 Phase-2 真实链路：**Full SFT + 完整数学/科学 benchmark suite 评测**。

### 安装 SFT / vLLM 依赖

```bash
# SFT 训练环境
conda activate df-train
uv pip install -e ".[df-train]" \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  --index-strategy unsafe-best-match

# vLLM 评测环境
conda activate df-eval
uv pip install -e ".[df-eval]" \
  --index-url https://mirrors.aliyun.com/pypi/simple/ \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match
```

> **transformers 版本刻意错开**：`df-train` 用 transformers 5.x（`pyproject.toml` 钉 `>=5.6,<5.7`），
> `df-eval` 用 4.57.x（`<5`）。训练导出的 tokenizer 元数据由 `llamafactory_math_sft_eval.py` 的
> checkpoint 转换阶段自动做 5.x→4.x 兼容处理（移除 `extra_special_tokens`），所以两个环境版本不同是预期行为，勿强行统一。

> 推荐将 SFT 和 vLLM 评测拆成两次调用：
> 先在 `df-train` 中产出 `sft_model/`，再在 `df-eval` 中通过 `--model-dir` 加载该模型做 benchmark。
> 不再建议为了跑通整条链路，在同一个环境里混装 `llamafactory/deepspeed` 和 `vllm/xformers`。

如果只对一个已经训练好的模型做评测，在 `df-eval` 中直接运行：

```bash
python scripts/llamafactory_math_sft_eval.py \
  --model-dir <out_dir>/sft_model \
  --output <out_dir> \
  --mode dev
```

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SFT_BASE_MODEL` | 本地路径 `/vepfs-mlp2/c20250602/500050/models/Qwen2.5-32B-Instruct` | 基座模型（HuggingFace ID 或本地路径）；脚本内默认是该机器上的本地权重 |
| `SFT_NUM_EPOCHS` | `15` | 训练 epoch 数（LIMO 配置） |
| `SFT_LEARNING_RATE` | `5e-6` | 学习率（LIMO：极低、无 warmup） |
| `SFT_BATCH_SIZE` | `1` | per-device batch size |
| `SFT_GRAD_ACCUM` | `4` | 梯度累积步数 |
| `SFT_NUM_GPUS` | `16` | 训练**集群总卡数**（torchrun；双机每机 8 卡）；`NPROC_PER_NODE = SFT_NUM_GPUS // NNODES` |
| `EVAL_TP_SIZE` | `4` | vLLM 推理 tensor parallel size |
| `EVAL_DP_SIZE` | `2` | vLLM 推理 data parallel size（TP×DP = 总 GPU 数，默认 4×2=8） |
| `EVAL_CUDA_VISIBLE_DEVICES` | 未设置 | 评测使用的 GPU（如 `0,1,2,3,4,5,6,7`）；留空则自动使用前 TP×DP 个 GPU |
| `EVAL_MAX_MODEL_LEN` | `32768` | vLLM 最大序列长度（Qwen2.5-32B 上下文窗口） |
| `EVAL_MAX_NEW_TOKENS` | `8192` | 评测推理最大生成 token 数 |
| `MATH_TRAIN_SAMPLE_SIZE` | `500` | dev 评测从 MATH train（7500 条）随机抽样的题数（`seed=42`）；设 `0` 跑全量 7500 |
| `HF_CACHE_DIR` | `./hf_cache` | HuggingFace 缓存目录（当前脚本中定义但未实际使用） |
| `BENCHMARK_ROOT` | `data/benchmarks/limo_eval` | 仓库内固定 benchmark 根目录约定，不是运行时配置项 |

> **双机执行模型**：`run_sft` 是 torchrun 分布式作业，两机(rank0/rank1)都参与；`convert_dataset` 与
> `run_eval` 只在 **rank0** 执行——rank1 训练完即退出，不重复评测、不竞态写 `results.json`。eval 的 vLLM
> 推理在 rank0 **单机 8 卡**跑（32B bf16 权重≈64GB，单机 640GB 显存放得下，训练才需双机 ZeRO-3）。

> `BENCHMARK_ROOT` 这里是文档里的目录约定名，不是代码会读取的环境变量或 YAML 配置项。
> test 模式 benchmark 现在只读取项目内固定相对路径，不再通过环境变量或 HF fallback 动态找数据。
> 目录命名固定为：`aime`、`amc`、`math`、`gsm8k`、`cn_math_2024`、`gaokao`、`kaoyan`、`grade_school_math`、`minerva`、`olympiadbench`、`gpqa`。
> 如果你要替换 benchmark，只需要覆盖 `data/benchmarks/limo_eval/<bench>/test.jsonl`。

### 运行方式

脚本遵循 Phase-2 契约，由框架自动调用（`trainer.dev_eval_cmd`）：

```bash
# dev 模式（MCTS 迭代，框架自动调用）：MATH train 默认随机抽样 500 题（MATH_TRAIN_SAMPLE_SIZE=0 跑全量 7500）
python scripts/llamafactory_math_sft_eval.py --mode dev --dataset <ds.jsonl> --output <out_dir>

# test 模式（终评，手动调用一次）：跑完整 benchmark suite
python scripts/llamafactory_math_sft_eval.py --mode test --dataset <ds.jsonl> --output <out_dir>
```

内部流程：
1. **格式转换**：DataFlow JSONL → LlamaFactory ShareGPT 格式。字段按优先级取：instruction ← `instruction`/`question`/`input`；output ← `generated_cot`/`output`/`answer`（本项目当前 schema 用的是 `instruction`+`output`）
2. **Full SFT**：Qwen2.5-32B + DeepSpeed ZeRO-3（双机 16 GPU），全量微调
3. **评测**：vLLM 加载 fine-tuned 模型推理+判分（Qwen2.5-Math prompt 格式）
   - dev：MATH train（7500 条公开竞赛训练题，非 benchmark test set；默认随机抽样 500 条作奖励信号，`MATH_TRAIN_SAMPLE_SIZE=0` 跑全量）
   - test：GSM8K + AIME2024 + MATH500 + AMC23 + OlympiadBench + CHMath + Gaokao + Kaoyan + GradeSchool + MinervaMath + GPQA
4. **输出**：bench 平均 accuracy 作为 `score` 写入 `results.json`

### 双机训练部署（`dispatch: ssh`）

LIMO 配置的 Qwen2.5-32B Full SFT 需**双机 16 卡**。但 MCTS 编排器是 node0 上的单进程，
无法直接驱动双机 torchrun（node1 上没有进程会合，会卡在 rendezvous 直到超时）。

解决方式：**免密 SSH 拉起**。当前职责边界是：
- 编排器只负责把 pipeline 产出的数据路径交给训练脚本；
- 训练脚本在 node0 内部先做 `convert_dataset`；
- 然后由训练脚本自己经免密 SSH 到 node1 拉起 worker，再本地进入 rank0 训练；
- 训练完成后只在 rank0 单机做 dev/test 评测。

要求两机之间有**共享文件系统**（NFS/Lustre/GPFS 等，dataset/output 路径两机一致）+
**免密 SSH**。

**1. 打开 ssh 调度**（`configs/default.yaml`）：

```yaml
trainer:
  dispatch: ssh                        # 默认 local（单机）；改为 ssh 启用双机
  ssh:
    node1_host: user@10.0.0.2          # node1 的 SSH 目标（或 ~/.ssh/config 别名）
    master_addr: 10.0.0.1              # node0 对 node1 可路由的 IP（rendezvous 主地址）
    master_port: 29500
    nnodes: 2
    gpus_per_node: 8                   # 每机 GPU 数
    remote_cwd: /shared/DataFlow-Evolver  # node1 上先 cd 到这个项目目录，再执行 scripts/llamafactory_math_sft_eval.py；留空=编排器 cwd
    ssh_opts: ""                       # 额外 ssh 选项，如 "-i ~/.ssh/id_rsa -p 22"
```

也可用环境变量注入：`NODE1_HOST` / `MASTER_ADDR` / `REMOTE_CWD`。
其中 `REMOTE_CWD` 只是远端 worker 的工作目录；如果 node0/node1 项目绝对路径完全一致，通常可以直接留空，默认复用 node0 启动时的 cwd。

**2. 在 node0 起编排器**（照常，`python -m dataflow_evolver.main ...`）。node1 上**无需**起任何常驻进程
——训练脚本会在每轮训练时自动 SSH 到 node1 拉起 worker。

编排器只负责给 node0 训练脚本注入 `NODE_RANK=0`、`MASTER_ADDR` / `MASTER_PORT` /
`NNODES` / `SFT_NUM_GPUS` / `NODE1_HOST` / `REMOTE_CWD` / `SSH_OPTS` 等环境变量。当前脚本会：
- 只在 rank0 / node0 做数据格式转换；
- convert 完成后由 node0 脚本内部先 SSH 拉起 node1 worker，再本地进入 rank0 训练；
- 训练日志流式落盘到 `<output>/sft_train_rank0.log` 与 `<output>/sft_train_rank1.log`，node1 外层 SSH 日志落 `<output>/train_node1.log`。

要点与容错：
- 双机需 100G+ RoCE/InfiniBand 互连，否则 ZeRO-3 通信成瓶颈（脚本 docstring 已注明）。
- rank0 超时或结束时会一并终止 node1 侧 SSH 进程树，避免僵尸 GPU 进程。
- SSH 拉起失败 / rank0 非零退出 / 缺 `results.json` 均降级为 `TrainResult(failed=True)`，不阻塞主循环。
- `dispatch: local`（默认）时行为与旧版完全一致，单机直接拉起训练子进程；`nnodes: 1` 时 ssh 模式也会跳过远端拉起，退化为单机。

### timeout 说明

`configs/default.yaml` 的 trainer 块有两个独立超时：

- `timeout_sec: 3600`（1h）——**流水线执行**（纯数据合成）超时，由执行器使用；数据合成通常几分钟完成。
- `train_timeout_sec: 28800`（8h）——**dev/test 评测**（含内层 Full SFT 训练 + benchmark suite 推理）超时。

内层 `llamafactory_math_sft_eval.py` 的 SFT 训练子进程有 6h 超时（脚本内 `torchrun`），故外层 8h > 内层 6h + 评测，训练能完整跑完。每轮 MCTS 迭代调用一次 `dev_eval_cmd`；超过 `train_timeout_sec` 则强制终止进程树并标记该节点 `failed`，不阻塞主循环。

### 评测 Benchmark 详情

| Benchmark | 题数 | 难度 | 答案格式 | 判分方式 |
|-----------|------|------|----------|----------|
| GSM8K | 1319 | 小学-初中 | `#### <number>` | 数值精确匹配 |
| MATH-500 | 500 | 高中竞赛 | `\boxed{...}` | LaTeX 语义等价（math_verify） |
| AIME 2024 | 30 | 竞赛级 | 整数 0-999 | 整数精确匹配 |
| AMC23 | 取决于配置数据源 | 竞赛级 | 整数或字母 | 数值/字母精确匹配 |
| OlympiadBench | 取决于配置数据源 | 竞赛级 | `\boxed{...}` / LaTeX | LaTeX 语义等价 |
| CHMath | 取决于配置数据源 | 中文数学 | `\boxed{...}` / LaTeX | LaTeX 语义等价 |
| Gaokao | 取决于配置数据源 | 高中 | `\boxed{...}` / LaTeX | LaTeX 语义等价 |
| Kaoyan | 取决于配置数据源 | 大学/考研 | `\boxed{...}` / LaTeX | LaTeX 语义等价 |
| GradeSchool | 取决于配置数据源 | 小学数学 | `#### <number>` / 数值 | 数值精确匹配 |
| MinervaMath | 取决于配置数据源 | 高难数学 | `\boxed{...}` / LaTeX | LaTeX 语义等价 |
| GPQA | 取决于配置数据源 | 科学问答 | 选项字母 / 答案文本 | 多选题精确匹配 |

最终 `score = mean(all_configured_benchmark_scores)`；test 模式要求整套 benchmark 都成功加载，缺任何一个就直接失败，不再静默跳过。

---

## 暖启动 / 强基线锚点（可选，默认关闭）

`baseline.enabled=false`（默认）= **冷启动**：虚拟空根 + 绝对分奖励。置 `true` = **暖启动**：
先用一条固定的强基线流水线跑一次得 `baseline_dev_score`，以它作 MCTS 根节点，并切换到
**Relative Gain 奖励**（`reward = node_dev − baseline_dev`）——即"必须打败强基线才算正向"。

- 基线由 `baseline.baseline_pipeline` 指定，**必须是"从零合成"型**（首个算子为生成类、不依赖输入语料）
  且产在**目标领域**：框架会把它的 `first_entry` 强制替换为空 `empty.jsonl`，并注入代理 API / key / 缓存路径
  （`core/baseline.py::adapt_baseline_source`，只改运行时常量、不动结构）。
  自带示例：`.../api_pipelines/reasoning_math_baseline_pipeline.py`（CondorGenerator 数学题 → CoT）。
- ⚠️ 该示例文件位于 `workspace/_agent_root/dataflow/` 下，**随 `.gitignore` 不入库**（属上游 DataFlow 源码副本的一部分）；
  换机器时需与 dataflow 副本一起手动放置。
- 开启会**额外花一次真实 API/训练预算**跑基线，故默认关闭；启用前留意配额。

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

## 数据池（可选副产品）

`configs/default.yaml` 的 `data_pool` 段控制跨节点高质量数据的累积池。定位见
[`docs/design-decisions.md`](./docs/design-decisions.md)：

- **交付主线是「最优 pipeline」**；池是可选成品，**与 MCTS 奖励解耦**（奖励保持 per-node）。
- 池价值 = 可直接用的成品 + 回收已花算力 + 承载诊断驱动的累积覆盖（单节点结构上做不到）。
- **接入状态**：`core/data_pool.py` 的接口/去重/上限**已接入 `main.py`**（启动时按 `data_pool.enabled` 实例化 `DataPool`）。
  当前 `default.yaml` 的 `enabled` 默认值为 **`true`**（随仓库配置）；若做独立性消融或省算力，可手动置 `false`。
- **落盘暂存**：运行期每个通过节点的数据先写到 `workspace/pool_staging/` 的 sidecar jsonl，内存与 checkpoint 只留元数据；
  运行结束 `dump()` 时才流式读回统一[门槛筛 → 去重 → 占比裁剪]——避免整份数据集常驻内存或反复写进 checkpoint。
- ⚠️ 「池 vs 单 pipeline 谁更强」仍是**待受控实验检验的假设**（须等量对比 + 多样性量化），**非既定结论**。

---

## 测试

```bash
# 单元测试只需 agent 环境
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
dataflow_evolver/
├── agents/      # base / task_profiler / pipeline_agent(SDK 驱动) / review / diagnostic
├── core/        # schemas / tree_node / tree_manager / global_memory / data_pool / baseline / checkpoint
├── engine/      # execution_wrapper / trainer_wrapper
├── llm/         # serving(API/vLLM) / agent_sdk / mcp_operators / prompts
├── utils/       # config / logging / parsing
└── main.py      # 双反馈事件环编排器
configs/         # default.yaml
scripts/         # reference_train_eval.py（契约示例）/ llamafactory_math_sft_eval.py（数学 SFT+评测）/ vllm_eval_standalone.py
workspace/       # 运行产物 + _agent_root（Agent SDK cwd；dataflow 源码副本随 .gitignore 不入库）
tests/
```
