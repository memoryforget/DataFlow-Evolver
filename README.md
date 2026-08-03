# DataFlow-Evolver

多智能体、MCTS 驱动的自动数据工程框架（双反馈环）。

根据高层任务描述，自动生成 / 变异 DataFlow 流水线 DAG，并通过 **双反馈环**——
快速的"左移"数据评审（Phase-1）+ 真实的下游模型训练评估（Phase-2，产出可执行诊断报告）——
不断自我修正，最终产出高质量训练数据集。

> 设计蓝图与里程碑见 [`TODO.md`](./TODO.md)；关键取舍的来龙去脉见 [`docs/design-decisions.md`](./docs/design-decisions.md)。
> 想直接把两条链路（数学 SFT / PostTrainBench）跑起来，见 [`docs/run-flows.md`](./docs/run-flows.md)。

---

## 架构一览

```
input/ ──> InputCorpus ──┐
                         ├──> TaskProfiler ──> TaskProfile
任务描述 ────────────────┘                          │
                                                    ▼
                              Init node（node_0 = 树根，跑出第一个分数）
                                                    │
        ┌───────────────────────────────────────────▼──────────────────┐
        │  select(UCB1) → PipelineAgent(Agent SDK 写流水线) → 执行       │
        │     → Phase-1 ReviewAgent 快评审(含去污染) → Phase-2 训练+诊断 │
        │     → backpropagate(得分) → GlobalMemory 更新                  │
        └───────────────────────────────────────────────────────────────┘
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

# df-agent 不做 GPU 计算 —— CPU-only 轮子即可（~200MB）。
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e ".[agent-runtime]"

uv pip install -e ".[dev]"       # pytest（可选）

# 外部数据工具（仅当 agent_sdk.external_data: true）：HF + ModelScope + 发现渠道
uv pip install -e ".[external-data]"
```

> `external-data` 里刻意**不写** `modelscope[datasets]`：那个 extra 把 `datasets` 钉在 `<=4.8.4`，
> 会把 HF 侧和 vendored DataFlow 用的 5.x 降级掉。组里单列的 `addict` / `simplejson` /
> `sortedcontainers` 就是 `MsDataset` 流式加载实测所需的最小补集。

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
export ANTHROPIC_API_KEY=<你的 key>
export DF_API_KEY=<你的 key>
export ANTHROPIC_BASE_URL=<自建/代理网关，用官方端点则不设>
export HF_HOME=<HuggingFace 缓存目录>

python -m dataflow_evolver.main \
  --config configs/default.yaml \
  --task "构建一个高质量的数学问答 SFT 数据集" \
  --log-level INFO 2>&1 | tee workspace/run_$(date +%m%d_%H%M).log
```

命令行覆盖（供消融实验在同一份 YAML 上换档）：

| 参数 | 覆盖的配置项 | 用途 |
|---|---|---|
| `--input-dir` | `project.input_dir` | 换输入语料目录；有语料=grounded 合成，空=从零合成 |
| `--init-code` | `init.reference_pipeline` | Init node 走「忠实改写一份已有实现」模式 |
| `--init-instruction` | `init.instruction` | Init node 按一句自然语言指令构建 |

输出：最优流水线所在节点目录、`field_flow` 与平均奖励。

---

## 外部训练脚本契约（Phase-2）

`DownstreamTrainerWrapper` 按如下契约子进程调用你的脚本：

```
<train_python> <eval_cmd> --dataset <ds.jsonl> --output <out_dir> [--dev-set <dev.jsonl>]
```

其中 `<train_python>` 由 `trainer.train_python` 配置（留空则用当前环境 python），
`<eval_cmd>` 对应 `trainer.dev_eval_cmd`（MCTS 迭代）或 `trainer.test_eval_cmd`（终评）。
配了 `trainer.dev_set.source` 时会追加 `--dev-set`。
如果同时配置了 `trainer.eval_python`，脚本会在训练完成后通过环境变量 `SFT_EVAL_PYTHON`
重新拉起同一个脚本的 `--model-dir` 纯评测分支，并在 `df-eval` 环境内执行 vLLM benchmark。

### 脚本负责什么、不负责什么

职责边界只有一条：**脚本执行，框架决策。** 凡是会影响「节点之间分数可比性」的选择都归框架，
因为那些选择一旦散进脚本，就没有任何机制能保证前后节点用的是同一套——不报错，日志里也看不出来。

脚本**该做**的：

| 事项 | 说明 |
|---|---|
| 读 `--dataset` 训练 | 拿到什么数据就训什么，不自行增删改采样 |
| 读 `--dev-set` 评测 | 给了哪份题集就评哪份，**整份评完** |
| `--mode test` 跑终评 | 固定的 benchmark 套件，与 dev 题集无关 |
| 写 `<out_dir>/results.json` | 见下面的字段约定 |
| 失败就非零退出 | 别把基础设施故障伪装成 `score=0.0`，那是两回事 |

脚本**不该做**的：

| 反模式 | 为什么 |
|---|---|
| 自己抽样 dev 题集 | 框架已按 `(source_sha, seed, size)` 物化并写进 manifest。脚本再抽一次，评的题数与 manifest 对不上，`dev_set.size` 形同虚设 |
| 缺 `--dev-set` 就回退默认题库 | 有退路 = 框架的口径只是建议。正确做法是**当场报错** |
| 自己截断训练数据 | 数据量直接决定梯度步数，切齐由 `CappedTrainer` 统一做（见 `generation.enforce_target_samples`） |
| 训练超参走脚本内环境变量随节点变 | 配方必须在整轮搜索里冻结，否则分差说不清是数据还是超参带来的 |
| 读取任何 benchmark test split 进训练路径 | 数据泄露 |

> 参照实现：`llamafactory_math_sft_eval.py --mode dev` 在**训练开始前**校验 `--dev-set`，
> 缺了直接非零退出。这条链路是先 SFT 6 小时再评测，等评测阶段才报错等于白烧一整轮 GPU。
> 早期版本这里留了「没给题集就用内置的 500 条」的退路，结果是把 `dev_set.size` 调到 500 以上
> 就被脚本静默抽回 500，而 manifest 记的是原值。退路本身就是 bug 的温床。

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
- 框架另在 `<out_dir>/run_manifest.json` 记录本次的完整命令、影响分数的环境变量
  （`SFT_*` / `EVAL_*` / `MATH_*` 等）与脚本内容摘要。训练配方散在框架之外，跑到一半被改动
  会让前后节点的分数不可比，而树里看不出任何差别——留一份指纹才能事后核验。

最小可跑示例见 [`scripts/reference_train_eval.py`](./scripts/reference_train_eval.py)：约 80 行，
不需要 GPU（"训练"就是把 `--dataset` 的问答对背进一个字典，再到 `--dev-set` 上逐题作答），
但形状与真实脚本一致——在合成数据上训练、在框架指定的那份题集上整份评完、产出真实
`exact_match` 与错题。上面两张表里的纪律它都遵守，包括拿不到题集就非零退出。

---

## 数学 SFT 下游链路（LlamaFactory + Math Benchmark）

当前已配置的 Phase-2 真实链路：**Full SFT + 完整数学/科学 benchmark suite 评测**。
依赖安装见上面的[安装](#安装)（`df-train` / `df-eval` 两个环境）。

> **transformers 版本刻意错开**：`df-train` 用 transformers 5.x（`pyproject.toml` 钉 `>=5.6,<5.7`），
> `df-eval` 用 4.57.x（`<5`）。训练导出的 tokenizer 元数据由 `llamafactory_math_sft_eval.py` 的
> checkpoint 转换阶段自动做 5.x→4.x 兼容处理（移除 `extra_special_tokens`），所以两个环境版本不同是预期行为，勿强行统一。

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
| `SFT_BASE_MODEL` | Qwen2.5-32B-Instruct 的本地路径 | 基座模型（HuggingFace ID 或本地路径）。**换基座是最该改的一项**，默认值是当初跑 LIMO 复现时留下的 |
| `SFT_NUM_EPOCHS` | `15` | 训练 epoch 数（LIMO 配置） |
| `SFT_LEARNING_RATE` | `5e-6` | 学习率（LIMO：极低、无 warmup） |
| `SFT_BATCH_SIZE` | `1` | per-device batch size |
| `SFT_GRAD_ACCUM` | `4` | 梯度累积步数 |
| `SFT_NUM_GPUS` | `8` | 本机参与训练的卡数（torchrun）。全局批次 = 卡数 × batch × grad_accum |
| `SFT_CUTOFF_LEN` | `16384` | 训练截断长度；超长样本在格式转换阶段被丢弃并告警 |
| `EVAL_TP_SIZE` | `4` | vLLM tensor parallel size |
| `EVAL_DP_SIZE` | `2` | vLLM data parallel size。TP×DP 须整除可用卡数；单卡放得下就设 TP=1、把卡都给 DP |
| `EVAL_CUDA_VISIBLE_DEVICES` | 未设置 | 评测使用的 GPU；留空则自动用前 TP×DP 个 |
| `EVAL_MAX_MODEL_LEN` | `32768` | vLLM 最大序列长度，需 ≤ 基座上下文窗口 |
| `EVAL_MAX_NEW_TOKENS` | `8192` | 评测推理最大生成 token 数 |

> **换基座时要一起改的**：`SFT_BASE_MODEL` 之外还有 `SFT_CUTOFF_LEN` / `EVAL_MAX_MODEL_LEN`
> （不能超过新基座的上下文窗口）与 `EVAL_TP_SIZE`（按权重大小定，小模型 TP=1 更快）。
> 学习率与 epoch 数是 LIMO 针对 32B + 800 条小数据调的，换小模型通常要重调。

> dev 评多少题**不在这张表里**：它由 `trainer.dev_set.size` 决定，框架物化后经 `--dev-set` 下传。
> 脚本侧没有对应的环境变量，也没有内置默认题库（见上面的[脚本负责什么](#脚本负责什么不负责什么)）。

> **benchmark 只按固定相对路径读**（`data/benchmarks/limo_eval/<bench>/test.jsonl`），
> 没有环境变量开关，也不做 HF fallback：缺文件直接抛错，不静默换数据源。目录名固定为
> `aime`、`amc`、`math`、`gsm8k`、`cn_math_2024`、`gaokao`、`kaoyan`、`grade_school_math`、
> `minerva`、`olympiadbench`、`gpqa`。要换题集就覆盖对应的 `test.jsonl`。

### 运行方式

脚本遵循 Phase-2 契约，由框架自动调用（`trainer.dev_eval_cmd`）：

```bash
# dev 模式（MCTS 迭代，框架自动调用）：--dev-set 由框架物化后传入，缺了直接报错
python scripts/llamafactory_math_sft_eval.py --mode dev \
  --dataset <ds.jsonl> --output <out_dir> --dev-set workspace/eval/dev.jsonl

# test 模式（终评，手动调用一次）：跑完整 benchmark suite，不需要 --dev-set
python scripts/llamafactory_math_sft_eval.py --mode test --dataset <ds.jsonl> --output <out_dir>
```

内部流程：
1. **格式转换**：DataFlow JSONL → LlamaFactory ShareGPT 格式。字段按优先级取：instruction ← `instruction`/`question`/`input`；output ← `generated_cot`/`output`/`answer`（本项目当前 schema 用的是 `instruction`+`output`）
2. **Full SFT**：DeepSpeed ZeRO-3 + FlashAttention-2，单机 `SFT_NUM_GPUS` 卡全量微调
3. **评测**：vLLM 加载 fine-tuned 模型推理+判分（Qwen2.5-Math prompt 格式）
   - dev：`--dev-set` 指向的那份题集，整份评完。默认由 `data/math_train.jsonl`（MATH competition train，7500 条公开训练语料，非 benchmark test set）抽 500 条物化而来
   - test：GSM8K + AIME2024 + MATH500 + AMC23 + OlympiadBench + CHMath + Gaokao + Kaoyan + GradeSchool + MinervaMath + GPQA
4. **输出**：bench 平均 accuracy 作为 `score` 写入 `results.json`

### 训练调度：单机

编排器在本机 `subprocess` 拉起训练脚本、阻塞等待、解析 `results.json`，没有跨机调度。
`SFT_NUM_GPUS`（默认 8）就是本机参与训练的卡数，torchrun 由 LlamaFactory 内部拉起。
训练日志流式落盘到 `<output>/sft_train.log`。

> 早期版本支持双机 16 卡（`dispatch: ssh` + 免密 SSH 拉起 node1 worker），随 32B 训练一起
> 移除了。要重新支持多机，需要重新引入的是：`SSHTrainerWrapper`（注入 `NNODES`/`NODE_RANK`/
> `MASTER_ADDR`）、脚本侧的远端拉起与 rank 分支，以及跨机 NCCL 的 RDMA 网卡推断。

### timeout 说明

`configs/default.yaml` 的 trainer 块有两个独立超时：

- `timeout_sec: 7200`（2h）——**流水线执行**（纯数据合成）超时，由执行器使用；4 算子约 30min，5 算子（交叉验证+精修）会超 1h。
- `train_timeout_sec: 28800`（8h）——**dev/test 评测**（含内层 Full SFT 训练 + benchmark suite 推理）超时。

内层 `llamafactory_math_sft_eval.py` 的 SFT 训练子进程有 6h 超时（脚本内 `torchrun`），故外层 8h > 内层 6h + 评测，训练能完整跑完。每轮 MCTS 迭代调用一次 `dev_eval_cmd`；超过 `train_timeout_sec` 则强制终止进程树并标记该节点 `failed`，不阻塞主循环。

### 评测 Benchmark 详情

题数为仓库内 `data/benchmarks/limo_eval/<bench>/test.jsonl` 的实际条数：

| Benchmark | 题数 | 难度 | 答案格式 |
|-----------|------|------|----------|
| GSM8K | 1319 | 小学-初中 | `#### <number>` |
| MATH-500 | 500 | 高中竞赛 | `\boxed{...}` |
| AIME 2024 | 30 | 竞赛级 | 整数 0-999 |
| AMC23 | 40 | 竞赛级 | 整数或字母 |
| OlympiadBench | 675 | 竞赛级 | `\boxed{...}` / LaTeX |
| CHMath | 30 | 中文数学 | `\boxed{...}` / LaTeX |
| Gaokao | 79 | 高中 | `\boxed{...}` / LaTeX |
| Kaoyan | 199 | 大学/考研 | `\boxed{...}` / LaTeX |
| GradeSchool | 210 | 小学数学 | `#### <number>` / 数值 |
| MinervaMath | 272 | 高难数学 | `\boxed{...}` / LaTeX |
| GPQA | 198 | 科学问答（多选） | 选项字母 |

**判分是统一的**，不按 benchmark 各写一套（`vllm_eval_standalone._grade`）：带 `gold_letter` 的
多选题比选项字母；其余抽出答案后交 `math_benchmarks/grader.py` 做 LaTeX/数值语义等价判定
（LIMO 口径，依赖 `latex2sympy2`，只装在 `df-eval`）。

> 多选题必须比字母：`_normalize_row` 给 MCQ 存的 `gold_answer` 是选项**原文**，拿抽出的字母去
> 比它永远不等。这条曾让 GPQA 在终评里稳定接近 0 分，与模型好坏无关。

按 unbiased pass@1 累计，最终 `score = mean(各 benchmark 分)`；test 模式要求整套 benchmark
都成功加载，缺任何一个直接失败，不静默跳过。

---

## 另一条链路：PostTrainBench

`configs/posttrainbench_gsm8k.yaml`。与上面这条数学 SFT 链路的关键区别是**奖励口径**：
它用 `reward_source: review`，搜索期只跑 200 条探针 + ReviewAgent 代理分（不占 GPU），
到终选阶段才对 top-K 放大重产并跑真实 SFT + 官方 `evaluate.py`。

两条链路的完整跑法（前置环境、启动命令、各阶段产物、常见故障）见
[`docs/run-flows.md`](./docs/run-flows.md)；PTB 特有的适配细节与已知的 dev ⊂ test 重叠见
[`docs/posttrainbench_adaptation.md`](./docs/posttrainbench_adaptation.md)。

---

## Init node：搜索的起点（node_0 = 树根）

树根不是虚拟空节点，而是一个持有真实流水线的普通实验节点。它跑出第一个分数，之后所有节点都是
它的变异后代；"相对起点提升了多少"就是 `score − node_0_score`，不需要任何奖励平移或锚点模块。
**起点跑不出分会直接终止本轮**——没有可变异的基础，继续搜索只是空转烧预算。

三档启动模式（`configs/default.yaml` 的 `init` 段，可被命令行覆盖）：

```yaml
init:
  reference_pipeline: ""   # 给路径 = 忠实改写：把一份已有实现搬到本框架契约下，不改数据处理逻辑
  instruction: ""          # 给一句话 = 按指令构建
                           # 两项都空 = Agent 自主设计一条最小可行流水线
```

```bash
python -m dataflow_evolver.main --task "..." --init-code path/to/reference_pipeline.py
python -m dataflow_evolver.main --task "..." --init-instruction "先合成题目再蒸馏 CoT，最后按答案一致性过滤"
```

> 消融要在同一份 YAML 上换档，所以两项都提供了命令行覆盖——改配置文件既容易漏改，
> 也让"这次跑的是哪一档"不可追溯。

---

## 输入数据：`input/` 决定合成范式

把语料放进 `input/`（或 `project.input_dir` 指定的目录）即可：

| `input/` 状态 | 范式 | 流水线入口 |
|---|---|---|
| 有数据文件（jsonl/json/csv/parquet） | **corpus-grounded 合成**：在语料之上加工（抽取/改写/增强/蒸馏/过滤） | 该语料文件 |
| 空 / 不存在 | **from-scratch 合成**：凭模型知识从零生成 | 0 字节 `empty.jsonl` |

范式由目录内容决定，**没有额外开关**——配了 grounded 却没语料、或放了语料却按 scratch 跑，
都只可能是误配。入口文件的行数、字段与前几行原文会自动注入生成 prompt，Agent 不必先写个探针
流水线去看数据长什么样。目录里有多个数据文件时按「格式优先级 + 字典序」确定性选择并告警，
可用 `project.input_entry` 显式指定。

> **这张表说的是本地输入,不是数据的全部来源。** 打开[外部数据搜索](#外部数据搜索可选)后，
> 上面两行都会变：`input/` 为空时 Agent 可以去 Hub 取一份真实公开数据集、`download_dataset`
> 落盘后把入口指向它,再用算子规整成目标 schema——「没有本地语料」不等于「只能凭模型知识编」;
> `input/` 有语料时外部数据只作**补充**,不能把用户给的语料顶掉。这两种口径由生成 prompt 的
> 范式规则按 `(input/ 有无语料) × (取数工具开没开)` 四种组合分别渲染
> （`llm/prompts.py::_paradigm_rule`）。

`input/` 内容不入库（见 `.gitignore`），仓库只保留目录结构。

### 产出 schema：`trainer.dataset_schema`

`input/` 的字段是任意的，下游训练脚本要的是另一套，中间由算子做映射。**「另一套」由配置说了算**：

```yaml
trainer:
  dataset_schema:
    instruction: str      # 题面 / 指令
    output: str           # 参考解答（含推理过程）
```

它会覆盖 TaskProfiler 的推断，并流到三处消费点：生成 prompt 的目标 schema、评审 prompt、
硬指标的关键字段。同时作为**训练前的硬门槛**——关键字段空缺率 ≥ `review.null_threshold`
（默认 0.5）直接判不通过，并把要求的字段名写进 issues 供自修正循环使用。

留空则退回「让 LLM 自行设计字段名」（会告警）。不建议：Profiler 给数学任务编出
`{problem, solution}` 完全合理，但 `llamafactory_math_sft_eval.py` 一条都读不到，
而这要到训练阶段才暴露，报错还只显示「训练脚本非零退出」。取值须与训练脚本的取字段逻辑一致，
两份脚本的别名表并不相同（`solution` 只有 PostTrainBench 那份认）。
背景见 [`docs/design-decisions.md`](./docs/design-decisions.md) §3.1.1。

### 评测数据：`data/`

两类，入库与否刚好相反（详见 [`data/README.md`](./data/README.md)）：

- **`data/benchmarks/limo_eval/`（入库）**：终评用的 11 份 test set。它是量具不是语料——
  跑分要跨节点、跨轮次可比，就得保证每次评在逐字节相同的题上，所以随仓库版本化。
- **`data/math_train.jsonl`（不入库）**：`trainer.dev_set.source`，即 MCTS 的考卷来源。
  是可再生的公开训练语料，自行获取。

注意 dev 题集**不是训练语料**——训练数据由每个节点现合成，框架没有可切的 train split，
所以固定下来的是一份评测题集，而非 train/val 划分。

---

## 外部数据搜索（可选）

`configs/default.yaml` 里默认**关闭**（`external_data: false`），PTB 配置里默认**开启**。

```yaml
agent_sdk:
  external_data: true             # 总开关：挂载 dataset-search + data-discovery
  discovery: true                 # 是否连带挂 data-discovery（只找线索的那台）
project:
  data_links_dir: ""              # 下载目录，留空 = <workspace>/data_links

dataset_search:                   # 取数：连接与下载
  endpoint: ""                    # HF_ENDPOINT；直连不通时填镜像，如 https://hf-mirror.com
  token: ""                       # HF_TOKEN；gated 数据集需要，留空=沿用环境/本地登录态
  cache_dir: ""                   # HF_HOME
  max_download_mb: 2048           # 单份数据集落盘上限
  modelscope_endpoint: ""         # 留空 = https://modelscope.cn

discovery:                        # 发现：凭据缺失只影响对应那一个工具
  serper_api_key: ""              # search_web 用（付费）
  github_token: ""                # 留空 = 匿名，GitHub 限流 10 次/分钟
```

**国内网络下的可用组合**（本仓库开发环境实测：`huggingface.co` 直连不通，其余均通）：

| 目标 | 可达性 | 配置 |
|---|---|---|
| HuggingFace | 直连 ✗ / 镜像 ✓ | `dataset_search.endpoint: https://hf-mirror.com` |
| ModelScope | ✓ | 默认即可 |
| GitHub 搜索 | ✓ | 匿名可用；配 token 放宽限流 |
| arXiv | ✓ | 无需配置（**必须 https**，明文 http 会被挡） |
| Serper | 需 key | 没有就把 `agent_sdk.discovery` 关掉，退回内置 WebSearch |

### 两台 MCP：取数与发现分开

**`dataset-search`（会落盘）** —— 两个 hub，契约完全一致，按 search → inspect → preview → download 顺序用：

| HuggingFace | ModelScope | 作用 |
|---|---|---|
| `search_datasets` | `search_ms_datasets` | 按关键词找候选 |
| `inspect_dataset` | —（元信息随搜索返回） | 看 license / configs / splits |
| `preview_dataset` | `preview_ms_dataset` | 取几行看字段与风格 |
| `download_dataset` | `download_ms_dataset` | 整个 split 落成 jsonl，返回绝对路径 |

HuggingFace 直连不通（国内常见）时改用 ModelScope 那一组，找中文语料时也优先它。ModelScope 侧检索
一次就带回 license / tags / tasks / downloads，所以没有单独的 inspect 步骤。两个 hub 的落盘文件名带
hub 短码前缀（`hf__` / `ms__`）：同名数据集在两处未必是同一份内容，不能共用缓存键。

依赖装 `pip install -e '.[external-data]'`；缺包时只让对应工具返回一条说明，不影响另一侧。

> **脚本型数据集不支持。** ModelScope 上仍有 legacy 的 script dataset（如 `modelscope/gsm8k`），
> 加载要求 `trust_remote_code=True`，等于在编排器进程里执行仓库里的任意代码。工具不开这个开关，
> 而是提示换一份 parquet/jsonl 存储的镜像仓库——同一份语料通常有多个 owner。

**`data-discovery`（只找线索，不落盘）** —— `search_web`（Serper）、`search_github`、`search_papers`
（arXiv，无需凭据）。Hub 的关键词搜索**不支持语义匹配**，"想要小学数学应用题的 CoT 语料"这种意图
往往得先在论文或博客里看到某个数据集被用过，才知道它叫什么。三个工具的返回都带一句"拿到 id 后回
`download_dataset` / `download_ms_dataset` 取数"——真正取数的动作必须经落盘工具留溯源。

相比 SDK 内置的 `WebSearch`/`WebFetch`，这三个的增量只是结构化结果与统一的下一步指引；用不上
Serper key 时把 `discovery` 关掉、退回内置工具也完全可行。

### 下载语义

- **默认下载整个 split**，不做"前 N 行"的默认截断。取多少、怎么采样、按什么过滤是**流水线**该做的
  决策，也正是被搜索的对象；固化在工具里等于替搜索先做了一个它无从推翻的选择。
- 只取单个 split，**不整仓库拉**——整仓库会把 `test` split 一起拖下来，那正是下一节要防的泄露入口。
- 流式边读边写，内存占用与数据集大小无关；落盘受 `max_download_mb` 保护，触顶截断并在 meta 里标
  `complete: false` + `truncated_by`。截断留下的是 split **头部**，不是随机样本，meta 与工具返回值
  都会点明这一点。
- 先写 `.part` 再原子改名：下载中途被打断不会留下半份文件被后续节点当成完整数据复用。
- 下载统一进共享 `data_links/`，跨节点按 `(hub, dataset_id, config, split)` 复用同一份文件；每个文件
  旁落一份 `.meta.json`（来源 / dataset_id / config / split / URL / license / 行数 / 字节 / 是否完整 / 时间）。
- 依赖 `huggingface_hub` / `datasets`（ModelScope 侧为 `modelscope`）；缺包或网络不通只会让工具返回
  一条说明，不影响主循环。

> ⚠️ 开启前请先配好去污染参考集（下一节）：Hub 上大量数据集本身就是各类 benchmark 的镜像或衍生。

### 每个节点的数据溯源

Agent 的内置工具是全放开的，它可以绕过上面这些工具直接 `load_dataset(...)` / `curl` 拉数据，
那条路 `data_links/` 里不留痕。所以每个节点目录还会落两份记录（**外部数据工具关着也照样落**）：

| 文件 | 内容 |
|---|---|
| `agent_tools.jsonl` | Agent 每次运行调过的工具与参数，按 `phase`（生成 / 第几轮自修正）标注，逐次追加 |
| `data_provenance.json` | 评分收尾静态扫描节点代码：引用的 `data_links/` 文件（附 `.meta.json`），以及绕过工具的 `load_dataset` / `hub_download` / `hf:`·`ms:` 远程 URI / 数据类 URL |

`data_provenance.json` 里 `unmanaged: true` 表示该节点存在无出处的外部数据引用，日志同时告警。

其中 `storage_uri` 那一类值得单独说：DataFlow 的 `FileStorage` 原生支持把 `first_entry_file_name`
写成 `hf:owner/name:config:split` 或 `ms:owner/name:split` 直接拉远程数据，而 `dataflow-dev` 技能的
knowledge base 正好教了这个写法。它比 `load_dataset` 更顺手（一个字符串就完事），却完全不长得像
"下载"——是最可能被走、也最容易漏掉的一条绕行路径，所以扫描器和 prompt 都专门覆盖了它。

静态扫描扫不到运行时拼出的 dataset id 与 URL，记录里的 `method: "static_scan"` 即在标这个限度。

---

## 防数据泄露：dev / test 纪律 + 去污染硬指标

MCTS 反复用同一 benchmark 当奖励会**对它过拟合**；若再用同一 benchmark 报告结果，分数虚高、不可泛化。
设计见 [`docs/design-decisions.md`](./docs/design-decisions.md) §4。

**① dev / test 隔离**

- **`dev`**（held-in）：**只**给 MCTS 当奖励信号，允许被「刷」——配 `trainer.dev_eval_cmd`。
- **`test`**（held-out）：**全程对 loop 不可见**，仅在选出最终模型后跑一次用于报告——配 `trainer.test_eval_cmd`。

**② dev 题集由框架物化，Agent 无权干预**

```yaml
trainer:
  dev_set:
    source: data/math_train.jsonl   # 留空 = 不物化；本仓库的评测脚本会因此拒绝跑 dev
    size: 500                       # 0 = 全量
    seed: 42
```

启动时按 `(source_sha, seed, size)` 固定成 `workspace/eval/dev.jsonl`，连同 `dev.manifest.json`
落盘，并以 `--dev-set <path>` 显式下传给评测脚本。三个理由：

- **可比性**。奖励是节点之间比出来的，前提是所有节点评在同一批题上。抽样若发生在每个节点的
  评测子进程里，任何人中途改一下条数或种子，前后节点就在不同题集上比——不报错、日志里也看不出来。
- **可去污染**。dev 是搜索直接优化的目标，把 dev 题混进合成数据比混 test 更能刷分（test 全程
  不参与 loop）。只存在于子进程内存里的题集没法当参考集，这个洞根本堵不上。物化后**自动**并入
  去污染参考集,不需要在 `reference_files` 里重复配。
- **一处切分、处处复用**。口径进 checkpoint，续跑时对不上直接报错；评测脚本退化成
  「在这个文件上评测」，不必各写一套抽样逻辑。

配套的纪律在脚本侧：本仓库的评测脚本**不接受"没给题集"这种情况**，`--mode dev` 缺 `--dev-set`
直接非零退出。只要留着"那就用我自己的默认题集"的退路，上面三条就都只是君子协定。

**③ 去污染（已接线）**

```yaml
review:
  decontamination:
    reference_files:                # test 侧参考；dev 题集物化后自动并入，不必在这里重复写
      - data/benchmarks/limo_eval/gsm8k/test.jsonl
      # ... default.yaml 里列的是终评那 11 份
    ngram: 13
    max_rate: 0.02
```

合成数据与基准测试集做归一化后的 13-gram 重合检测，得 `contamination_rate`：
按比例给 tier1 打折，超过 `max_rate` 直接判 `passed=False` 并把命中片段作为 issue 喂回 Agent。

> 为什么必须是硬指标：搜索的目标函数就是下游分数，把测试题混进训练集是提分最省力的路径。
> 只要泄露不被扣分，MCTS 迟早会找到它——这不是 Agent 不听话，是奖励函数在鼓励它。

**参考集必须与你实际跑的 benchmark 对应。** 这一条在 PostTrainBench 链路上尤其容易踩：
`configs/posttrainbench_gsm8k.yaml` 的参考集经 `${PTB_DECONTAM_REF:...}` 默认指向仓库内的
GSM8K test split，而 benchmark 是由 `PTB_BENCHMARK` 选的。**换 benchmark 时两者必须一起换**：

```bash
export PTB_BENCHMARK=humaneval
export PTB_DECONTAM_REF=/path/to/humaneval_test.jsonl   # 忘了这行 = 拿 gsm8k 的题查 humaneval
```

漏掉的话不会报错——去污染照常"启用"、照常报 0% 污染，只是它在查一份毫不相干的题集，
给你的是虚假的安全感。除 GSM8K 外的 PTB benchmark（aime2025 / gpqamain / humaneval / bfcl /
healthbench / arenahardwriting）本仓库没有本地副本，需自行落一份**仅供比对**的 test 文件
（放在训练路径之外）再指过来。

> Phase-1 评审有五个维度（正确性/相关性/难度/多样性/schema）加硬指标扫描，但它仍**不能**
> 单独替代 Phase-2 的真实反馈：代理分再准也只是代理。正解是保留 Phase-2，并把评测放在 dev 上、
> 与 test 严格隔离。

---

## 奖励口径：`search.reward_source`

| | `train`（默认） | `review` |
|---|---|---|
| 搜索期奖励 | 真实 SFT + dev 评测的真分 | ReviewAgent Tier-1 复合分（代理分） |
| 搜索期规模 | `generation.target_samples` | `generation.probe_samples`（探针，便宜、不占 GPU） |
| 收尾 | 不需要 | 对 top-K 放大重产 + 真实 SFT + 评测，把代理分兑现成真分 |

口径带来的连带纪律（执行规模、是否训练、是否需要收尾兑现、时间预算预留）都由
`core/evaluation.py` 的策略对象一处给出。**拼错的口径名会在启动时直接报错**，
而不是静默回退成默认值——否则一次跑几小时的实验会安静地跑成另一种实验。

### 训练集切齐：`generation.enforce_target_samples`（默认 `true`）

训练配方（`epochs`/`lr`/`batch`）在框架外冻结，总优化步数因此是 `epochs × N / global_batch`——
**数据条数直接决定算力预算**。不切齐的话，多产三倍数据的节点白拿三倍梯度步数，分高了说不清是
数据更好还是单纯训得更久，而 MCTS 会如实地学走"多产数据"这条与数据质量无关的捷径。

`target_samples` 对 pipeline 只是经 `DF_TARGET_SAMPLES` 注入的**建议值**：算子可以不读它，
过滤会少产、增强会多产。所以框架在数据进训练脚本前切齐到 `target_samples` 条（确定性抽样、
保序），并在训练输出目录落一份 `train_set.manifest.json`：

| 字段 | 含义 |
|---|---|
| `produced` / `used` | pipeline 实际产出 / 训练实际使用 |
| `truncated` | 是否发生截断 |
| `short` | 产出**不足** `cap`——反方向的同一个问题（欠训），切不出数据来，只记录 + 告警 |

搜索期打分与终选 Tier-2 是两条独立的训练路径，切齐由 `CappedTrainer` 包在 trainer 外面统一
保证，不靠调用点各自记得。做"更多数据是否更好"的消融时把它设成 `false`，显式把数据量放回
搜索空间。设计背景见 [`docs/design-decisions.md`](./docs/design-decisions.md) §2.5.1。

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
├── core/        # schemas / tree_node / tree_manager / global_memory / checkpoint
│               # input_corpus(input/ 契约) / evaluation(评分策略) / decontamination(去污染)
│               # dev_set(dev 题集物化) / provenance(外部数据溯源)
├── engine/      # execution_wrapper / trainer_wrapper
├── llm/         # serving(API/vLLM) / agent_sdk / prompts
│               # mcp_operators(算子源码参考) / mcp_datasets(取数) / mcp_discovery(找线索)
├── utils/       # config / logging / parsing
└── main.py      # 双反馈事件环编排器
configs/         # default.yaml（数学 SFT 链路）/ posttrainbench_gsm8k.yaml（PTB 链路）
input/           # 输入语料目录（放了数据=grounded 合成；内容不入库）
data/            # benchmarks/（终评题集，入库）+ dev 题集来源（不入库，见 data/README.md）
scripts/         # reference_train_eval.py（契约示例）
                 # llamafactory_math_sft_eval.py + vllm_eval_standalone.py（数学 SFT 链路）
                 # posttrainbench_sft_eval.py + ptb_benchmarks.py（PTB 链路）
workspace/       # 运行产物 + _agent_root（Agent SDK cwd）+ data_links（外部数据集下载）
                 # + eval/dev.jsonl（物化的 dev 题集与 manifest）
docs/            # design-decisions（取舍来龙去脉）/ run-flows（两条链路怎么跑）
                 # posttrainbench_adaptation（PTB 适配细节）
tests/
```
