# DataFlow Evolver

[English](README.md) | [简体中文](README_zh.md)

DataFlow Evolver 在用户提供的固定原始数据集上持续改进数据处理 Pipeline。每轮迭代都遵循一个透明的反馈闭环：

```text
当前最优 Pipeline -> 挑战者数据集 -> Review Agent -> 接受/拒绝
        ^                                      |
        +--------- 精简反馈 + 决策历史 ---------+
```

每个挑战者都从 best-so-far 的当前最优方案出发。被拒绝的 Pipeline 会以精简形式保留在进化历史中，帮助后续迭代避免重复相同策略，但其代码不会被继承。达到配置的迭代次数后，可以由人工使用选中的最优数据集进行训练和评测。

## 仓库结构

```text
DataFlow-Evolver/
├── dataflow_evolver/                  # 进化循环、Agent、Review 与运行时
├── configs/
│   └── default.yaml                   # 主配置文件
├── data/                              # 本地数据集（Git 忽略）
├── model/                             # 本地 embedding 模型（Git 忽略）
├── docs/                              # 设计与工作流文档
├── tests/                             # 单元测试与集成测试
└── workspace/                         # Agent 工作区与实验制品
    ├── dataflow/                      # 纳入版本管理的 DataFlow 源码副本
    ├── .claude/skills/dataflow-dev/   # Claude 原生 DataFlow Skill
    ├── .agents/skills/dataflow-dev/   # Codex 原生 DataFlow Skill
    ├── .opencode/skills/dataflow-dev/ # OpenCode 原生 DataFlow Skill
    ├── cache/das/                     # 共享 embedding 缓存
    └── runs/<run_name>/               # 相互隔离的运行目录
```

每次运行均为自包含结构：

| 运行制品 | 用途 |
| --- | --- |
| `manifest.json` | 记录选中的 incumbent、最优数据集和完整运行摘要 |
| `iterations.jsonl` | 按时间排列的精简进化历史 |
| `iteration_NNN/pipeline.py` | 单轮迭代提出的 Pipeline |
| `iteration_NNN/operators/` | 自动生成的自定义算子 |
| `iteration_NNN/review.jsonl` | Review Agent 的结构化评审证据 |
| `iteration_NNN/sessions/<backend>/` | 原生 Agent 状态和逐轮 token 用量 |
| `downstream_eval/checkpoint_NNN/` | 可选的外部 SFT/评测请求、结果与日志 |

框架会以 `workspace/` 为工作目录启动 Claude、OpenCode 或 Codex。每个后端都会从其原生发现路径获得一份完整的 DataFlow Skill；三份 Skill 有意保持相同指导内容，仅适配各后端的根路径。

## 安装

推荐使用 Python 3.11。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install "setuptools<81"
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[agent,agent-runtime,dataset-quality,dev]"
```

选择相应后端时，需要另外安装 `claude`、`opencode` 或 `codex` CLI。Codex 通过 `codex exec --json` 调用；可用 `CODEX_BIN` 指定可执行文件。

## 配置

主配置文件是 `configs/default.yaml`。至少需要提供原始数据集、Review API、唯一的运行名称和任务描述：

```bash
export RAW_DATA_PATH=/path/to/raw-data/train.jsonl
export DF_API_URL=https://api.example/v1/chat/completions
export DF_API_KEY=...
export DF_API_MAX_WORKERS=10
export DF_RUN_NAME=experiment_001
export DF_MAX_ITERATIONS=5
export DF_EXECUTION_REPAIRS=1
export DF_COMPILE_TIMEOUT_SEC=120
export DF_AGENT_PYTHON="$(command -v python)"
```

`DF_MAX_ITERATIONS` 控制 Review 驱动的 Pipeline 迭代次数。`DF_EXECUTION_REPAIRS` 控制生成的 Pipeline 在制品校验、compile 预检、执行或输出校验失败后，额外调用 Agent 修复的次数。修复请求会收到失败阶段、traceback 和可用的逐步骤 JSONL 诊断。Agent 只负责编辑制品；框架始终执行确定性的 compile-only 预检，预检通过后才允许完整执行。Review 反馈会进入下一次正常迭代。

框架不强制指定目标样本数。Pipeline 从完整原始数据集开始，并根据任务要求对其进行清洗、筛选、转换或扩充。

请为选定的 PipelineAgent 后端配置与提供商无关的凭据：

```bash
export DF_AGENT_API_KEY=...
export DF_AGENT_BASE_URL=https://provider-compatible.example/
```

端点必须实现所选后端的协议：Claude 使用 Anthropic 兼容端点，OpenCode 使用其配置的 provider 协议，Codex 使用 OpenAI Codex/Responses 兼容端点。这些凭据与 `DF_API_*` 相互独立；后者用于配置 ReviewAgent 和生成后 Pipeline 中的 LLM 调用。

Claude 配置：

```bash
export AGENT_BACKEND=claude
export CLAUDE_AGENT_MODEL=claude-opus-4-8
```

OpenCode 配置：

```bash
export AGENT_BACKEND=opencode
export OPENCODE_BIN=opencode
export OPENCODE_MODEL=anthropic/<model-name>
export OPENCODE_PROVIDER=anthropic
```

Codex 配置：

```bash
export AGENT_BACKEND=codex
export CODEX_BIN=/path/to/codex
export CODEX_MODEL=gpt-5.4
export CODEX_EFFORT=high
```

后端专用的 `CLAUDE_AGENT_API_KEY_ENV`/`CLAUDE_AGENT_BASE_URL`、`OPENCODE_API_KEY_ENV`/`OPENCODE_BASE_URL` 和 `CODEX_API_KEY_ENV`/`CODEX_BASE_URL` 可以覆盖通用配置。旧版的 `ANTHROPIC_API_KEY`、`ANTHROPIC_BASE_URL`、`CODEX_API_KEY`、`OPENAI_API_KEY` 和 `OPENAI_BASE_URL` 仍作为后备配置保留兼容。

## 可选的 embedding 数据集评审

DAS 将可配置数量的候选样本与固定的高质量代理数据集进行比较。同一份候选 embedding 还会用于生成不依赖阈值的语义多样性证据：

```text
DAS(candidate) = -MMD(candidate_embeddings, proxy_embeddings)
diversity(candidate) = cosine Vendi Score + nearest-neighbour cosine profile
```

在本地准备代理数据和 embedding 模型，例如：

```text
data/ODA-Math-460k/
model/Qwen3-Embedding-8B/
```

启动一个 OpenAI 兼容的 embedding 端点，然后配置：

```bash
export DF_EMBEDDING_REVIEW_ENABLED=true
export DF_DAS_SAMPLE_SIZE=5000
export DF_DAS_PROXY_SOURCE=local
export DF_DAS_PROXY_PATH=$PWD/data/ODA-Math-460k
export DF_DAS_EMBEDDING_MODEL=$PWD/model/Qwen3-Embedding-8B
export DF_DAS_EMBEDDING_URL=http://127.0.0.1:8000/v1
```

候选数据默认使用 `instruction` 和 `output` 字段，代理数据默认使用 `question` 和 `response` 字段。使用其他 schema 时，请在 YAML 中同时修改两边的字段映射。每个被采样数据集的候选 embedding 只计算一次，并由 DAS、cosine Vendi Score 和 nearest-neighbour cosine 统计共同复用。这些数据集级信号既不需要领域 codebook，也不依赖固定的语义重复阈值，同时不会直接混合原始数值。启用数据集评审后，它们首先被校准为 0–1 的 embedding 质量分数，然后按配置比例计入 `review_score`：

```text
das_score = clip(1 - MMD / sqrt(2))
nn_score  = clip((1 - nearest_neighbor_cosine_p95) / 2)
embedding_quality = 0.40 * das_score + 0.30 * normalized_vendi + 0.30 * nn_score
review_score = (0.75 * llm_composite + 0.25 * embedding_quality) * hard_metric_factor
```

可通过 `DF_DATASET_QUALITY_SCORE_WEIGHT` 修改默认的 `0.25` 权重。

## 可选的周期性 SFT/评测反馈

框架可以在每隔若干轮静态 Review 后触发一次高成本下游信号：把当时的 best-so-far
数据集交给外部命令完成 SFT 和 benchmark 评测，严格验证各 benchmark 分数与完整 bad
cases，再将这些证据注入后续 PipelineAgent prompt。外部命令可以使用独立 Conda 环境，
DataFlow Evolver 不会安装或导入任何训练、CUDA、模型与评测依赖。

```yaml
downstream_eval:
  enabled: true
  interval: 2
  command:
    - /opt/conda/envs/downstream/bin/python
    - /absolute/path/to/run_sft_and_eval.py
  benchmarks: [gsm8k, math, math500, aime]
  baseline:
    enabled: true  # 第 0 轮仅评测未训练模型，不执行 SFT
  bad_cases_per_benchmark: 3  # 上限；实际错误少于 3 时允许返回更少
  timeout_sec: 86400
```

下游结果不会混入静态 `review_score`，只作为后续 Pipeline 变异的诊断依据，并完整写入
运行清单。请求/结果协议、失败语义及 benchmark 泄漏注意事项详见
[`docs/downstream_evaluation.md`](docs/downstream_evaluation.md)。

## 运行

```bash
python -m dataflow_evolver.main \
  --config configs/default.yaml \
  --task "Transform the supplied raw corpus into high-quality SFT records." \
  --log-level INFO
```

如果 `workspace/runs/<run_name>/` 已存在且非空，框架会拒绝启动。请使用新的运行名称，不要混用不同实验的制品。最终的 `manifest.json` 会记录选中的 Pipeline、数据集、Review 结果以及全部迭代。每轮 token 用量保存在 `token_usage.json` 中，并分为 `agent`、`pipeline_llm`、`review_llm` 和 `embedding` 四个独立命名空间。Review API 和 embedding 端点采用提供商返回的 usage；缺失的 usage 会被明确标记，而不会进行估算。如果 embedding 评估完全命中缓存，该轮会记录零请求和零 token。

## 测试

```bash
python -m pytest -q
```

Review 和修复语义详见 `docs/workflow.md`。
