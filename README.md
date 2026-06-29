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
  让 SDK 内的 Claude 借助本地技能 `generating-dataflow-pipeline` / `dataflow-dev`
  把完整可运行的 DataFlow 流水线 `.py` 写到节点工作目录。
- **DownstreamTrainerWrapper** 是 **可插拔外部脚本桥接**：框架不内置训练逻辑，
  而是按 I/O 契约子进程调用你提供的"训练+评估"脚本。

---

## 安装

```bash
# 1. 创建并激活 conda 环境（名字固定为 DataFlow-Evolve）
conda create -n DataFlow-Evolve python=3.11 -y
conda activate DataFlow-Evolve

# 2. 安装本项目（含 claude-agent-sdk / openai / pyyaml / pydantic）
pip install -e .
pip install -e ".[dev]"   # 含 pytest

# 3. 真正运行 DataFlow 流水线（Phase-2 之前）需另装 DataFlow 包（重依赖 torch/transformers）
#    cd /path/to/DataFlow && pip install -e .
```

> **Windows 提示**：用 `conda run` 跑会 spawn 子进程的命令（测试、主循环）可能崩溃；
> 请直接用环境内解释器：`"<conda>/envs/DataFlow-Evolve/python.exe" -m pytest tests/ -q`。

### 一次性准备 Agent 工作目录 `workspace/_agent_root/`

Agent SDK 的 `cwd` 指向共享的 `workspace/_agent_root/`，其中需并存技能与 DataFlow 源码副本
（`dataflow-dev` 技能假设 cwd 即 DataFlow 仓库根）。一次性拷贝：

```bash
mkdir -p workspace/_agent_root/.claude/skills
# 三个技能（generating-dataflow-pipeline / dataflow-dev / core_text，从 DataFlow-Skills 拷入）
cp -r <DataFlow-Skills>/generating-dataflow-pipeline workspace/_agent_root/.claude/skills/
cp -r <DataFlow-Skills>/dataflow-dev               workspace/_agent_root/.claude/skills/
cp -r <DataFlow-Skills>/core_text                  workspace/_agent_root/.claude/skills/
# DataFlow 源码完整副本（供 dataflow-dev 的 grep / registry 探测）
cp -r <DataFlow>/dataflow                          workspace/_agent_root/dataflow
```

> `_agent_root/dataflow/` 是上游 DataFlow 的拷贝，会随上游更新而变旧，必要时手动重拷同步。

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

---

## 运行

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
python <train_eval_cmd> --dataset <ds.jsonl> --output <out_dir>
```

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
把 `configs/default.yaml` 的 `trainer.train_eval_cmd` 指向它即可先验证管路。

---

## 防数据泄露：dev / test 评测纪律

MCTS 反复用同一 benchmark 当奖励会**对它过拟合**；若再用同一 benchmark 报告结果，分数虚高、不可泛化。
因此评测必须分两套（设计见 [`docs/design-decisions.md`](./docs/design-decisions.md) §3）：

- **`dev`**（held-in）：**只**给 MCTS 当奖励信号，允许被「刷」——配 `trainer.dev_eval_cmd`（留空则回退 `train_eval_cmd`）。
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
"<conda>/envs/DataFlow-Evolve/python.exe" -m pytest tests/ -q
```

覆盖：配置/解析工具、UCB1 树搜索、辅助 Agent（桩后端）、执行与训练引擎、
以及不依赖 API 密钥的主事件环集成冒烟测试。

---

## 目录结构

```
dataflow_evolve/
├── agents/      # base / task_profiler / pipeline_agent(SDK 驱动) / review / diagnostic
├── core/        # schemas / tree_node / tree_manager / global_memory
├── engine/      # execution_wrapper / trainer_wrapper
├── llm/         # serving(API/vLLM) / agent_sdk / prompts
├── utils/       # config / logging / parsing
└── main.py      # 双反馈事件环编排器
configs/         # default.yaml
scripts/         # reference_train_eval.py（外部训练脚本契约示例）
workspace/       # 运行产物 + _agent_root（Agent SDK cwd）
tests/
```
