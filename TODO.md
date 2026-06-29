# DataFlow-Evolve —— 整体设计方案与 TODO 文档

> 状态：**M0–M5 已实现并实测打通；M7 奖励锚点 / baseline 作 root / Relative Gain / 数据池准入已接线（默认关闭，受 `baseline.enabled` 开关控制）；M6 健壮性、M7 剩余实测项为后续**
> 最后更新：2026-06-26
> 本文档是总体蓝图 + 里程碑清单。设计层面的取舍（交付物定位、奖励、数据池、防泄露）见 [`docs/design-decisions.md`](./docs/design-decisions.md)。

---

## 0. 一句话目标

构建一个 **多智能体、MCTS 驱动的自动数据工程框架**：它能根据高层任务描述，自动地生成 / 变异 DataFlow 流水线 DAG，并通过 **双反馈环（Dual-Feedback Loop）**——快速的"左移"数据评审 + 真实的下游模型训练评估——不断自我修正，最终产出高质量训练数据集。

---

## 1. 关键调研结论（决定架构的事实）

来自对四个本地代码库的深入阅读，以下结论直接塑造了本方案：

| 代码库 | 我们要借鉴的东西 | 关键文件 |
|---|---|---|
| **DataFlow** (`Desktop\DataFlow\DataFlow`) | 流水线 = **Python 类**（`__init__` 配置 storage + llm_serving + operators；`forward()` 顺序调用 `op.run(storage=self.storage.step(), ...)`）。算子继承 `OperatorABC`；`FileStorage.step()` 做检查点；`OPERATOR_REGISTRY` 可枚举算子；serving 后端有 `APILLMServing_request` 与 `LocalModelLLMServing_vllm` | `md_to_qa_pipeline.py`、`dataflow/core/operator.py`、`dataflow/utils/storage.py`、`dataflow/utils/registry.py` |
| **DataFlow-Skills** (`...\DataFlow-Skills`) | `generating-dataflow-pipeline` 技能 = **两阶段生成**（① 算子决策 JSON `{ops, field_flow, reason}` → ② 完整流水线代码）；`dataflow-dev` 提供内部知识库与诊断文档 | `generating-dataflow-pipeline/SKILL.md`、`templates/pipeline_template.py` |
| **DataMaster** (`Desktop\DataMaster`) | **UCB1/MCTS** 选择-扩展-回传；**全局/子节点记忆** `fetch_child_memory()`；**子进程执行**含 timeout/OOM 捕获与资源分配；**JSON 指标解析** | `playground/ml_master/core/mcts_node.py`、`mcts_agent.py`、`evomaster/env/local.py`、`utils/response.py` |
| **ANDES-4DAE** (`Desktop\ANDES-4DAE`) | ⚠️ **仅做 LLM 数据合成，没有真实训练**。其"bad case"是 critique 的 effort score（>4 被直接丢弃、不保存）。可借鉴：Router→Generator→Refiner 分阶段思想、FileStorage/JSONL 约定、critique 正则抽取 | `andes/andes_refiner.py`、`andes/utils/storage.py` |

> **最重要的一条**：ANDES 并不执行真实训练，因此"真实下游训练 + 真实 Bad Case 抽取"必须**自行构建**，复用 DataMaster 的子进程 + 结果解析模式。根据已确认决策，`DownstreamTrainerWrapper` 采用 **可插拔外部脚本桥接** 方式。

---

## 2. 已确认的设计决策

1. **PipelineAgent = Claude Agent SDK 驱动（核心架构点）**
   - PipelineAgent **不是**常规的大模型 chat-completion API 调用模块；它通过 **Claude Agent SDK**（Python，`query()` 无头运行）来**实际编写 DataFlow 流水线代码**。
   - Python 框架只负责 **MCTS 主循环调度**（选点 / 扩展 / 评审 / 训练 / 诊断 / 回传）；"写代码"这件事完全交给 Agent SDK 内的 Claude 自主完成（它能 Read/Write/Edit 文件、调用本地 Skill）。
   - 通过 `setting_sources=["project"]` + 项目 `.claude/skills/` 暴露**两个本地技能**：**`generating-dataflow-pipeline`**（主：写流水线）与 **`dataflow-dev`**（辅：架构知识库 + `known_issues.md` 报错速查，用于自我修正与超出核心算子集的需求；连同 `core_text`）。详见 §5.3 与 §5.9。
   - `dataflow-dev` 假设 cwd 即 DataFlow 仓库根，因此把 **DataFlow 源码完整 copy** 进一个**共享 agent 根目录**（`workspace/_agent_root/`），与 `.claude/skills/` 并存；SDK 的 `cwd` 指向它，生成的 `pipeline.py` 写到其下的 node 子目录。其交互式（AskUserQuestion）与依赖 git/gh 的"知识库更新感知"子流程在无头模式下**禁用**。
2. **DownstreamTrainerWrapper = 可插拔外部脚本桥接**
   - 不内置具体训练逻辑；而是 `subprocess` 调用用户提供的"训练+评估"脚本。
   - 约定 I/O 契约（见 §5.6）：框架传入数据集路径与输出目录，脚本回写 `results.json`（含 `score` 与 `bad_cases`）。
   - 复用 DataMaster `evomaster/env/local.py` 的 timeout / OOM / 非零退出码捕获模式，**绝不让训练失败拖垮主编排器**。
3. **LLM 后端 = 可配置**（API OpenAI-compatible 或 本地 vLLM），通过 YAML 配置切换；底层统一走 DataFlow 的 `LLMServingABC` 抽象。此项针对 **DataFlow 流水线运行时** 以及非-SDK 的辅助 Agent（Profiler/Review/Diagnostic）的模型调用；PipelineAgent 走 Agent SDK（见决策 1）。
4. **构建顺序**：探索 ✅ → 目录脚手架 → 先定义全部 ABC 与 dataclass（严格类型）→ 再实现 Agents/Wrappers/事件环。
5. **文档语言**：中文。

---

## 3. 目标目录结构

```
DataFlow-Evolve/
├── dataflow_evolve/
│   ├── __init__.py
│   ├── agents/
│   │   ├── base.py              # AgentABC（统一 LLM 调用 / 重试 / JSON 解析）
│   │   ├── task_profiler.py     # TaskProfilerAgent：任务描述 → 形式化数据需求 schema
│   │   ├── pipeline_agent.py    # PipelineAgent：封装 Agent SDK 运行，写/变异流水线代码
│   │   ├── review_agent.py      # ReviewAgent：训练前的快速数据评审（schema 正确性 + 任务相关性）
│   │   └── diagnostic_agent.py  # DiagnosticAgent：Bad Case → 可执行诊断报告
│   ├── core/
│   │   ├── schemas.py           # 所有 dataclass：TaskProfile / PipelineConfig / ReviewResult / TrainResult / BadCase / DiagnosticReport / NodeStats
│   │   ├── tree_node.py         # PipelineNode（UCB1 统计、记忆、诊断报告存储）
│   │   ├── tree_manager.py      # TreeManager：select(UCB1)/expand/backpropagate + 全局记忆
│   │   ├── global_memory.py     # GlobalMemory：跨节点共享知识
│   │   └── data_pool.py         # DataPool：跨节点数据累积池（脚手架，默认关闭，见 docs/design-decisions.md §4）
│   ├── engine/
│   │   ├── execution_wrapper.py # ExecutionWrapper：运行生成的 DataFlow 流水线，捕获崩溃栈
│   │   └── trainer_wrapper.py   # DownstreamTrainerWrapper：外部训练脚本子进程桥接
│   ├── llm/
│   │   ├── serving.py           # 非-SDK 模型后端工厂（API / vLLM 可配置）
│   │   ├── agent_sdk.py         # Claude Agent SDK 封装：async query() → 同步 run_pipeline_agent()
│   │   └── prompts.py           # 各 Agent 的 prompt 模板
│   ├── utils/
│   │   ├── config.py            # YAML 配置加载（含 ${ENV} 展开）
│   │   ├── logging.py           # 结构化日志
│   │   └── parsing.py           # extract_json / extract_code 正则工具
│   └── main.py                  # 双反馈事件环编排器
├── configs/
│   └── default.yaml             # LLM 后端、Agent SDK、树搜索超参、路径、训练脚本命令、data_pool/baseline 段
├── docs/
│   └── design-decisions.md      # 设计取舍：交付物定位 / 奖励锚点 / 数据池 / 防泄露
├── scripts/
│   └── reference_train_eval.py  # 参考用外部训练/评估脚本 + I/O 契约样例
├── data/                        # 输入种子数据（jsonl）
├── workspace/                   # 运行产物
│   └── _agent_root/             # 共享 Agent SDK 工作目录（cwd）；一次性拷入下列内容
│       ├── .claude/skills/      # generating-dataflow-pipeline / core_text / dataflow-dev（从 DataFlow-Skills 拷入）
│       ├── dataflow/            # DataFlow 源码完整 copy（供 dataflow-dev 的 grep/registry 探测）
│       └── <node_id>/           # 每个节点：生成的 pipeline.py、数据集、日志、results.json
├── tests/
├── TODO.md                      # ← 本文件
├── pyproject.toml
└── README.md
```

---

## 4. 核心数据结构（`core/schemas.py`，先行定义）

```python
@dataclass
class TaskProfile:            # TaskProfilerAgent 输出，全局唯一
    task_description: str
    target_schema: dict       # 期望字段及类型，如 {"instruction": "str", "answer": "str"}
    domain: str
    quality_criteria: list[str]
    downstream_task: str      # 例如 "math-QA-SFT"

@dataclass
class PipelineConfig:         # PipelineAgent 产物（一个节点对应一个配置）
    operators: list[dict]     # [{name, params}, ...]
    field_flow: str           # "text -> chunk -> qa_pairs"
    code: str                 # 完整可执行的 DataFlow 流水线 Python 源码
    rationale: str

@dataclass
class ReviewResult:           # Phase-1 快反馈
    schema_score: float       # 0-1
    relevance_score: float    # 0-1
    passed: bool
    issues: list[str]

@dataclass
class BadCase:                # 来自真实评估
    input: str
    prediction: str
    ground_truth: str
    error_type: str | None

@dataclass
class TrainResult:            # Phase-2 慢反馈（DownstreamTrainerWrapper 解析得到）
    score: float
    metric_name: str
    bad_cases: list[BadCase]
    raw_log_path: str
    failed: bool              # 训练/评估系统级失败（OOM/timeout 等）
    error_msg: str | None

@dataclass
class DiagnosticReport:       # DiagnosticAgent 产物，回写节点，指导下一轮变异
    summary: str              # 例："模型在含分数的几何题上失败"
    failure_patterns: list[str]
    actionable_suggestions: list[str]   # 例："增加 Web Search RAG 几何分数领域数据"
```

---

## 5. 模块职责与实现要点

### 5.1 TaskProfilerAgent（初始化时运行一次）
- 输入：高层任务描述（+ 可选样例数据）。
- 输出：`TaskProfile`。
- 实现：单次 LLM 调用 + JSON 解析（复用 `utils/parsing.extract_json`）。

### 5.2 TreeManager + PipelineNode（借鉴 DataMaster）
- `PipelineNode` 持有：`config: PipelineConfig`、UCB1 统计（`visits`、`total_reward`）、`review_result`、`train_result`、`diagnostic_report`、父子指针。
- `uct_value()` 复用 DataMaster 公式：`Q/n + c*sqrt(ln(N)/n)`，未访问节点返回 `inf`。
- `select()`：自虚拟根沿最大 UCT 下行至可扩展节点。
- `backpropagate(reward)`：沿祖先链累加。
- `GlobalMemory`：聚合各节点的 `(config 摘要, score, diagnostic 摘要)`，供 PipelineAgent 生成时参考（借鉴 `fetch_child_memory`）。

### 5.3 PipelineAgent（构建者 / 变异者，**Claude Agent SDK 驱动**）
- **本质**：它不直接产出文本，而是**启动一次无头 Agent SDK 运行**，让 SDK 内的 Claude 借助本地技能与文件工具，把一份完整可运行的 DataFlow 流水线 `.py` **写到该节点的 workspace**。
- 输入（拼进 prompt）：`TaskProfile` + 父节点 `DiagnosticReport`（若有）+ `GlobalMemory` 摘要 + 种子样例数据路径。
- 调用方式（见 §5.9）：`run_pipeline_agent(prompt, cwd=workspace/<node_id>)`，内部 `asyncio.run(query(...))`。
- prompt 指令要点：
  1. 显式要求"使用 `generating-dataflow-pipeline` 技能"，传入 `Target` 与 `Sample file`（种子 jsonl 路径）。
  2. 把流水线写到约定文件名（如 `<node_id>/pipeline.py`），`first_entry_file_name` 指向种子路径。
  3. **若存在诊断报告**：把 `actionable_suggestions` 注入，要求据此调整算子选择 / 参数（有的放矢的变异）；超出 6 个核心算子时，准许查阅 `dataflow-dev` 的 `knowledge_base.md` 找合适算子。
  4. **自修正时**（执行崩溃后重试）：要求用 `dataflow-dev` 的 `known_issues.md` 速查表匹配报错并修复。
  5. 明确禁止 `dataflow-dev` 的交互式（AskUserQuestion）与 git/gh"知识库更新感知"子流程（无头不可用）。
- 产出：读取 workspace 下生成的 `pipeline.py` 及 Stage-1 算子决策 JSON → 组装 `PipelineConfig`（`code` / `operators` / `field_flow` / `rationale`）。
- 失败处理：捕获 `CLINotFoundError` / `ProcessError` / `CLIJSONDecodeError`；找不到生成文件视为本次扩展失败 → 惩罚节点。

### 5.4 ExecutionWrapper
- 执行 PipelineAgent 产出的 DataFlow 流水线代码（隔离子进程或受控 `exec`，写入 `workspace/<node_id>/`）。
- 捕获崩溃栈：失败时把 stack trace 反馈给 PipelineAgent 做自我修正（事件环步骤 4）。
- 产出：生成的数据集路径（jsonl）。

### 5.5 ReviewAgent（Phase-1 快反馈，训练前）
- 输入：生成的数据集（抽样）+ `TaskProfile`。
- 评分：`schema_score`（字段/类型符合度）+ `relevance_score`（任务相关性）。
- 低于阈值 → `passed=False` → 惩罚节点、**不训练**、回到选点。

### 5.6 DownstreamTrainerWrapper（Phase-2 真实执行，外部脚本桥接）
- **I/O 契约**（写入 README 与 `scripts/reference_train_eval.py`）：
  - 框架调用：`python <train_eval_cmd> --dataset <ds.jsonl> --output <out_dir>`
  - 脚本必须在 `<out_dir>/results.json` 回写：
    ```json
    {
      "score": 0.0,
      "metric_name": "accuracy",
      "bad_cases": [
        {"input": "...", "prediction": "...", "ground_truth": "...", "error_type": "..."}
      ]
    }
    ```
- 用 `subprocess.run(..., timeout=..., capture_output=True)`，捕获 `TimeoutExpired` 与非零退出码 → 返回 `TrainResult(failed=True, error_msg=...)`，**不抛出到主循环**。
- 解析 `results.json` → `TrainResult`。

### 5.7 DiagnosticAgent（Phase-2 慢反馈 / Bad Case 分析）
- 输入：`TrainResult.bad_cases`。
- 输出：`DiagnosticReport`（聚类失败模式 → 可执行建议）。
- 实现：将 bad cases 采样后交 LLM 归因，强制结构化 JSON 输出。

### 5.8 main.py 双反馈事件环
```
1. TaskProfilerAgent → TaskProfile（一次）
2. node = TreeManager.select()
3. config = PipelineAgent.generate(profile, node.diagnostic_report, global_memory)
4. dataset = ExecutionWrapper.run(config)
     └ 崩溃 → 把栈喂回 PipelineAgent 自修正（限次重试），仍失败则惩罚并回 2
5. review = ReviewAgent.review(dataset, profile)        # Phase-1
     └ 未过 → 惩罚节点，不训练，回 2
6. train = DownstreamTrainerWrapper.run(dataset)        # Phase-2 真实训练
     report = DiagnosticAgent.analyze(train.bad_cases)
7. TreeManager.backpropagate(node, reward=train.score)
   node.train_result = train; node.diagnostic_report = report; GlobalMemory.update(...)
8. 回到 2（下次扩展该分支时，PipelineAgent 必须读取诊断报告）
```

### 5.9 Claude Agent SDK 集成要点（`llm/agent_sdk.py`）
- **安装**：`pip install claude-agent-sdk`（自带 Claude Code CLI，无需单独装）。需要环境变量 `ANTHROPIC_API_KEY`（或 Bedrock/Vertex/Foundry 开关）。Python 3.10+。
- **入口**：`query()` 是 async 生成器：
  ```python
  import asyncio
  from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, ResultMessage

  async def _run(prompt: str, cwd: str) -> str:
      result_text = ""
      async for msg in query(
          prompt=prompt,
          options=ClaudeAgentOptions(
              cwd=cwd,                                   # = workspace/_agent_root（含 .claude/skills + dataflow 源码）；pipeline 写到其下 node 子目录
              allowed_tools=["Read", "Write", "Edit", "Glob", "Grep", "Bash"],
              permission_mode="bypassPermissions",       # 无头自动化（或 "acceptEdits"）
              setting_sources=["project"],               # 关键：加载 cwd/.claude/（三个技能在此）
              system_prompt="You write DataFlow pipelines. Use the generating-dataflow-pipeline skill to author; consult dataflow-dev (knowledge_base/known_issues) for ops beyond the core set and to fix execution errors. Do NOT run interactive or git/gh repo-sync subflows.",
          ),
      ):
          if isinstance(msg, ResultMessage):
              result_text = msg.result
      return result_text

  def run_pipeline_agent(prompt: str, cwd: str) -> str:   # 供同步 MCTS 循环调用
      return asyncio.run(_run(prompt, cwd))
  ```
- **技能发现（关键）**：SDK 默认不加载任何文件系统设置。必须 `setting_sources=["project"]`（从 `cwd` 下的 `./.claude/` 读取）。`cwd` 设为共享的 `workspace/_agent_root/`，其中放三个技能（`generating-dataflow-pipeline`/`core_text`/`dataflow-dev`）和一份 **DataFlow 源码完整 copy**（供 `dataflow-dev` 的相对 grep / registry 探测生效）。这是**一次性手动拷贝**（命令记录在 README / M0），无需专门脚本。
- **权限模式**：自动化用 `bypassPermissions`（沙箱/CI）或 `acceptEdits`（自动批准编辑）。
- **错误**：捕获 `CLINotFoundError` / `CLIConnectionError` / `ProcessError`（含 `.exit_code`）/ `CLIJSONDecodeError`。
- **可配置项**进 `configs/default.yaml`：`model`、`permission_mode`、`allowed_tools`、`setting_sources`、技能名清单。

---

## 6. 里程碑 / TODO 清单

### M0 — 项目脚手架
- [x] 创建 §3 目录结构与空 `__init__.py`
- [x] `pyproject.toml`（依赖：本地 DataFlow 包、`claude-agent-sdk`、pydantic/dataclasses、pyyaml、openai 等）
- [x] `configs/default.yaml`（LLM 后端选择、Agent SDK 配置、UCB1 超参 c、阈值、训练脚本命令、路径）
- [x] `utils/config.py`（YAML + `${ENV}` 展开）、`utils/logging.py`、`utils/parsing.py`
- [x] 确认本地 DataFlow 包的导入方式（结论：`_agent_root/dataflow/` 为 Agent 参考用；实跑已在 env 内安装 `open-dataflow` v1.0.10，`import dataflow` 可用）
- [x] **一次性**准备 `workspace/_agent_root/`：拷三个技能（`generating-dataflow-pipeline`/`core_text`/`dataflow-dev`）进 `.claude/skills/`，并**完整 copy** DataFlow 源码进 `_agent_root/dataflow/`（命令记入 README）
- [x] 验证 `claude-agent-sdk` 可安装（0.2.88）、CLI 自带可用；✅ `ANTHROPIC_API_KEY` 已配（复用同一 `sk-` 密钥 + `ANTHROPIC_BASE_URL` 代理），实跑连通

### M1 — 类型与抽象基类（先定义，严格类型）
- [x] `core/schemas.py`：实现 §4 全部 dataclass
- [x] `agents/base.py`：`AgentABC`（`__call__` / LLM 调用 / 重试 / JSON 解析）
- [x] `engine` 两个 Wrapper 的 ABC 接口签名
- [x] `llm/serving.py`：非-SDK 后端工厂（API / vLLM），返回统一 `LLMServingABC`
- [x] `llm/agent_sdk.py`：`run_pipeline_agent()` 同步封装（§5.9），含 SDK 错误捕获

### M2 — 树搜索内核（借鉴 DataMaster）
- [x] `core/tree_node.py`：`PipelineNode` + `uct_value()`
- [x] `core/tree_manager.py`：`select` / `expand` / `backpropagate`
- [x] `core/global_memory.py`：跨节点记忆聚合
- [x] 单测：纯 mock reward 下 UCB1 选点行为正确（`tests/test_tree.py`）

### M3 — 智能体实现
- [x] `TaskProfilerAgent`（非-SDK，单次 LLM + JSON 解析）
- [x] `PipelineAgent`（**Agent SDK 驱动**：拼 prompt 注入 Profile/诊断/记忆 → `run_pipeline_agent` → 解析生成的 `pipeline.py`）
- [x] 验证 Agent SDK 能发现并调用 `generating-dataflow-pipeline` 技能、产出可运行流水线 ✅ **已实测通过**（2026-06-03）：SDK 经 `ANTHROPIC_BASE_URL=http://35.220.164.252:3888/` 代理 + 复用同一 `sk-` 密钥连通 Anthropic 协议；`setting_sources=["project"]` 成功发现技能并写出 `pipeline.py`+`decision.json`；生成的流水线嵌入了正确的代理 API 配置（api_url/model_name/`DF_PIPELINE_API_KEY`）
- [x] `ReviewAgent`（schema + 相关性打分）
- [x] `DiagnosticAgent`（bad case → 可执行报告）

### M4 — 执行与训练引擎
- [x] `ExecutionWrapper`：子进程运行 DataFlow 流水线 + 崩溃栈捕获
- [x] `DownstreamTrainerWrapper`：外部脚本桥接 + timeout/OOM 防御 + `results.json` 解析
- [x] `scripts/reference_train_eval.py`：实现并文档化 I/O 契约的参考脚本（含一个可跑通的最小例子）

### M5 — 主事件环与端到端
- [x] `main.py`：实现 §5.8 双反馈环 + 自修正重试 + 优雅降级
- [x] 端到端冒烟测试：桩替换 SDK/LLM，跑通真实 select/执行/参考训练/回传（`tests/test_event_loop.py`）；✅ **真·全链路已验证**（2026-06-03）：DataFlow 已安装（open-dataflow v1.0.10）于 DataFlow-Evolve 环境；Agent SDK 实跑写出流水线 → 真实 DataFlow 执行产出 jsonl → ExecutionWrapper 正确选中非空 step 文件
- [x] `README.md`：安装、配置、运行、外部训练脚本契约说明

### M6 — 健壮性与打磨（可选 / 后续）
- [x] 最大轮数 / 最大节点数：`search.max_iterations`（一次迭代=一个节点，已实现）
- [ ] token/USD 全局预算熔断（`agent_sdk.max_budget_usd` 仅限单次 SDK 调用；全局上限暂不做）
- [ ] 运行产物与树状态的持久化 / 断点续跑
- [x] 探索常数 c 的衰减策略（借鉴 DataMaster mcts_utils）：`search.exploration_decay`/`exploration_min`，按已评估节点数指数衰减，默认 1.0 不衰减

### M7 — 交付物定位 / 奖励锚点 / 数据池 / 防泄露（设计见 `docs/design-decisions.md`）
> 决策：**主贡献 = pipeline**（data 为证据）；奖励 = **Relative Gain**（相对强基线，Δ=0、seed=42）；data pool 与奖励**解耦**、默认关闭。
- [x] 设计文档（`docs/design-decisions.md`）+ memory 落盘讨论结论
- [x] `configs/default.yaml`：`trainer.dev_eval_cmd` / `test_eval_cmd` + `data_pool` + `baseline` 配置段（脚手架，默认关闭）
- [x] `core/data_pool.py`：DataPool 接口骨架 + 内容哈希去重 + per-pipeline 上限（**准入策略/接线为 TODO**）
- [x] `README.md`：dev/test 评测契约 + 去污染 + 数据池说明
- [x] **已接线（2026-06-26）**：main.py 接入 Relative Gain 奖励 + baseline 作 root（warm start，开 `baseline.enabled` 时先用 seed=42 跑一次基线得 `baseline_dev_score`；关时退回当前虚拟空根冷启动）。新增 `core/baseline.py`（`RewardPolicy` + `adapt_baseline_source`）、`TreeManager.set_baseline_root`、`parent_code` 变异注入、`trainer.dev_eval_cmd` 回退；单测 `tests/test_baseline.py`（warm/cold/Relative Gain/penalty 切换/池门槛）；27 个既有测试不受影响全绿。
- [x] **已接线（2026-06-26）**：main.py 第⑧步接入池（仅 review 通过且 `node_dev ≥ baseline_dev + Δ`（暖）/ `score_threshold`（冷）的节点收割 + 落盘）
- [ ] **待实测**：行级 bad-case-aware 上采样（命中 failure_patterns 的行保留/加权）
- [ ] **待实测**：去污染脚本（生成源语料 vs test benchmark 的 n-gram 重叠过滤）
- [ ] **待实测**：实验与对照（设计文档 §5）——独立 SOTA 对照 + 承重消融（冷启动独立性 / 搜索贡献 / 双反馈）+ 池 vs 单 pipeline 等量对比
- [ ] **可选（后做）**：改主打 pool 时的边际贡献奖励

---

## 7. 风险与开放问题
- **Agent SDK 环境就绪**：`claude-agent-sdk` 需可安装、`ANTHROPIC_API_KEY` 配好、自带 CLI 可运行；本环境外网被屏蔽，需确认 SDK 运行时不依赖被拦截的域名。
- **技能发现可靠性**：需实测 `setting_sources=["project"]` + `_agent_root/.claude/skills/` 能让 SDK 正确发现并调用三个技能；并确认无头模式下 `dataflow-dev` 不会触发交互式 / git-gh 子流程。
- **DataFlow 源码 copy 的维护**：`_agent_root/dataflow/` 是上游 DataFlow 的拷贝，会随上游更新而变旧；需在 README 记录拷贝时的版本，需要时手动重新拷贝同步。
- **DataFlow 包安装与算子可用性**：实现 M0 时需确认本地 DataFlow 能否 import 及其算子注册是否正常。
- **训练脚本的真实存在性**：当前为可插拔契约；真正跑 Phase-2 需要用户提供符合契约的训练脚本（或先用 `reference_train_eval.py` 验证管路）。
- **流水线代码执行的安全性**：Agent 生成代码在子进程中执行，需限定工作目录与资源；后续可考虑沙箱。
- **成本/时延**：每次扩展都启动一次 Agent SDK 运行（多轮工具调用），比单次 API 调用更慢更贵；需配 `max_turns`/预算上限。
- **奖励 / 数据池 / 泄露 / 通用性调参**：相关设计决策（Relative Gain 奖励、池与奖励解耦、dev/test 防泄露、以基线锚消除按任务调参）已统一收敛到 [`docs/design-decisions.md`](./docs/design-decisions.md)，此处不再重复。

---

> 下一步：M6 健壮性 / M7 数据池与奖励锚点的"待实测"项，需真实 dev 训练脚本就绪后推进。本文档随实现进度更新勾选状态。
