# DataFlow-Evolve —— 整体设计方案与 TODO 文档

> 状态：**M0–M5 已实现并实测打通；M8 架构重整、M9 dev 题集物化与训练留痕、M10 外部数据溯源、M11 全量下载与连接配置、M12 第二个 hub 与发现渠道、M13 外部数据工具线上实测与修复、M14 训练集切齐、M15 产出 schema 契约与评测数据出库已完成；M6 健壮性与各项实测为后续**
> 最后更新：2026-08-02
> 本文档是总体蓝图 + 里程碑清单。设计层面的取舍（交付物定位、起点与奖励、输入契约、防泄露）见 [`docs/design-decisions.md`](./docs/design-decisions.md)。

**架构重整（2026-07-31，M8）**：
- **删除 `baseline` 模块**：一条基线曾同时充当"搜索根 / 奖励锚点 / 论文对照行"三个角色，改动其一必然牵动其二。
  现在根就是 **Init node（node_0）**，一个持有真实流水线、与其余节点同口径评分的普通实验节点；
  奖励即节点自己的分（原来的 Relative Gain 只是纯平移，本就不影响搜索）；论文对照回归论文层面。
- **删除 `data_pool` 模块**：与"pipeline 为主"的定位不一致，且带来一整套只服务于默认关闭副产品的准入策略。
  节点数据各存各的，需要并集时事后拼即可。
- **新增 `input/` 契约**：目录有语料 = corpus-grounded 合成，空 = from-scratch 合成，范式由目录内容决定。
- **新增评分策略 `core/evaluation.py`**：train/review 两种奖励口径连同其连带纪律收敛到策略对象。
- **新增外部数据搜索与去污染**：`llm/mcp_datasets.py`（取数：HF + ModelScope）+ `llm/mcp_discovery.py`
  （找线索：web/github/arXiv）+ `core/decontamination.py`（n-gram 硬指标）。前两者默认关闭。

**架构演进说明（2026-07-03）**：
- PipelineAgent 已从"选择已有算子"模式演进为"**编写自定义算子**"模式。Agent SDK 现在为每个节点在 `operators/` 目录下编写继承 `OperatorABC` 的自定义算子，仅将已有算子作为编码风格参考（通过 MCP 查看源码）。
- 原 `generating-dataflow-pipeline` 技能已弃用，当前仅使用 `dataflow-dev` 技能（含算子/Pipeline 模板、知识库、已知问题诊断）。
- 多模型 serving 支持：`pipeline_llm` 配置段现按用途区分（如 `generation` 用 GPT-5.5，`reasoning` 用 DeepSeek-V4-Pro），每个算子根据功能绑定对应 serving 实例。
- 下文中提及旧技能/旧选择模式的段落为历史设计记录，实际实现以上述演进为准。

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
├── dataflow_evolver/
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
│   │   ├── input_corpus.py      # InputCorpus：input/ 契约与范式判定、data preview
│   │   ├── dev_set.py           # DevSet：dev 题集物化，保证所有节点评在同一批题上
│   │   ├── evaluation.py        # EvaluationStrategy：train / review 两种奖励口径
│   │   ├── decontamination.py   # ContaminationIndex：与 benchmark test 的 n-gram 去污染
│   │   ├── provenance.py        # 外部数据溯源：工具调用日志 + 节点代码静态扫描
│   │   └── checkpoint.py        # 断点续跑：树/profile/进度序列化
│   ├── engine/
│   │   ├── execution_wrapper.py # ExecutionWrapper：运行生成的 DataFlow 流水线，捕获崩溃栈
│   │   └── trainer_wrapper.py   # DownstreamTrainerWrapper：外部训练脚本子进程桥接
│   ├── llm/
│   │   ├── serving.py           # 非-SDK 模型后端工厂（API / vLLM 可配置）
│   │   ├── agent_sdk.py         # Claude Agent SDK 封装：async query() → 同步 run_pipeline_agent()
│   │   ├── mcp_operators.py     # in-process MCP：算子源码参考
│   │   ├── mcp_datasets.py      # in-process MCP：取数（HF + ModelScope，默认关闭）
│   │   ├── mcp_discovery.py     # in-process MCP：找线索（web/github/arXiv，不落盘）
│   │   └── prompts.py           # 各 Agent 的 prompt 模板
│   ├── utils/
│   │   ├── config.py            # YAML 配置加载（含 ${ENV} 展开）
│   │   ├── logging.py           # 结构化日志
│   │   └── parsing.py           # extract_json / extract_code 正则工具
│   └── main.py                  # 双反馈事件环编排器
├── configs/
│   └── default.yaml             # LLM 后端、Agent SDK、树搜索超参、路径、训练脚本命令、init/去污染/dataset_search 段
├── docs/
│   └── design-decisions.md      # 设计取舍：交付物定位 / Init node / 输入契约 / 评分口径 / 防泄露
├── scripts/
│   └── reference_train_eval.py  # 参考用外部训练/评估脚本 + I/O 契约样例
├── input/                       # 输入语料目录（有数据=grounded 合成，空=from-scratch；内容不入库）
├── workspace/                   # 运行产物
│   ├── eval/                    # 物化的 dev 题集 dev.jsonl + dev.manifest.json
│   ├── data_links/              # 外部数据集下载目录（跨节点共享，含 .meta.json 溯源）
│   └── _agent_root/             # 共享 Agent SDK 工作目录（cwd）；一次性拷入下列内容
│       ├── .claude/skills/      # dataflow-dev / core_text（从 DataFlow-Skills 拷入）
│       ├── dataflow/            # DataFlow 源码完整 copy（供 dataflow-dev 的 grep/registry 探测）
│       └── <node_id>/           # 每个节点：生成的 pipeline.py、数据集、日志、results.json
│                                #   + agent_tools.jsonl / data_provenance.json（外部数据溯源）
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
- `select()`：自 Init node（node_0，树根）沿最大 UCT 下行至可扩展节点。
- `backpropagate(reward)`：沿祖先链累加。
- `GlobalMemory`：聚合各节点的 `(config 摘要, score, diagnostic 摘要)`，供 PipelineAgent 生成时参考（借鉴 `fetch_child_memory`）。

### 5.3 PipelineAgent（构建者 / 变异者，**Claude Agent SDK 驱动**）
- **本质**：它不直接产出文本，而是**启动一次无头 Agent SDK 运行**，让 SDK 内的 Claude 借助本地技能与文件工具，把一份完整可运行的 DataFlow 流水线 `.py` **写到该节点的 workspace**。
- 输入（拼进 prompt）：`TaskProfile` + `InputCorpus`（入口路径、范式、data preview）+ 父节点 `pipeline.py` /
  `rationale` / Tier-1 归因 + `DiagnosticReport`（若有）+ `GlobalMemory` 摘要 + 兄弟节点已试方向负面清单。
- 两个入口：`generate_initial()` 建 Init node（三档启动模式），`generate()` 做变异（注入 `parent_code`）。
- 调用方式（见 §5.9）：`run_pipeline_agent(prompt, cwd=workspace/<node_id>)`，内部 `asyncio.run(query(...))`。
- prompt 指令要点：
  1. 借助 `dataflow-dev` 技能与 `dataflow-operators` MCP（算子源码参考）编写自定义算子。
  2. 把流水线写到约定文件名（如 `<node_id>/pipeline.py`），`first_entry_file_name` 指向 `InputCorpus.entry_path`
     （grounded 时是真实语料，scratch 时是 0 字节 `empty.jsonl`）。
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
- 硬指标：`schema_score`（字段/类型符合度）+ `relevance_score`（任务相关性）+
  `contamination_rate`（与 benchmark test 的 n-gram 重合，见 `core/decontamination.py`）。
- 软指标：Tier-1 五维 LLM 评审，合成复合分（`reward_source=review` 时即为搜索期奖励）。
- 污染率按比例给 tier1 打折；超过 `review.decontamination.max_rate` 直接 `passed=False`。
- 低于阈值 → 同节点自修正环（喂 issues 原文 + 中间产物位置），限次后判 `passed=False`、**不训练**、回到选点。

### 5.6 DownstreamTrainerWrapper（Phase-2 真实执行，外部脚本桥接）
- **I/O 契约**（写入 README 与 `scripts/reference_train_eval.py`）：
  - 框架调用：`python <train_eval_cmd> --dataset <ds.jsonl> --output <out_dir> [--dev-set <dev.jsonl>]`
  - 配了 `trainer.dev_set.source` 时追加 `--dev-set`：脚本只在这份固定题集上评测，整份评完；
    脚本侧不得保留任何抽样/默认题库退路，拿不到题集应当直接报错（README「脚本负责什么、不负责什么」）
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
0. InputCorpus.discover(input_dir) → 入口文件 + 范式（grounded / scratch）+ data preview
   dev_set.build(trainer.dev_set) → 物化 dev 题集（全程同一批题）+ 并入去污染参考集
1. TaskProfilerAgent → TaskProfile（一次）
2. Init node：config = PipelineAgent.generate_initial(...) → _evaluate_node → root 得第一个分
     └ 起点跑不出分 → 直接终止本轮（没有可变异的基础）
3. node = TreeManager.select()
4. config = PipelineAgent.generate(profile, parent_code, parent_review, diagnostic, memory)
5. _evaluate_node(node)：
     a. dataset = ExecutionWrapper.run(config)
          └ 崩溃 → 把栈喂回 PipelineAgent 自修正（限 self_correction_retries），仍失败则惩罚
     b. review = ReviewAgent.review(dataset, profile)              # Phase-1（含去污染）
          └ 未过 → 同节点自修正环（限 review_self_correction_retries），仍未过则惩罚
     c. 由 EvaluationStrategy 决定要不要 Phase-2：
          train 口径 → DownstreamTrainerWrapper.run(dataset) + DiagnosticAgent.analyze(bad_cases)
          review 口径 → 跳过训练，用 Tier-1 复合分作奖励
6. TreeManager.backpropagate(node, reward)   # 失败恒为 PENALTY_REWARD=0
   node.train_result / diagnostic_report / review_result 落盘；GlobalMemory.update(...)
7. 回到 3
8. 收尾（仅 review 口径）：按 tier1 取 top-K → 放大重产 + 真实 SFT + dev 评测，把代理分兑现成真分
9. 终选 TreeManager.best_node()（含 root）→ 跑一次 test_eval_cmd 用于报告
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
- [x] 运行产物与树状态的持久化 / 断点续跑：`search.checkpoint` 非空则每轮落盘树状态+奖励策略+池暂存+进度+profile，启动时存在则载入续跑（`core/checkpoint.py`）；默认空=关闭
- [x] 探索常数 c 的衰减策略（借鉴 DataMaster mcts_utils）：`search.exploration_decay`/`exploration_min`，按已评估节点数指数衰减，默认 1.0 不衰减

### M7 — 交付物定位与防泄露（设计见 `docs/design-decisions.md`）
> 决策：**主贡献 = pipeline**（data 为证据）；不维护跨节点数据池；防泄露走 dev/test 隔离 + 去污染硬指标。
- [x] 设计文档（`docs/design-decisions.md`）+ memory 落盘讨论结论
- [x] `configs/default.yaml`：`trainer.dev_eval_cmd` / `test_eval_cmd` 配置段
- [x] `README.md`：dev/test 评测契约 + 去污染说明
- [ ] **待实测**：行级 bad-case-aware 上采样（命中 failure_patterns 的行保留/加权）
- [ ] **待实测**：实验与对照（设计文档 §6）——独立 SOTA 对照 + 承重消融（起点独立性 / 搜索贡献 / 双反馈 / 输入范式 / 外部数据）

### M8 — 架构重整：Init node / 输入契约 / 评分策略 / 外部数据 / 去污染（2026-07-31）
> 详见 [`docs/design-decisions.md`](./docs/design-decisions.md) §1.1、§2、§3、§4、§5。
- [x] **删除 `core/baseline.py`**（`RewardPolicy` / `adapt_baseline_source` / `set_baseline_root`）：
      奖励回归"节点自己的分"，失败恒为 `PENALTY_REWARD=0`，不再有随模式切换的 penalty。
- [x] **删除 `core/data_pool.py`** 及其准入策略、`pool_staging/` 落盘与 checkpoint 相关段。
- [x] **Init node**：`Orchestrator._run_init_node` + `PipelineAgent.generate_initial` + 三档启动模式
      （`init.reference_pipeline` 忠实改写 / `init.instruction` 指令构建 / 两空自主设计，均可命令行覆盖）；
      `TreeManager.best_node()` 把 root 纳入终选候选；起点跑不出分直接终止本轮。
- [x] **`core/input_corpus.py`**：`input/` 契约，目录内容决定 grounded / scratch 范式；
      入口文件行数/字段/前几行原文注入生成 prompt；多文件时确定性选择 + 告警，`project.input_entry` 可显式指定。
- [x] **`core/evaluation.py`**：`TrainScore` / `ReviewScore` 策略对象统一执行规模、是否训练、
      是否需要收尾兑现、时间预算预留；未知 `reward_source` 直接抛异常。
- [x] **`llm/mcp_datasets.py`**：外部数据集 search/inspect/preview/download（`agent_sdk.external_data`，默认关闭）；
      统一落 `data_links/` + `.meta.json` 溯源 + 体积上限；依赖惰性导入，缺包不影响主循环。
      （M11 改全量下载与字节上限，M12 加 ModelScope 一组同契约工具。）
- [x] **`core/decontamination.py`**：与 benchmark test 的归一化 n-gram 重合检测；
      `ReviewAgent` 按比例给 tier1 打折 + `max_rate` 硬门槛判不通过 + 命中片段作为 issue 回喂。
- [x] checkpoint `SCHEMA_VERSION=3`（移除 reward_policy / pool 两段，root 即 Init node）。
- [x] 单测：`test_init_node` / `test_input_corpus` / `test_evaluation_strategy` /
      `test_decontamination` / `test_external_data_tools`。
- [ ] **待实测**：ReviewAgent `passed` 门槛的相对化（兄弟分位数 + 灾难下限），当前仍是绝对阈值。
- [ ] **待实测**：Tier-1 代理分与下游真分的相关性（用 review 口径收尾阶段的 `(tier1, dev_score)` 样本对）。
- [ ] **待实测**：外部数据消融与去污染率的联合报告（证明增益不是来自泄露）。

### M9 — 对齐 DataMaster：dev 题集物化 + 训练留痕（2026-07-31）
> 对照调研结论：Init node（都必须过一次 LLM 生成代码）与「训练代码冻结、只搜数据侧」两处，
> 我们本就与 DataMaster 同构，无需改动（详见
> [`docs/design-decisions.md`](./docs/design-decisions.md) §2.3/§2.5）。真正的差距在验证集划分
> 与训练配方留痕，本里程碑补齐。
- [x] **`core/dev_set.py`**：dev 题集按 `(source_sha, seed, size)` 物化成
      `workspace/eval/dev.jsonl` + manifest，`--dev-set` 显式下传给评测脚本，
      评测脚本不再自行抽样。纪律取自 DataMaster，但它只写在 prompt 里、没有代码生成
      `input/val.csv`，缺席时每个节点各切各的；这里落成代码（见 §4.1.1）。
      抽出的题集与旧 `MATH_TRAIN_SAMPLE_SIZE` 逻辑等价，历史分数仍可比（有单测钉住）。
- [x] **dev 集自动并入去污染参考集**：dev 是搜索直接优化的目标，混进合成数据比混 test 更能刷分，
      而此前它只存在于评测子进程内存里、根本无从检测。
- [x] **checkpoint 记 dev 口径**：续跑时对不上直接报错，避免两批不可比的分数混进同一棵树。
- [x] **训练留痕**：`<out_dir>/run_manifest.json` 记命令 / `SFT_*` 等影响分数的环境变量 / 脚本内容摘要。
- [x] **root 即 node_0**：id 计数器从 0 起并在 `TreeManager` 里钉死根编号，
      消除"文档写 node_0、磁盘上却是 node_1"的不一致。
- [x] 单测：`test_dev_set`（口径固定/幂等/换口径报错/与旧抽样等价/dev 泄露可检出）+
      `test_engine` 的 manifest 与 `--dev-set` 转发。
- [ ] **不采纳**：DataMaster 的 Black/Red 节点分型（每次扩展 5 黑 1 红）。我们的搜索空间比它小，
      强行分型可能反而限制变异；先跑起来看数据再定。

### M10 — 外部数据溯源：绕过工具的路径也要可查（2026-07-31）
> 缺口：`.meta.json` 只覆盖走 `download_dataset` 的下载。Agent 内置工具全放开，直接在流水线里
> `load_dataset(...)` / curl 拉数据这条路完全无痕，而这恰恰是它更顺手的写法。去污染（数据层）
> 兜住了泄露风险，兜不住"污染从哪个源进来的"这个审计问题。详见 §3.3。
- [x] **`core/provenance.py`**：`record_tool_uses`（工具调用含参数落 `node_N/agent_tools.jsonl`，
      按 phase 标生成/第几轮自修正）+ `scan_node`（静态扫 `pipeline.py` 与 `operators/**.py`，
      归集 `data_links/` 引用与绕行的 `load_dataset`/`hub_download`/数据类 URL，写
      `node_N/data_provenance.json`，命中直连引用则 `unmanaged: true` 并告警）。
- [x] **按尝试记录**：`run_pipeline_agent` 每次尝试结束都追加，失败那次的下载已落盘、重试不会
      收回，只记最终结果会漏掉它们。
- [x] **成败都扫**：`_evaluate_node` 用 `finally` 收尾，"拉了不该拉的数据集导致扣分"只发生在
      失败节点上，是最需要查来源的场景。
- [x] **`.meta.json` 补 `license` / `url` / `source`**：此前 docstring 承诺记 license 但实际没写；
      Agent 在 inspect 时看到的许可只存在于它的上下文里，落盘的边车才是审计与论文能引的东西。
- [x] **prompt 两段纪律成对**：工具开启时"外部数据一律走这四个工具"；**关闭时**"不得联网拉取
      任何外部数据"——工具没挂载不等于 Agent 没法联网。
- [x] 单测：`test_provenance`（工具日志追加/截断/直连引用检出/LLM URL 不误报/产物目录跳过/
      失败节点仍扫）+ `test_external_data_tools` 的 license 落盘与两段 prompt 纪律。
- [ ] **已知限度**：静态扫描扫不到运行时拼出的 dataset id / URL（f-string、变量拼接）。
      要堵死得靠沙箱断网（与"流水线代码执行安全性"这条风险合并考虑）。

### M11 — 外部数据下载改为全量 + 连接配置收归 YAML（2026-07-31）
> 缺口：`download_dataset` 默认只取前 5000 行，且复用键不含 `max_rows`。详见 §3.2.1。
- [x] **默认下载整个 split**：取多少 / 怎么采样 / 按什么过滤本就是流水线该做的决策（也是被搜索
      的对象），固化在工具里等于替搜索先砍掉一维；且"前 N 行"不是随机样本，有序数据集头部有偏。
      只取单个 split 而非整仓库——整仓库会把 `test` split 一起拖进来。
- [x] **上限改字节**（`dataset_search.max_download_mb`）：行大小跨数据集差几个数量级，行数上限
      对"会不会拉爆磁盘"几乎没约束力。流式边读边写，内存与数据集大小无关。触顶截断而非报错，
      meta 记 `complete: false` / `truncated_by`，并说明截断留下的是头部、不是随机样本。
- [x] **原子落盘**：先写 `.part` 再 `os.replace`。此前中途被打断会留下半份文件，下个节点
      `exists()` 照样命中，把截断数据当完整的用——不报错也查不出来。
- [x] **复用提示说清完整性**：命中缓存时明确回答这份文件完整与否、多少行。此前
      node_3 拉 5000 行、node_7 想拉 50000 行会静默拿回 5000 行，让"加数据没用"成为错误结论。
- [x] **新增 `dataset_search` 配置段**（endpoint / token / cache_dir / max_download_mb）：
      dataset-search 是 in-process MCP，读不到 `agent_sdk.env`（那是给 SDK 子进程的），
      连接信息只能走这里；且必须在启动时设，`huggingface_hub` import 期就把 `HF_ENDPOINT`
      读成模块常量（§3.2.2）。
- [x] 单测：全量/字节上限/显式 max_rows/中断不留可复用文件/复用提示标不完整/配置进环境。
- [ ] **待实测**：大数据集（>2GB）流式下载的实际耗时，确认不会吃掉搜索的时间预算。

### M12 — 第二个 hub + 发现渠道 + 补上溯源盲区（2026-07-31）
> 缺口有两层。一是**能力**：取数只有 HuggingFace 一个源，国内直连不通时无路可走；Hub 关键词搜索
> 不支持语义匹配，"不知道数据集叫什么"这一步只能靠 SDK 内置 WebSearch 返回的自由文本。二是
> **溯源盲区**：`FileStorage` 支持 `hf:` / `ms:` 前缀直接拉远程数据，而 M10 的扫描器三条正则一个
> 都不命中——最顺手、最可能被走的那条路，恰恰是漏掉的那条。详见 §3.2.0、§3.3。
- [x] **扫描器补 `storage_uri`**：`hf:owner/name:config:split` / `ms:owner/name:split` 计入
      `unmanaged`。这个写法是我们自己的 `dataflow-dev` 技能 knowledge base 教的，一个字符串就完事，
      比 `load_dataset` 更顺手却完全不长得像"下载"；漏掉时 `unmanaged` 会安静地保持 false。
- [x] **prompt 成对覆盖**：开启时"先 download 落盘再填绝对路径，别用 `hf:`/`ms:` 远程写法"；
      关闭时那条禁令也一并列出 `hf:` / `ms:`。
- [x] **ModelScope 工具组**（`search_ms_datasets` / `preview_ms_dataset` / `download_ms_dataset`）：
      理由是国内网络下 HF 常直连不通、中文语料覆盖更好，不是"多一个源"。落盘实现与 HF 完全共用
      （`_download(hub, args)`），只在 `_row_iter()` 分叉——字节上限/原子落盘/溯源字段/复用提示
      任一 hub 漏掉都是后门。缓存键加 hub 短码：同名数据集在两处未必同一份内容。
- [x] **检索走 SDK `HubApi.list_repos`**：最初手写的 `PUT /api/v1/dolphin/datasets`（社区流传的
      旧端点）实测 404。SDK 一次就带回 license/tags/tasks/downloads/gated，等于 HF 侧 search +
      inspect 两步，所以 ModelScope 组只有三个工具。
- [x] **拒绝脚本型数据集**：ModelScope 上 legacy script dataset 不少（`modelscope/gsm8k` 即是），
      加载要 `trust_remote_code=True` = 在编排器进程里跑仓库里的任意代码。不开，改为提示换一份
      parquet/jsonl 镜像仓库——同一份语料通常有多个 owner。
- [x] **`llm/mcp_discovery.py`**：`search_web`（Serper）/ `search_github` / `search_papers`（arXiv）。
      **只发现不落盘**，返回统一带"回 download_dataset / download_ms_dataset 取数"的指引。
      凭据缺失按**工具**降级而非整台 server 罢工——少配一个 key 不该让另两个也用不了。
- [x] **配置段**：`dataset_search.modelscope_endpoint`、`discovery.{serper_api_key,github_token}`、
      `agent_sdk.discovery` 单独开关。
- [x] 单测：`test_discovery_tools`（凭据降级/匿名 GitHub/结果归一化/arXiv Atom 解析不引新依赖/
      发现工具一律指回下载工具）+ ModelScope 契约一致、两 hub 缓存键不串、SDK 缺失可读报错、
      检索结构变形容忍 + `test_provenance` 的 `storage_uri` 检出与本地路径不误报。
- [x] **可裁剪**：`data-discovery` 相比 SDK 内置 WebSearch/WebFetch 的增量有限（结构化结果 + 统一
      下一步指引）。若跑下来 Agent 更爱用内置工具，`agent_sdk.discovery: false` 直接关掉。
- [ ] **待实测**：`search_web` 未用真实 Serper key 打通（付费，环境里没有 key）；除此之外
      ModelScope 搜索/预览/下载、HF 镜像下载、GitHub、arXiv 均已线上实测通过。

### M13 — 实测暴露的三个 bug（2026-08-02）
> 都是"单测全绿但线上必挂"的类型，共同根因是**单测把 SDK 和网络都 stub 掉了**，测的是自己的封装
> 契约，恰好把"我们调得对不对"这层挡在外面。详见 §3.2.1.1。
- [x] **`search_datasets` 每次必挂**：`huggingface_hub` 1.x 移除了 `list_datasets(direction=...)`，
      我们还在传，线上稳定 `TypeError`。
- [x] **`search_papers` 每次必挂**（受限网络下）：arXiv 官方示例给的是 `http://`，明文 http 在本
      环境被直接挡掉（返回 000），https 通。改 https。
- [x] **`search_ms_datasets` 序列化崩溃**：SDK 的 `last_modified` 是 `datetime`，`json.dumps` 直接
      抛 TypeError——一次本来成功的搜索变成工具报错。`_ok` 统一加 `default=str`。
- [x] **补签名契约测试**：不打网络，拿真实已安装 SDK 的签名 `bind()` 一次参数，覆盖 HF 的
      `list_datasets`/`dataset_info`、`load_dataset`、ModelScope 的 `list_repos`/`MsDataset.load`。
      毫秒级，正好补上 stub 留下的洞；SDK 未装则 `importorskip` 跳过。
- [x] **`pyproject` 新增 `external-data` 组**：刻意不写 `modelscope[datasets]`——该 extra 把
      `datasets` 钉在 `<=4.8.4`，会降级 HF 侧与 vendored DataFlow 用的 5.x。`addict` /
      `simplejson` / `sortedcontainers` 是实测跑通 `MsDataset` 流式加载的最小补集。

### M14 — 训练集切齐：堵上"数据量买算力"（2026-08-02）
> 缺口：§2.5 冻结了训练**代码**，但没冻结训练**预算**。总步数 `epochs × N / global_batch` 里
> `N` 由节点自己决定，多产数据 = 白拿梯度步数，奖励归因失效；而 MCTS 会如实学走这条捷径。
> `DF_TARGET_SAMPLES` 只是给 pipeline 的建议值（可以不读、过滤会少产、增强会多产）。详见 §2.5.1。
- [x] **`core/train_set.py`**：进训练脚本前定量物化到 `cap` 条，确定性抽样、保序；未超量则原样
      透传不做无谓复制，manifest 照写（差额本身是要留痕的信息）。
- [x] **`CappedTrainer` 包在 `self.trainer` 外**：搜索期打分与终选 Tier-2 是两条独立的训练调用
      路径，包一层则"凡是训练都受约束"由构造处一次保证——在调用点各切一次，漏掉任一条都不报错。
- [x] **产出不足只记录不修补**：欠训是同一个问题的另一半，切不出数据来。落 `short` 字段 + 告警，
      让它可见而不是安静地进分数。
- [x] **`core/rows.py`**：dev 题集与训练集共用读行/定量抽样两个原语。各写一份必然漂移（种子、
      保序、不足时的行为），而这类差异不报错。
- [x] **`generation.enforce_target_samples`（默认 true）**：关掉 = 显式把数据量放回搜索空间，
      做"更多数据是否更好"的消融时才用。
- [ ] **待实测**：有 GPU 后确认切齐后各节点的实际训练步数一致（查 `train_set.manifest.json`
      与 `run_manifest.json` 对齐）。

### M15 — 产出 schema 收归配置 + 评测数据出库（2026-08-02）
> 缺口：下游要什么字段，原先由 TaskProfiler 让 LLM 现编，配置里没有锚点；而真正的契约在训练
> 脚本的取字段逻辑里，两份脚本的别名表还不一样。编出 `{problem, solution}`（很合理）→ 流水线
> 照做 → 评审拿同一份编出来的 schema 打分、满分放行 → MATH 脚本一条读不到、抛错 → 记成
> 「训练脚本非零退出」。GPU 烧完才暴露，归因还是错的。详见 §3.1.1。
- [x] **`trainer.dataset_schema` 为权威**：写进 Profiler prompt（模型的 quality_criteria 也围绕
      真字段展开）、解析后覆盖、续跑时覆盖 checkpoint 里的旧 schema。三处消费点（生成 prompt /
      评审 prompt / 硬指标 required）本来就读 `profile.target_schema`，自动跟着走。
- [x] **评审硬门槛**：空缺率 ≥ `review.null_threshold`（0.5）判不通过，issues 里点名要求的字段。
      纯代码、不花 GPU 不花 LLM，自修正循环拿着就能改对；个别行缺字段仍只按比例扣 tier1。
- [x] **评测数据出库**：`data/math_train.jsonl`（6.7M）与死文件 `data/seed.jsonl` 删除，
      `data/README.md` 写清三种数据角色与获取方式，`.gitignore` 收口，两处 FileNotFoundError
      改为指向说明（原先还提了一个根本没实现的 `MATH_TRAIN_PATH` 环境变量）。
- [x] 清掉 `tests/test_final_selection.py` 里 seed 时代的 `seed_path` 残留（位置传参使它实际
      接到的是 `sample_text`，名字已经是错的）。
- [x] **PTB 的 dev 口径查清并留档（决定维持现状）**：先前记的「每个节点各跑各的 `--dev-limit 150`」
      不成立——`inspect_ai` 的 `--limit N` 走 `slice(0, N)`，是确定性前缀，几次 Tier-2 评的是同一批题；
      且 PTB 用 `reward_source: review`，官方评测只在终选跑 `tier2_top_k` 次，搜索期根本不碰 benchmark。
      真实问题只剩 **dev ⊂ test**（官方没有独立 dev 池，`--limit 150` 是 `--limit -1` 的前缀），
      影响面限于「3 选 1」这一步，后果是终评分偏乐观。已写进 `docs/posttrainbench_adaptation.md` §6.1、
      PTB 配置注释与终选/终评摘要输出；两种彻底不相交的改法及其代价一并记在 §6.1。
      `trainer.dev_set` 对 PTB 不适用：官方 `evaluate.py` 写死读 benchmark 自己的题池，接不了 `--dev-set`。
- [x] **`data/benchmarks/` 决定留库**：2.3M 的终评集是量具不是语料——入库才能保证每次跑分评在
      逐字节相同的题上，且 `load_benchmark_rows` 本就按固定相对路径找它、缺失即抛错。与
      `math_train.jsonl`（可再生的训练语料）不同类，不出库。

### M16 — 死代码清理（2026-08-02）
> 旧 LIMO 判分迁移后，统一 scorer 接管了打分口径，各 benchmark 模块
> 里的老判分函数没人调用却还在；配置里也留了一个不会被读取的开关。
- [x] **删 5 个 `score_predictions`**（gsm8k / math500 / math_train / aime2024 / amc23）：
      `__init__.py` 只导出 loader 与 prompt，判分函数从包外不可达，确认无动态派发后连同变成
      死引用的 import 一并删除。
- [x] **删 `agent_sdk.skills`**（两份配置）：代码从不读它，技能实际由 `setting_sources=[project]`
      自动发现。一个长得像开关、改了却没有任何效果的配置键是陷阱，信息并入上一行注释。
- [x] **删 `PipelineNode.is_root` 与 `ContaminationReport.clean`**：两个无人调用的属性。
- [x] **删三份评测过程笔记**（`docs/eval_{comparison_limo,modification_report,review_checklist}.md`）：
      迁移期的临时记录，结论已进 `design-decisions.md`，彼此互引之外无人引用。
- [x] **修 GPQA 终评判分（清理时发现的真 bug）**：`_normalize_row` 给 MCQ 存的是 `gold_answer`=
      选项**原文**、`gold_letter`=字母，而 `_score_benchmark` 拿抽出的字母去比 `gold_answer`，
      永远不等——GPQA 在终评里稳定报接近 0 分，与模型好坏无关。新增 `_grade`：带 `gold_letter`
      的题比字母、抽不出字母才退回数学等价比较，bad case 的 ground_truth 也改记字母。
      顺带去掉 `extract_answer(p, bench_name)` 这个误导写法（bench 名全部落进 auto 分支）。
- [x] **删 `additional_benchmarks.py` 的 10 个判分函数**：`_grade` 接管 MCQ 之后它们才真正变成
      死代码（此前 `_mcq_score` 是"没接线的正确实现"，不能当死代码删）。模块降为纯 loader。
- [x] **测试改测线上路径**：原 `test_project_gpqa_..._scores_letter_answer` 测的是没接线的
      `gpqa_score`，bug 正是这么活下来的。改为直接调统一 scorer，并把数学等价判分器
      换成"被调到就断言失败"的替身：选择题本就不该走 sympy 判分，被调到即说明分支走错；替身同时
      绕开了 `grader` 在 `df-agent` 里装不了的问题（见 §7）。

### M17 — README 翻修 + 一个被默认值掩盖的 dev 口径 bug（2026-08-02）
> README 是很早期的文档，多处描述已与实现脱节。核对时顺带发现 dev 物化并没有真正落地到脚本侧。
- [x] **修 `load_math_train` 对物化 dev 集二次抽样**：框架用 `--dev-set` 下传题集后，脚本仍按
      `MATH_TRAIN_SAMPLE_SIZE` 再抽一次。两边默认都是 500，`len > 500` 不成立，所以从未触发；
      把 `dev_set.size` 改成 800 就会静默评 500 条，而 `dev.manifest.json` 记的是 800——物化
      承诺的"所有节点评在同一批题上"退化成"恰好还成立"。改为 `data_path` 非空即不抽样，
      并补回归测试（用 `_SAMPLE_SIZE + 100` 构造，避免再被默认值相等掩盖）。
- [x] **澄清 `review` 的五维与两个门槛**：`schema_threshold` / `relevance_threshold` 常被误读成
      第 6、7 个维度。实际是同一批五维分的两种用法——加权成 `tier1_score`（连续，MCTS reward）
      与卡 `passed`（布尔，决定是否自修正/是否进训练）。`default.yaml` 的 review 段按这两组重排
      并加了说明。
- [x] **README 翻修**：删与"安装"章重复的 SFT/vLLM 安装段；删失效的 `HF_CACHE_DIR`（定义未使用）
      与 `BENCHMARK_ROOT`（根本不是环境变量）；`timeout_sec` 1h→2h 对齐配置；benchmark 表补真实
      题数、判分列改为说明统一的 `_grade`；`data/` 一节改写为 benchmarks 入库 / dev 源不入库；
      "ReviewAgent 只查表面 schema/相关性"改为五维+硬指标；运行示例里的机器专有路径与 IP 换成占位符；
      补 PTB 链路指引、`PTB_DECONTAM_REF` 必须随 `PTB_BENCHMARK` 同步切换的警告、目录结构补齐。
- [x] **删 PTB 配置里"external_data：关"的过期注释**：上一轮已改成开启。

### M18 — 评测脚本彻底交出「评哪些题」的决策权（2026-08-03）
> M17 的修法只是让脚本侧抽样"给 `--dev-set` 让路"，平行的档位还在，退路也还在。既然口径归框架，
> 就不该在脚本里留第二套机制——留着就迟早会被走回去，而且每次都不报错。
- [x] **`load_math_train` 降为纯 loader**：`data_path` 改必填，删掉 `_SAMPLE_SIZE` /
      `_SAMPLE_SEED` / `MATH_TRAIN_SAMPLE_SIZE` 与内置默认题库路径。读进来多少条就返回多少条。
- [x] **`--mode dev` 缺 `--dev-set` 直接报错**：旧自定义评测脚本的 loader 与 SFT 入口两处都拦。
      SFT 入口的校验放在 **SFT 之前**——这条链路
      先训数小时再评测，等评测阶段才报错就白烧一轮。
      代价：`dev_set.source` 留空时数学链路跑不了 dev（旧行为是脚本自己挑 500 条）。这是想要的。
- [x] **README 补「脚本负责什么、不负责什么」**：把职责边界从散落的注释提炼成两张表
      （该做 5 条 / 不该做 5 条），并说明每条反模式为什么会毁掉节点间的可比性。
- [x] **同步措辞**：`default.yaml`、`dev_set.py` 的报错、`design-decisions.md` §4.1、
      配置速查表、TODO §5.6 契约里"留空=退回脚本自行抽样"的说法全部改掉——留空只是框架不物化，
      不构成"脚本可以自己挑题"的许可。

### M20 — 输入范式规则把外部取数堵死（2026-08-03）
> 症状：开了 `agent_sdk.external_data`、`input/` 又是空的，Agent 仍然只会凭模型知识合成，
> 几乎不去 Hub 取数。工具挂了、prompt 里也写了，但从没被用起来。
- [x] **定位**：范式规则是**编号硬要求**（`PIPELINE_AGENT_PROMPT` 第 3 条），scratch 档写死
      「流水线不依赖任何输入」「`first_entry_file_name` **必须**指向那个空 jsonl」「凭模型知识
      合成」；而外部数据说明排在编号要求之后、自称「（可选）」，还说下载的文件可以「作为
      FileStorage 的 first_entry」。两段在 `first_entry` 上直接冲突，强度却不对等——
      一边是编号"必须"，一边是靠后的"可选"，Agent 只会听前者。
      grounded 档同理偏严：「不要绕开它另起一套」会连带抑制"本地语料 + 外部补充"。
- [x] **修法**：范式判定从「只看 `input/`」改成 `(input/ 有无语料) × (取数工具开没开)` 四种
      组合。新增 `SCRATCH_WITH_EXTERNAL_PARADIGM_RULE`（摆出取真实数据/凭知识生成/混合三条
      路，明确入口可指向 `download_dataset` 返回的绝对路径，并要求在 decision.json 的 reason
      里写明选型理由）与 `GROUNDED_WITH_EXTERNAL_PARADIGM_RULE`（外部数据只作补充，
      「补充不等于替换」）。`data_links_dir` 非空即等价于工具已挂载，直接用作开关。
- [x] **连带**：`field_flow` 模板里"无输入语料时写 Empty"改为区分"纯生成"与"以外部数据集为
      入口"两种情形。README「输入数据」一节补一段说明这张表只讲本地输入。
- [x] **回归**：新增两个 prompt 渲染测试，分别钉住 scratch+external 不再出现"流水线不依赖任何
      输入"、grounded+external 保留"补充不等于替换"。

### M19 — 单机化 + scripts 死代码清理（2026-08-03）
> 不再训 32B，双机 16 卡那套没有使用场景了。留着的代价不只是几百行代码，还有每个读到
> `dispatch` / `rank0` / `NNODES` 的人都要先判断"这条链路到底走不走多机"。
- [x] **拆掉双机调度**：删 `SSHTrainerWrapper`（128 行，与 `DownstreamTrainerWrapper` 的差别
      只是多注入一组集群环境变量）、`main.py` 的 `dispatch` 分支、两份配置里的 `dispatch` 与
      `trainer.ssh` 块。`_ENV_PREFIXES` 去掉 `NNODES`/`NODE_RANK`/`MASTER_`（run_manifest 不必
      再记录不存在的量）。
- [x] **拆掉训练脚本的远端逻辑**：删 `_build_remote_worker_ssh_cmd` / `_launch_remote_worker` /
      `_stop_remote_worker` / `_wait_remote_worker` / `_forward_remote_env` / `_should_launch_remote_worker`
      与 `_node_rank` 等 6 个 env 读取器；连同**跨机 NCCL 的 RDMA 网卡推断**
      （`_detect_nccl_ib_hca_and_gid_index` 等 3 个函数，单机走 NVLink/PCIe 用不上）。
      `run_sft` / `main` 里的 rank 分支拉直，日志从 `sft_train_rank{rank}.log` 改回 `sft_train.log`。
      脚本从 942 行降到 ~630 行。
- [x] **默认值改单机**：`SFT_NUM_GPUS` 16 → 8。`SFT_BASE_MODEL` 仍是 32B 路径但已在 README
      标注"换基座最该改的一项"，并列出连带要改的 `SFT_CUTOFF_LEN` / `EVAL_MAX_MODEL_LEN` /
      `EVAL_TP_SIZE`（LIMO 的 lr 与 epochs 是针对 32B + 800 条调的，换小模型要重调）。
- [x] **补回超时测试**：SSH 那四个测试里只有超时降级是 `DownstreamTrainerWrapper` 没覆盖的，
      移植成 `test_trainer_times_out_and_degrades`，其余三个删除。
- [x] **scripts 死代码清理**：删旧数学归一化模块（唯一引用是
      `grader.py` 里一个从未被调用的 import）；删 `grader.py` 的 `timeout_handler` /
      `math_equal_process` / `math_equal_simple` / `check_is_correct_simple` / `_test_math_equal`
      与随之失效的 4 个 import；删 `__init__.py` 里零外部引用的 `extract_gsm8k_answer`、
      `SYSTEM_PROMPT` 导出（前者是 `extract_answer` 的内部实现，后者与旧自定义评测脚本里的同名硬编码重复）。
      `reference_train_eval.py` 与 `ptb_benchmarks.list_benchmarks` 经核查**不是**死代码：
      前者被 `test_engine.py` 真实 subprocess 调用并作为 I/O 契约示例，后者用在未知 benchmark
      的报错文案里（`get_benchmark` 抛 KeyError 时枚举可选值）。
- [x] **`reference_train_eval.py` 补齐到现契约**：它此前只认 `--dataset/--output`，而框架在
      `trainer.dev_set.source` 非空时会自动追加 `--dev-set`——`argparse` 直接 exit 2，谁把
      `dev_eval_cmd` 指过来每个节点都会判 failed。没人踩到只是因为它不在任何 YAML 里。
      更深的问题是它的 `score` 是训练数据自身的「answer 非空比例」，而 README 已把脚本职责
      写成「读 `--dev-set` 评测，整份评完」：一个自称契约示例却演示错形状的脚本，比死代码更
      容易误导人。现改为查表式"训练"（`--dataset` 的问答对进字典）+ 在 `--dev-set` 上逐题
      作答算 `exact_match`，`--mode dev` 缺 `--dev-set` / `--mode test` 缺 `--test-set` 均
      非零退出。`test_event_loop` 顺带配上 `trainer.dev_set`，让物化→下传这条链路进冒烟测试。

---

## 7. 风险与开放问题
- **判分路径在 df-agent 内不可单测**：`grader.py` 依赖 `latex2sympy2==1.9.1`（仅在 `df-eval`
  extra 里），它把 `antlr4-python3-runtime` 钉在 4.7.2，与 `df-agent` 侧 DataFlow 算子所用
  `math-verify`（经 `latex2sympy2_extended` 要求 ≥4.9.3）互斥。三环境隔离下这不是缺陷，是
  设计使然——评测本就只在 `df-eval` 跑。但副作用是编排环境里 import 不了 `grader`，
  `_score_benchmark` 因此长期没有单测，M16 的 GPQA 判分错误正是这么活下来的。现用替身模块
  覆盖不经过 sympy 的分支；需要覆盖数学等价判定时，得把用例放到 `df-eval` 下跑。
- **Agent SDK 环境就绪**：`claude-agent-sdk` 需可安装、`ANTHROPIC_API_KEY` 配好、自带 CLI 可运行；本环境外网被屏蔽，需确认 SDK 运行时不依赖被拦截的域名。
- **技能发现可靠性**：需实测 `setting_sources=["project"]` + `_agent_root/.claude/skills/` 能让 SDK 正确发现并调用三个技能；并确认无头模式下 `dataflow-dev` 不会触发交互式 / git-gh 子流程。
- **DataFlow 源码 copy 的维护**：`_agent_root/dataflow/` 是上游 DataFlow 的拷贝，会随上游更新而变旧；需在 README 记录拷贝时的版本，需要时手动重新拷贝同步。
- **DataFlow 包安装与算子可用性**：实现 M0 时需确认本地 DataFlow 能否 import 及其算子注册是否正常。
- **训练脚本的真实存在性**：当前为可插拔契约；真正跑 Phase-2 需要用户提供符合契约的训练脚本（或先用 `reference_train_eval.py` 验证管路）。
- **流水线代码执行的安全性**：Agent 生成代码在子进程中执行，需限定工作目录与资源；后续可考虑沙箱。
- **成本/时延**：每次扩展都启动一次 Agent SDK 运行（多轮工具调用），比单次 API 调用更慢更贵；需配 `max_turns`/预算上限。
- **外部数据引入的泄露风险**：Hub 上大量数据集本身是 benchmark 的镜像或衍生；`external_data` 开启时必须同时配好去污染参考集，否则搜索会奔着"找到含测试题的数据集"去。溯源侧见 §3.3（`data_provenance.json` 的 `unmanaged` 标记）；静态扫描对运行时拼接的来源无能为力，真要堵死需与下条的沙箱一并解决。
- **起点 / 奖励 / 泄露 / 通用性调参**：相关设计决策（Init node 作为唯一起点与参照、奖励即节点自己的分、dev/test 隔离 + 去污染硬指标、尺度无关化超参）已统一收敛到 [`docs/design-decisions.md`](./docs/design-decisions.md)，此处不再重复。

---

> 下一步：M6 健壮性 / M7–M8 的"待实测"项，需真实 dev 训练脚本就绪后推进。本文档随实现进度更新勾选状态。
