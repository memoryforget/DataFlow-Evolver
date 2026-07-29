# PostTrainBench 本地适配（feat/posttrainbench-adapt）

把 DataFlow-Evolver 从「每节点全量 SFT+评测驱动 MCTS」的数学路径，扩展成能在**本机单卡**上按
PostTrainBench 的评测口径，对给定 base model 做 post-training 并最大化任一 benchmark 分数的
**两段式（Tier-1 / Tier-2）框架**。所有改动**additive、走配置开关**，默认 `search.reward_source: train`
时行为与原数学 SFT 路径完全一致。

---

## 1. 核心机制：两段式反馈

| | Tier-1（搜索期，每节点） | Tier-2（终选，仅 top-K） | 终评（1 次） |
|---|---|---|---|
| 触发 | MCTS 每次扩展 | 搜索结束按 tier1 取 top-K | 选出 final_model 后 |
| 产出规模 | `generation.probe_samples`（200） | `generation.target_samples`（4000） | — |
| 动作 | ReviewAgent 五维复合分 | 全量 SFT + 官方 `evaluate.py --limit dev` | `evaluate.py --limit -1` |
| GPU | 不占（纯 LLM 打分 + 代码扫描） | 单卡 1×H100 | 单卡 |
| reward/输出 | `tier1_score` 回传 MCTS | 真分选最优 → `final_model/` | 最终报告分 |

**为什么**：每节点全量 SFT 太贵、MCTS 搜不深；且从官方评测抽 per-sample bad_cases 会把 test 题面/答案带进
合成回路（污染 + 分数失真）。故搜索期用便宜、不接触 benchmark 的 Tier-1 打分器驱动，只在终选对少数
top-K 花真实训练/评测预算。

### Tier-1 = 升级版 ReviewAgent（1 次 LLM + 纯代码硬指标）
- **五维 LLM-judge**（各 0–1）：`correctness / relevance / difficulty / diversity / schema`。
- **固定权重复合分**：`llm_composite = 0.35·correctness + 0.20·relevance + 0.15·difficulty + 0.15·diversity + 0.15·schema`。
- **硬指标惩罚**（扫全份文件）：`tier1 = llm_composite · (1-dup_rate) · (1-null_rate) · parse_valid_rate`。
- **随机蓄水池采样**（替换旧的「取前 N 行」）：一趟遍历、全份等概率、`sample_size=80`。
- **F1–F7 失败信号**（借鉴 ANDES 坏味道分类）：评审在打分之外标注命中的信号编号（F1 描述过窄 / F2 华而不实 /
  F3 模板锁定 / F4 覆盖塌陷 / F5 约束漂移 / F6 有形无实 / F7 描述空泛），规整去重后存入 `ReviewResult.failure_signals`，
  并以 `[Fx] 描述` 形式并入 `issues` 供 PipelineAgent 自纠 prompt 复用。**仅观测/可解释，不改变 `tier1_score` 打分公式。**
- 合规：只看**自己合成**的数据（correctness 用自验），不碰任何 benchmark test。

### Tier-2 + 终评
- 按 `tier1` 取 **top-K（`search.tier2_top_k`，默认 3）**；各自用 `target_samples` 重跑**同一** `pipeline.py`
  （经 `DF_TARGET_SAMPLES` env，无需重新调 Agent SDK 生成代码）。
- **安全阀**：放大重产后重跑 ReviewAgent + 硬指标；若质量塌陷（大面积重复 / null 激增 / 不可解析 / tier1 大幅回退）
  则跳过用次优（backfill）。
- 存活候选跑 `dev_eval_cmd`（单卡全量 SFT + 官方 `evaluate.py --limit dev-limit`，**只读聚合 accuracy**），
  取真分最高者 → `workspace/final_model`（符号链接到其 `sft_model/`）。
- **终评**：对 `final_model` 跑一次官方 `evaluate.py --limit -1`（复用桥接 `--mode test`，只评不训），得最终报告分。
- **预算门控**：`search.time_budget_sec` 预留「K 次 Tier-2 + 终评」时间后停止扩展；预算不足自动缩小 K（极端 K=1）。

---

## 2. 改了哪些文件

### 修改（additive，默认行为不变）
- `dataflow_evolver/core/schemas.py`：`ReviewResult` 新增 `correctness/difficulty/diversity_score`、硬指标
  `dup_rate/null_rate/parse_valid_rate`、`llm_composite`、`tier1_score`（全部带默认值）。
- `dataflow_evolver/agents/review_agent.py`：随机蓄水池 `_sample_jsonl`、全文件硬指标扫描 `_scan_hard_metrics`、
  五维复合分 + 惩罚 `tier1_score`、读 `review.weights` / `sample_size(80)` / `hard_metrics`，保留 `passed` 门槛。
- `dataflow_evolver/llm/prompts.py`：`REVIEW_PROMPT` 增 3 维；`build_pipeline_prompt` 的规模段改为指示算子
  **从 `os.environ['DF_TARGET_SAMPLES']` 读条数**（不写死）。
- `dataflow_evolver/engine/execution_wrapper.py`：`run()` 增可选 `env_overrides`（注入 `DF_TARGET_SAMPLES`）。
- `dataflow_evolver/main.py`：`search.reward_source` 开关；review 模式 reward=`tier1_score` 且搜索期不训练；
  搜索执行注入探针规模；`search.time_budget_sec` 预算终止；`_final_selection`（top-K 重产@4000 + 安全阀 +
  Tier-2 + `final_model` + 终评）。
- `configs/default.yaml`：仅新增新键的**文档注释**（数学默认值不变）。

### 新增
- `scripts/ptb_benchmarks.py`：7 个 benchmark 的可插拔映射（派生 `--task` 文案、dev-limit、eval 参数画像、
  答案格式 / system prompt、`format_requirement` 三档结构约束）。
- `scripts/posttrainbench_sft_eval.py`：Tier-2 桥接（单卡全量 SFT + 官方 `evaluate.py`；`dev/test` 模式；
  `--base-model/--ptb-task-dir/--dev-limit/--benchmark`；格式对齐；`results.json` 契约）。
- `configs/posttrainbench_gsm8k.yaml`：env 参数化配置，runner 可复用到所有 benchmark。
- `scripts/run_posttrainbench_local.sh`：`<benchmark> <base_model>` 一键起 `main.py`（含终评）。
- `tests/`：`test_review_agent_tier1.py` / `test_generation_two_tier.py` / `test_orchestrator_reward_tier1.py` /
  `test_final_selection.py` / `test_time_budget.py` / `test_posttrainbench_bridge.py`（+ 辅助 `_ptb_stubs.py`）。
- `docs/posttrainbench_adaptation.md`（本文）。

---

## 3. 7 个 benchmark 映射（`scripts/ptb_benchmarks.py`）

| benchmark | 家族 | dev-limit | eval token 参数 | 需 judge | format_requirement | 训练格式要点 |
|---|---|---|---|---|---|---|
| gsm8k | inspect | 150 | `--max-tokens 4000` | 否 | unstructured | 末行 `ANSWER: <数值>` |
| aime2025 | inspect | -1(全 30) | `--max-tokens 16000` | 否 | unstructured | `\boxed{整数0-999}` |
| gpqamain | inspect | 50 | `--max-tokens 16000` | 否 | unstructured | 末行 `ANSWER: <字母>` |
| bfcl | inspect | 150 | `--max-tokens 16000` | 否 | tool_call | 单个工具调用（tools 字段） |
| humaneval | inspect | 150 | `--max-tokens 4000` | 否 | code | 完整可运行 Python 实现 |
| healthbench | server | 32 | `--max-new-tokens 16384` | 是(OPENAI_API_KEY) | unstructured | 安全准确、覆盖 rubric |
| arenahardwriting | server | 32 | `--max-new-tokens 16384` | 是(OPENAI_API_KEY) | unstructured | 结构清晰的高质量写作 |

- **inspect 家族**（thin wrapper over `inspect_ai`）：`model=vllm/<model_path>`，`--max-tokens`，`--json-output-file → accuracy`。
- **server 家族**：脚本自起 `vllm serve` + GPT judge，用 `--max-new-tokens` / `--judge-workers`，需 `OPENAI_API_KEY`。
- `score_key` 统一取 `accuracy`（榜单口径）；`limit=-1` 时不传 `--limit`（= 全量）。

### `format_requirement` 三档（借鉴 ANDES，内容级结构约束）
参考 [ANDES](../../ANDES) 的数据合成设计：在**训练 schema（我们仍用 ShareGPT，不改动）之外**，用一个正交的
`format_requirement` 枚举控制**答案文本本身的结构**，取三档（`FORMAT_REQUIREMENTS` in `ptb_benchmarks.py`）：
- `unstructured`：不加结构约束，答案格式仍由 `answer_format` / system prompt 管（gsm8k/aime/gpqa/health/arena）。
- `code`：答案必须是完整可运行代码、包在 ```` ```python … ``` ```` fence 中、前后无多余文字（humaneval）。
- `tool_call`：答案必须是合法工具调用 JSON、包在 ```` ```json … ``` ```` fence 中、前后无多余文字（bfcl）。

落地两处（`unstructured` 档均为 no-op，向后兼容）：
1. **合成端**：runner 用 `ptb.composed_task_text(name)` 把结构约束以「【输出格式要求】…」追加进派生 `--task`，
   经 TaskProfilerAgent → PipelineAgent 引导合成 output 的结构。
2. **训练端**：桥接 `convert_dataset` 用 `ptb.format_instruction_for(name)` 把同一约束追加进 SFT system prompt，
   训练时强化该结构，与评测端解析口径一致。

### 格式对齐（硬约束）
评测端按模型族用固定 `templates/{qwen3,gemma3,smollm}.jinja` 格式化推理输入。Tier-2 训练时桥接按 base model
判族（`qwen/gemma/smollm`）选对应 **LlamaFactory 模板**（`qwen/gemma/smollm`）渲染 SFT 数据，与评测 chat 模板一致；
system prompt / 最终答案格式按 benchmark 对齐（见映射表）。可用 `SFT_TEMPLATE` env 覆盖。

---

## 4. 本地环境准备

三个 conda 环境（已就绪；`orchestrator` 从 `df-agent` 启动）：
- **df-agent**（跑 `main.py`）：claude-agent-sdk、openai、本框架（`pip install -e .`）+ 其运行依赖
  （本适配补装了 vendored dataflow 运行所需的 `colorlog` / `rich`）。
- **df-train**（Tier-2 全量 SFT）：LlaMA-Factory、torch、transformers、deepspeed —— 已有。
- **df-eval**（跑官方 `evaluate.py`）：本适配**唯一新依赖大头**，已安装官方 pin：

```bash
uv pip install --python <df-eval-python> \
  inspect-ai==0.3.150 \
  "inspect_evals @ git+https://github.com/UKGovernmentBEIS/inspect_evals.git@06001a83e6d7c709c2ede0570dce7f1031a0bad8"
```

config 中 `trainer.train_python → df-train`，`trainer.eval_python → df-eval`。

---

## 5. 如何跑 GSM8K demo（含终评）

一键 runner（推荐）：

```bash
export DF_API_KEY=...            # 辅助 Agent(gpt-5.4)/算子(gpt-4o) 的密钥
# 快速试：把预算调小（否则默认 ~8h）
export PTB_TIME_BUDGET_SEC=1200
scripts/run_posttrainbench_local.sh gsm8k /workspace/xiaqifeng/models/Qwen3-1.7B
```

runner 会：查表得派生 `--task` 与 dev-limit → export `PTB_*` → 用 df-agent 起
`python -m dataflow_evolver.main --config configs/posttrainbench_gsm8k.yaml --task "<派生文案>"`。
`main.py` 内部完成 **Tier-1 搜索 → Tier-2 top-K 选模 → `final_model/` → 官方 `evaluate.py --limit -1` 终评**，
并在末尾打印 dev / 终评分数。

手动等价：

```bash
export PTB_BASE_MODEL=/workspace/xiaqifeng/models/Qwen3-1.7B
export PTB_TASK_DIR=/workspace/xiaqifeng/PostTrainBench/src/eval/tasks/gsm8k
export PTB_BENCHMARK=gsm8k PTB_DEV_LIMIT=150 PTB_TIME_BUDGET_SEC=1200
/workspace/xiaqifeng/miniconda3/envs/df-agent/bin/python -m dataflow_evolver.main \
  --config configs/posttrainbench_gsm8k.yaml \
  --task "合成 GSM8K 风格的小学数学应用题 SFT 数据（含 CoT、末行 ANSWER: <数值>）以 post-train 该模型"
```

换 benchmark 只需换 runner 第一个参数（`bfcl` / `humaneval` / `aime2025` / `gpqamain` /
`healthbench` / `arenahardwriting`）；healthbench / arenahardwriting 另需 `export OPENAI_API_KEY=...`（judge）。

单测：

```bash
/workspace/xiaqifeng/miniconda3/envs/df-agent/bin/python -m pytest tests/ -q   # 113 passed
```

---

## 6. 合规（防数据泄露）
- 训练路径**绝不**加载任何 benchmark 的 test split；数据全部由本框架自行合成，来源写入 `run.log`。
- 对官方 `evaluate.py` **只评测不训练**、**只读聚合 accuracy**（`--json-output-file`），绝不解析/落盘
  per-sample 的 test 题面或答案；`results.json.bad_cases` 恒为 `[]`。
- Phase-2 DiagnosticAgent 在 review 模式下不触发（无 bad_cases）。

---

## 7. 风险 / TODO
- **4B 单卡全量 SFT 显存**：Qwen3-4B / Gemma-3-4B 全量 SFT 在单卡 80G 上偏紧。已默认开梯度检查点；
  仍不够时设 `SFT_DEEPSPEED=zero2_offload`（ZeRO-2 + 优化器 CPU offload，桥接已内置该配置）。
- **HF 数据集访问**：`inspect_evals` 的 gsm8k/gpqa/aime 等在评测时会从 HuggingFace 拉数据集；离线环境需预置
  HF 缓存或 `HF_HOME`。
- **SmolLM3 模板**：LlamaFactory 若无 `smollm` 模板名，请用 `SFT_TEMPLATE` 指定最接近的 ChatML 模板，或核对
  与 `templates/smollm.jinja` 的一致性（think/no_think）。
- **BFCL 工具调用格式**：当前桥接把 tools 作为 sharegpt `tools` 字段、assistant 输出工具调用文本；与官方
  BFCL 原生 tool-call token 未必逐字一致，需按实际 pass 率微调（映射表 `tool_calling` 分支）。
- **judge 成本 / 密钥**：healthbench / arenahardwriting 需 `OPENAI_API_KEY`；judge 调用有成本，dev-limit 已设小(32)。
- **可选增强**（未实现，留接口）：
  - *Answer-Consistency*：Tier-1 correctness 之外，另做「多次采样答案自洽/多数投票」硬校验。
  - *IFD-on-base*：用 base model 的 IFD / Superfiltering 分数作 difficulty 的定量代理（现为 LLM-judge 估计）。

---

## 8. 验证状态
- `python -m pytest tests/ -q` → **121 passed**（含新增项 + 原有全绿）。
- 覆盖：五维复合分 + 硬指标惩罚数学、F1–F7 失败信号规整/并入 issues、蓄水池采样（确定性 + 非前缀偏置）、
  reward 开关（review=tier1 且搜索期不训练 / train=真分且训练）、两档规模（probe 200 / 重产 via env）、
  终选 top-K + 安全阀 + K 缩减、`results.json` 契约（bad_cases 空、只读聚合）、格式对齐转换（含 format_requirement
  注入）、时间预算终止。
- **未做**：本交付不含真实的端到端搜索 / SFT / vLLM 评测 run（按确认范围为 code-only）。
