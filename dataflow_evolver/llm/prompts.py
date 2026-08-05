"""各 Agent 的 prompt 模板集中管理。

- PipelineAgent 的模板面向 Claude Agent SDK（指示其通过 MCP + 源码阅读 + compile 校验写流水线）。
- Profiler / Review / Diagnostic 的模板面向 OpenAI-compatible API，强制 JSON 输出。
"""
from __future__ import annotations

import json

from dataflow_evolver.core.input_corpus import InputCorpus
from dataflow_evolver.core.recipe_context import RecipeContext
from dataflow_evolver.core.schemas import DiagnosticReport, ReviewResult, TaskProfile, TrainResult

# ---------------------------------------------------------------------------
# TaskProfilerAgent
# ---------------------------------------------------------------------------
PROFILER_PROMPT = """你是一名数据工程任务分析师。根据下面的高层任务描述（及可选样例），\
把它形式化为结构化的数据需求。

任务描述：
{task_description}

{sample_block}{schema_block}
请只输出一个 JSON 对象，字段如下：
{{
  "task_description": "对任务的精炼复述",
  "target_schema": {{"字段名": "类型(str/int/float/list/...)", ...}},
  "quality_criteria": ["高质量数据应满足的要点", ...],
  "downstream_task": "下游训练任务标识，如 math-QA-SFT"
}}
不要输出 JSON 以外的任何内容。"""


def build_profiler_prompt(
    task_description: str,
    sample_text: str | None = None,
    fixed_schema: dict | None = None,
) -> str:
    """`fixed_schema` 非空时，schema 由下游训练脚本决定，不再让模型自行设计字段名。

    其余字段（quality_criteria / downstream_task）仍由模型产出，但会围绕这份 schema 展开——
    所以是写进 prompt 而不是只在解析后覆盖：让模型看见真正的字段名，它给的质量要点才对得上。
    """
    sample_block = ""
    if sample_text:
        sample_block = f"样例数据（节选）：\n{sample_text}\n"
    schema_block = ""
    if fixed_schema:
        schema_block = (
            f"\n【schema 已固定】下游训练脚本按这组必需字段读取数据，`target_schema` 必须原样输出它，"
            f"不要改名或换同义词；最终数据可以保留额外元数据/溯源字段，只要必需字段存在且非空：\n"
            f"{fixed_schema}\n"
        )
    return PROFILER_PROMPT.format(
        task_description=task_description,
        sample_block=sample_block,
        schema_block=schema_block,
    )


# ---------------------------------------------------------------------------
# PipelineAgent（面向 Agent SDK）
# ---------------------------------------------------------------------------
PIPELINE_AGENT_PROMPT = """为下述目标编写一条完整、可直接运行的 DataFlow 流水线（含自定义算子）。

Target: {target}
Expected outputs: {expected_outputs}
Entry file: {entry_path}
{input_block}
要求：
1. 先输出算子决策摘要，再生成完整流水线代码。摘要将原样写入 `{node_dir}/decision.json`，\
**键名必须严格按下面这份 schema**（框架按键名读取，改名/换同义词会导致信息丢失）：
{{"ops": [{{"name": "算子类名", "purpose": "这一步做什么（一句话）", "params": {{}}}}, ...],
 "field_flow": "入口字段（纯生成、无入口语料时写 Empty；以外部数据集为入口时写它的真实字段） \
→ 字段A → 字段A + 字段B → ...",
 "reason": "整体设计理由"}}
其中 `ops` 必须是**对象数组**且按流水线执行顺序列全每一个算子（含自定义算子），\
每个都要给 `purpose`；不要写成 `operators`、`description`、`design_rationale` 等其它键名。
2. 在 `{node_dir}/operators/` 目录下为每个处理步骤编写独立的自定义算子文件（每个算子一个 .py）。\
算子必须继承 OperatorABC（`from dataflow.core import OperatorABC`），实现 run() 方法。\
然后把流水线主文件写入 `{node_dir}/pipeline.py`，决策 JSON 写入 `{node_dir}/decision.json`。
3. {paradigm_rule}
4. 目标 schema（下游训练读取的必需字段及类型）：{target_schema}。最终 JSONL 至少要包含这些字段且非空；
   可额外保留中间字段、溯源字段或辅助标签，下游训练脚本会忽略不使用的字段。
5. 流水线里的 LLM serving 必须用 `APILLMServing_request`，并严格使用下列参数：
{llm_block}
6. FileStorage 的 `cache_path` 设为 `{node_dir}/cache`，使产物落在本节点目录内。\
确保流水线最终会在 `{node_dir}/` 下产出一个 jsonl 结果文件。
7. 写完代码后，**必须执行 compile 校验**：在 `if __name__` 块中按此顺序编写：\
`pipeline.compile()` → `if os.environ.get('DF_COMPILE_ONLY'): sys.exit(0)` → `pipeline.forward()`。\
然后运行 `DF_COMPILE_ONLY=1 python {node_dir}/pipeline.py`。如果 compile() 报 KeyError/ImportError/TypeError，\
修复后重跑，直到 compile 通过。
{recommendation_block}{size_block}{external_data_block}{init_block}{parent_block}{diagnostic_block}{review_block}{sibling_block}{memory_block}
约束：这是无头自动化运行，禁止触发任何交互式询问（AskUserQuestion）或 git/gh 子流程。\
写自定义算子时，可用 MCP get_operator_detail 查看 reasoning 类别的已有算子源码作为编码风格参考。"""

PIPELINE_SELFCORRECT_PROMPT = """上一次生成的流水线 `{node_dir}/pipeline.py` 执行时崩溃了。\
请用 `dataflow-dev` 的 `known_issues.md` 报错速查表匹配下面的错误并修复。

**诊断步骤**：
1. 先读取 `.claude/skills/dataflow-dev/diagnostics/known_issues.md` 匹配错误关键词
2. 如果命中 Issue #010（LLM API）或 #011（FileStorage），直接应用 Issue 中的修复示例
3. 如果不确定 API 签名，查阅 DataFlow 源码确认：
   - FileStorage API: `dataflow/utils/storage.py`
   - LLM Serving API: `dataflow/serving/api_llm_serving_request.py`
   - OperatorABC: `dataflow/core/operator.py`
4. 用 Read 工具查看 `{node_dir}/operators/` 下的自定义算子源码确认问题位置

**常见错误快速修复**：
- `FileStorage.write() takes 2 positional arguments but 3 were given`
  → 修复：`storage.write(df)` (删除第二个参数)

- `'APILLMServing_request' object has no attribute 'request'`
  → 修复：`self.llm_serving.generate_from_input(user_inputs=..., system_prompt=...)`

修复后覆盖写回对应文件（pipeline.py 或 operators/ 下的算子文件），然后运行 \
`DF_COMPILE_ONLY=1 python {node_dir}/pipeline.py` 验证 compile 通过。

{decision_sync_note}

错误栈：
```
{traceback}
```

约束：无头运行，禁止交互式询问与 git/gh 子流程。只修复问题、保持流水线目标不变。"""


# 输入范式共四种组合：input/ 有没有语料 × 外部取数工具开没开。后者必须参与判定——
# 只按 input/ 判定的话，「没有语料」会被写成「不依赖任何输入、凭模型知识合成」，
# 而这句编号硬要求会直接盖掉靠后那段自称"可选"的外部数据说明，Agent 于是从不去取数。
SCRATCH_PARADIGM_RULE = (
    "这是【从零合成（from-scratch synthesis）】任务：input/ 里没有语料，外部取数也未开放，"
    "流水线不依赖任何输入。`first_entry_file_name` 必须指向上面的 Entry file（一个空 jsonl，"
    "仅为满足 DataFlow 的 first_entry 形参）；请用自定义的**生成类算子**凭模型知识合成数据。"
)

SCRATCH_WITH_EXTERNAL_PARADIGM_RULE = (
    "这是【无输入语料（from-scratch synthesis）】任务：input/ 里没有语料。但下面的"
    "【外部数据搜索】工具是**开着的**，所以「凭模型知识生成」不是唯一选项——数据从哪来"
    "是你要做的设计决策。下面几条都合法：\n"
    "   (a) **取真实数据打底**：用 dataset-search / data-discovery 找到合适的公开数据集，"
    "`download_dataset` 落盘后把 `first_entry_file_name` 指向它返回的**绝对路径**，"
    "再用算子把它规整成目标 schema（字段映射、改写、蒸馏、过滤、配对皆可）。"
    "真实语料的多样性通常好过纯生成，也不容易塌缩成模型自己的口癖。\n"
    "   (b) **凭模型知识合成**：`first_entry_file_name` 指向上面的 Entry file"
    "（一个空 jsonl，仅为满足 DataFlow 的 first_entry 形参），用生成类算子从零产出。\n"
    "   (c) **混合**：以真实语料为种子，再用生成类算子增广。\n"
    "   请在 decision.json 的 reason 里写明选了哪条以及为什么。"
)

GROUNDED_PARADIGM_RULE = (
    "这是【语料 grounded 合成（corpus-grounded synthesis）】任务："
    "`first_entry_file_name` 必须指向上面的 Entry file——那是用户提供的真实语料。"
    "流水线要**在这份语料之上**加工：抽取、改写、增强、蒸馏、过滤、配对皆可，"
    "但产出必须能追溯到输入，不要绕开它另起一套凭空生成。"
    "语料字段与目标 schema 不一致时，用算子做字段映射/转换，而不是丢掉输入。"
)

GROUNDED_WITH_EXTERNAL_PARADIGM_RULE = (
    GROUNDED_PARADIGM_RULE
    + "\n   下面的【外部数据搜索】工具也是开着的：可以再引入公开数据集作为**补充**"
    "（补覆盖面、加多样性），在自定义算子里按下载返回的绝对路径读入即可。"
    "但 input/ 的语料是本次任务的主语料，补充不等于替换——不要用外部数据把它顶掉。"
)


def _external_data_block(data_links_dir: str) -> str:
    """外部数据搜索工具的用法与纪律（仅在 agent_sdk.external_data 打开时注入）。

    纪律里最关键的是「下载即落 data_links/、pipeline 里写绝对路径」：Agent 下载的文件若只
    存在于它当时的工作目录假设里，流水线在子进程里重跑就会找不到——而那次失败看起来会像
    是流水线代码的问题，很难追到真正原因。
    """
    return (
        "\n【外部数据搜索（可选）】除了凭模型知识合成，你还可以引入真实的公开数据集。\n"
        "  **dataset-search MCP（取数）**，HuggingFace 侧按此顺序用：\n"
        "  1. `search_datasets(query)` 按关键词找候选（Hub 搜索**不支持语义匹配**，关键词要具体）；\n"
        "  2. `inspect_dataset(dataset_id)` 看 license / configs / splits，**先确认 license 允许使用**；\n"
        "  3. `preview_dataset(dataset_id, config, split)` 取几行真实数据，确认字段与风格对得上目标 schema；\n"
        "  4. `download_dataset(dataset_id, config, split)` 把**整个 split** 落成 jsonl，返回**绝对路径**。\n"
        "  ModelScope 侧是**独立工具链**，不要把 ModelScope 的 dataset_id 传给 HuggingFace 工具：\n"
        "  1. `search_ms_datasets(query)` 按关键词找候选（返回项已带 license/tags/tasks/downloads 等元信息）；\n"
        "  2. `inspect_ms_dataset(dataset_id)` 看仓库元信息与文件列表，**不会下载样本行，也不会执行数据集脚本**；\n"
        "  3. `preview_ms_dataset(dataset_id, config, split)` 取几行真实数据，确认字段与风格；\n"
        "  4. `download_ms_dataset(dataset_id, config, split)` 把 split 落成 jsonl，返回绝对路径。\n"
        "  **HuggingFace 连不上时改用 ModelScope**，找中文语料时也优先考虑。"
        "注意：`inspect_dataset` / `preview_dataset` / `download_dataset` 只用于 HuggingFace；"
        "ModelScope ID（如 `modelscope/...`）只能走 `*_ms_dataset` 工具。\n"
        "  **data-discovery MCP（只找线索、不取数）**：`search_web`（网页）、`search_github`"
        "（数据集仓库）、`search_papers`（arXiv）。Hub 的关键词搜索找不到你想要的东西时用它们——"
        "很多语料是先在论文或博客里被提到，你才知道它叫什么。拿到名字后仍要回上面的 download 工具取数。\n"
        f"  下载的文件统一落在共享目录 {data_links_dir}（跨节点复用）。在流水线里**必须用返回的"
        "绝对路径**读取它（作为 FileStorage 的 first_entry，或在自定义算子里显式读入），"
        "不要假设相对路径——流水线是在独立子进程里跑的。\n"
        "  **不要传 `max_rows` 去限制下载量**：默认全量下载是刻意的——取多少条、怎么采样、按什么"
        "过滤，是你的流水线该做的决策（也是会被评估的部分），在算子里做才可控、可复现、可被后续"
        "节点改进。`max_rows` 只保留 split 的**前 N 行**，不是随机采样：数据集若按难度/来源/时间"
        "排过序，你拿到的就是一段有偏样本。同理，若返回的 meta 里 `complete` 为 false（体积触顶被"
        "截断），说明这份文件也只是头部，据此判断数据源风格时要留意。\n"
        "  **所有外部数据必须经这些工具获取**：不要在流水线或算子里直接 `load_dataset(...)`、"
        "`hf_hub_download(...)`、`requests.get(...)` 或 curl 拉取。特别注意：FileStorage 的 "
        "`first_entry_file_name` 虽然支持 `hf:owner/name:config:split` / `ms:owner/name:split` "
        "这种远程写法（dataflow-dev 技能里有），**这里同样不要用**——请先 download 落盘，再填返回"
        "的本地绝对路径。工具落盘时会写 .meta.json 溯源边车（来源/许可/行数/是否完整/时间），"
        "绕开工具取来的数据没有出处记录，无法通过数据审计。\n"
        "  取舍提醒：引入外部数据是为了拿到模型知识之外的多样性，不是为了凑数。与目标任务分布"
        "不一致的数据会拉低下游表现；宁可少而对，也不要多而杂。\n"
        "  **严禁引入评测基准的测试集**（及其任何镜像/衍生版本）——那是数据泄露，会让分数失真"
        "并使整轮实验作废。\n"
    )


# 外部数据工具未挂载时的对应纪律。少了这段，"不许联网取数据"这条设定就只存在于配置里，
# prompt 对 Agent 只字未提，而它的内置工具（Bash/WebFetch）和 datasets 库一直都在手边。
NO_EXTERNAL_DATA_RULE = (
    "\n【外部数据：本轮不启用】流水线只能用上面的输入语料与模型自身知识，"
    "**不要联网拉取任何外部数据**（`load_dataset(...)` 远程数据集、`hf_hub_download`、"
    "`requests`/curl 下载，以及 FileStorage 的 `hf:` / `ms:` 远程 first_entry 写法，一律禁止）。"
    "这是实验设定的一部分：本轮要衡量的是流水线设计本身，混入外部语料会让结果不可比。"
    "对 LLM serving API 的调用不受此限。\n"
)


def _paradigm_rule(grounded: bool, external: bool) -> str:
    """输入范式（第 3 条要求）：由 input/ 有没有语料（见 core/input_corpus.py）
    与外部取数工具开没开共同决定。

    两个维度都要看：「input/ 为空」只说明用户没提供语料，不代表这条流水线就该凭空生成——
    开了外部取数时，去 Hub 上取一份真实语料再规整成目标 schema 同样合法，往往还更好。
    """
    if grounded:
        return GROUNDED_WITH_EXTERNAL_PARADIGM_RULE if external else GROUNDED_PARADIGM_RULE
    return SCRATCH_WITH_EXTERNAL_PARADIGM_RULE if external else SCRATCH_PARADIGM_RULE


def _init_block(reference_code: str | None, instruction: str | None) -> str:
    """Init node（node_0）专属指引，三档启动方式各渲染一段。

    共同约束是"可靠且有竞争力"：整棵搜索树都从这一个节点变异出去，起点跑不稳
    等于整轮搜索报废；但起点也不应被限制为最小流水线，应允许在有依据时使用外部数据、
    多步处理和质量控制，形成一个可继续变异的强基线。
    """
    common = (
        "\n【初始节点（node_0）】这是整棵搜索树的起点，没有父节点可继承。后续所有变异都基于"
        "它演化，所以请设计一条**稳定、可复现、且具备竞争力**的初始流水线。不要为了追求"
        "最小实现而刻意压低复杂度；如果外部数据、分层采样、过滤、增强或质量控制能带来明确收益，"
        "可以在 node_0 中直接使用，但每个步骤都要有清晰目的并保持可调、可审计。\n"
    )
    if reference_code:
        return common + (
            "\n【参考实现（忠实改写）】下面是一份已有的参考实现。请把它改写成符合本框架契约的"
            "DataFlow 流水线（算子拆分、FileStorage 路径、LLM serving 参数、decision.json 等按上述要求），"
            "**保持其数据处理逻辑与算法取舍不变**——不要顺手做优化、不要增删处理步骤、不要改变"
            "生成或过滤策略。这一档的用途是把参考实现原样搬到同一评测口径下，改动了就失去对照意义：\n"
            f"```python\n{reference_code}\n```\n"
        )
    if instruction:
        return common + (
            f"\n【构建指令】请按下述指令设计这条初始流水线：{instruction}\n"
        )
    return common + (
        "\n【自主设计】没有给定参考实现或指令，请你根据上面的 Target 与目标 schema 自行设计"
        "一条完整且有竞争力的初始流水线：既要稳定产出符合 schema 的数据，也要为后续节点留下"
        "可解释、可局部变异的改进空间。\n"
    )


def build_pipeline_prompt(
    profile: TaskProfile,
    corpus: InputCorpus,
    node_dir: str,
    diagnostic: DiagnosticReport | None = None,
    memory_summary: str | None = None,
    llm_block: str = "",
    parent_code: str | None = None,
    target_samples: int = 0,
    recommended_categories: dict | None = None,
    parent_rationale: str | None = None,
    parent_review: ReviewResult | None = None,
    parent_train_result: TrainResult | None = None,
    siblings: list[str] | None = None,
    is_init: bool = False,
    init_reference_code: str | None = None,
    init_instruction: str | None = None,
    data_links_dir: str = "",
    validated_recipe: dict | None = None,
    recipe_hash: str = "",
    implementation_issues: dict | None = None,
) -> str:
    frozen_source_mode = bool(
        validated_recipe and validated_recipe.get("mode") in {"external", "hybrid"}
    )
    # 目标产出规模：注入后让 Agent 据此设置算子参数（合成式调 num_samples / 转换式控采样过滤）。
    # 关键：产出条数必须【从环境变量 DF_TARGET_SAMPLES 运行时读取】，不要写死——这样同一份
    # pipeline.py 可被复用两档规模：搜索期小规模探针（便宜、给 Tier-1 打分），终选期大规模重产
    # （用于真实 SFT+评测），无需重新调用 Agent SDK 生成代码（见 docs/posttrainbench_adaptation.md）。
    # 默认 0 = 不指定（避免 CondorGenerator 默认仅 3 条这种意外的小数据集，见 review）。
    size_block = ""
    if target_samples and target_samples > 0:
        size_block = (
            f"\n【目标产出规模 / 运行时可调，务必遵守】最终数据集条数**必须从环境变量 "
            f"DF_TARGET_SAMPLES 读取，不要写死具体数字**。在生成类算子里这样取值：\n"
            f"       import os\n"
            f"       num_samples = int(os.environ.get('DF_TARGET_SAMPLES', '{target_samples}'))\n"
            f"   然后用 num_samples 控制产出条数（或循环/批次次数），使最终产出接近该值。\n"
            f"   - 生成类算子：num_samples 直接作为生成条数；若算子内部会对 prompt 数做倍乘/除"
            f"（如 CondorGenerator 每 prompt 产多题），请换算使最终落盘条数≈num_samples；\n"
            f"   - 过滤/验证类算子：确保不会过度过滤导致最终产出远低于 num_samples；\n"
            f"   - 环境变量缺省时回退 {target_samples}，保证脱离编排器单独运行也能出数据。\n"
        )
    # 父节点 pipeline.py 代码作为变异起点：让 Agent「在父节点基础上做有的放矢的变异」，
    # 而非每次从零重写。首层节点的父是 Init node，故除 Init node 自身外这段恒存在。
    parent_block = ""
    if parent_code:
        parent_block = (
            f"\n【变异起点】下面是父节点已验证可运行的流水线代码。请在它的基础上做"
            f"**针对性变异**（改进算子逻辑、调整 prompt、增删处理步骤、改参数等），目标是产出比它更优的数据；"
            f"不要原样照抄，也不要无依据地大改：\n```python\n{parent_code}\n```\n"
        )
    # 设计意图（decision.json 的 reason）独立于代码注入：代码只能表达"做了什么"，不能表达
    # "为什么这么做"。而 code 为空（续跑时 node 目录已被清理，_load_node_code 返回空串）恰恰
    # 是子节点信息最匮乏的时候——没有变异起点只能冷启动，此时设计意图是仅存的方向性信息，
    # 按最需要它的时候把它丢掉，取舍是反的。措辞随代码有无切换：有代码时"在此基础上"的"此"
    # 指那段代码，没代码时这句会悬空。
    if parent_rationale:
        if parent_code:
            parent_block += (
                f"\n【父节点设计意图】父节点这样设计的理由：{parent_rationale}\n"
                f"请把它当作已知约束：在此基础上改进，明确判断该取舍不成立再推翻它。\n"
            )
        else:
            parent_block += (
                f"\n【父节点设计意图】父节点这样设计的理由：{parent_rationale}\n"
                f"（父节点代码本次不可用，需从零编写流水线。）这是父节点的设计取舍，可作为方向参考："
                f"延续其中站得住的部分，或明确判断它不成立后另走一条路。\n"
            )
    if parent_train_result is not None and not parent_train_result.failed:
        parent_block += (
            f"\n【父节点下游验证反馈】父节点在下游验证集上的聚合分数："
            f"{parent_train_result.metric_name}={parent_train_result.score:.4f}。"
            f"请把它当作当前变异起点的真实效果基线：保留可能贡献分数的设计，"
            f"并优先修正评审反馈中指出的弱项。\n"
        )
    diagnostic_block = ""
    if diagnostic and diagnostic.actionable_suggestions:
        suggestions = "\n".join(f"  - {s}" for s in diagnostic.actionable_suggestions)
        diagnostic_block = (
            f"\n【父节点诊断反馈】（请据此有的放矢地调整算子选择/参数）：\n"
            f"   失败概述：{diagnostic.summary}\n"
            f"   可执行建议：\n{suggestions}\n"
        )
    # parent_review 恒有值（父节点必然已通过评审才可被选中扩展），None 仅作防御性兜底。
    review_block = _parent_quality_block(parent_review) if parent_review is not None else ""
    # 兄弟节点已试方向：同一父节点下要长出多个子节点，若每次 prompt 完全相同，就只能指望
    # 采样随机性撞出不同方向，等于把多次昂贵扩展浪费成一次的信息量。
    sibling_block = ""
    if siblings:
        sibling_block = (
            "\n【兄弟节点已试过的方向（请勿重复，另选一个变异方向）】\n"
            + "\n".join(f"  - {s}" for s in siblings)
            + "\n"
        )
    memory_block = ""
    if memory_summary:
        memory_block = f"\n参考历史记忆：\n{memory_summary}\n"
    # 预注入 MCP 推荐结果：指引 Agent 可参考哪些类别的算子源码作为编码风格参考
    recommendation_block = ""
    if recommended_categories:
        rec_list = recommended_categories.get("recommended_categories", [])
        rec_text = "; ".join(
            f"{r['category']}(reason: {r.get('reason', '')})"
            for r in rec_list
        )
        recommendation_block = (
            f"\n【MCP 算子源码参考（强烈建议使用）】\n"
            f"   相关类别：{rec_text}\n"
            f"   **使用步骤**：\n"
            f"   1. 调用 MCP tool: `list_operators` 查看 {rec_list[0]['category']} 类别的算子列表\n"
            f"   2. 选择 1-2 个相关算子（如 ReasoningQuestionGenerator, ReasoningAnswerGenerator）\n"
            f"   3. 调用 MCP tool: `get_operator_detail` 获取算子的**完整源码**（在返回的 source_code 字段中）\n"
            f"   4. 学习源码中的：\n"
            f"      - Prompt 构造模式（如何拼接 system_prompt 和 user_input）\n"
            f"      - LLM 调用方式（generate_from_input 的正确用法）\n"
            f"      - 响应解析技巧（如何从 LLM 输出中提取结构化数据）\n"
            f"      - 错误处理模式（如何处理 None 返回值）\n"
            f"   5. 编写你自己的算子时，遵循相同的模式和 API 约定\n"
            f"   \n"
            f"   **注意**：MCP 提供的是**实现参考**，不要直接复用已有算子，而是学习其编码风格后写自定义算子。\n"
        )
    prompt = PIPELINE_AGENT_PROMPT.format(
        target=profile.task_description,
        entry_path=corpus.entry_path,
        input_block=corpus.prompt_block(),
        paradigm_rule=(
            "这是【冻结来源实现】任务：数据来源已经由 RecipeAgent 选择并下载。"
            "若有多个 external JSONL，可以在流水线中分别读取，也可以先规范化/合并为节点内的派生"
            "中间文件再作为 first_entry；无论采用哪种实现，都必须保留每条真实行的 source_id 和 record_id，"
            "并在 decision.json 说明派生文件来自哪些 recipe sources。external 只做清洗/映射/过滤/"
            "有约束改写，hybrid 还需按 mixing_policy 保留真实行并生成可追溯 seed 的合成行。"
            if frozen_source_mode
            else _paradigm_rule(corpus.grounded, bool(data_links_dir) and not validated_recipe)
        ),
        expected_outputs=list(profile.target_schema.keys()),
        node_dir=node_dir,
        target_schema=profile.target_schema,
        llm_block=llm_block or "   （未提供具体 API 参数，沿用技能默认。）",
        recommendation_block=recommendation_block,
        size_block=size_block,
        external_data_block=(
            "\n【数据源职责边界】本阶段没有数据发现/下载 MCP。只能使用已验证 recipe.sources 中"
            "的本地 location；不得临时搜索、下载、替换或追加数据源。\n"
            if frozen_source_mode
            else (_external_data_block(data_links_dir) if data_links_dir else NO_EXTERNAL_DATA_RULE)
        ),
        init_block=_init_block(init_reference_code, init_instruction) if is_init else "",
        parent_block=parent_block,
        diagnostic_block=diagnostic_block,
        review_block=review_block,
        sibling_block=sibling_block,
        memory_block=memory_block,
    )
    if validated_recipe:
        mode = validated_recipe.get("mode", "")
        sources = [s.get("source_id") for s in validated_recipe.get("sources", []) if isinstance(s, dict)]
        mode_rules = {
            "external": (
                "external：最终每行必须标记 `_source_type=external`，`_source_id` 必须来自 recipe.sources，"
                "并有全局唯一 `_record_id`；不得生成独立 synthetic 行。优先让 FileStorage.first_entry "
                "指向真实数据，再做清洗、映射、过滤或有来源约束的改写。"
            ),
            "hybrid": (
                "hybrid：推荐拓扑为 external first_entry → 规范化并保留 external 行 → 基于每个 seed 生成 "
                "synthetic 行 → pandas.concat → 验证/去重/覆盖选择。external 行必须有 recipe source_id 和唯一 "
                "`_record_id`；synthetic 行必须有 `_source_type=synthetic`、唯一 `_record_id` 和指向保留 external "
                "行的 `_seed_id`。最终比例必须满足 mixing_policy。"
            ),
            "synthetic": (
                "synthetic：first_entry 必须是空入口，不得读取真实/外部数据。最终每行必须标记 "
                "`_source_type=synthetic`、非空 `_source_id` 和全局唯一 `_record_id`。"
            ),
        }.get(mode, "")
        prompt += (
            "\n【已通过框架校验并冻结的 Data Recipe】\n"
            f"recipe_hash: {recipe_hash}\n"
            f"```json\n{json.dumps(validated_recipe, ensure_ascii=False, sort_keys=True, indent=2)}\n```\n"
            "你现在只负责实现这份 recipe，不得更改 recipe.json。pipeline.py、operators 和 "
            "decision.json 必须忠实实现其 mode、coverage buckets、proposal strategies、verification "
            "和 selection。hybrid/synthetic 产物必须为每条记录保留 `_bucket`、`_generation_strategy` "
            "与 `_operation_graph` 元数据，供框架做覆盖和结构坍塌审计。纯合成不得用固定 prompt "
            "+ variant/index number 充当多样性；应按 recipe 的不同技能、运算结构、上下文和难度生成候选，"
            "再验证、规范化去重并按 bucket 配额选择。\n"
            f"【mode 数据来源硬契约】声明的 source_id：{sources}\n{mode_rules}\n"
            "框架会扫描最终 JSONL 的来源比例、source_id、record_id 和 hybrid seed link；仅在 decision.json "
            "里声称遵守不会通过审计。\n"
        )
    if implementation_issues:
        prompt += (
            "\n【父流水线的实现级问题】这些信号用于避免重复实现缺陷，不允许据此修改冻结 recipe：\n"
            f"```json\n{json.dumps(implementation_issues, ensure_ascii=False, sort_keys=True, indent=2)}\n```\n"
        )
    return prompt


def build_recipe_prompt(
    profile: TaskProfile,
    corpus: InputCorpus,
    node_dir: str,
    target_samples: int,
    external_data: bool,
    parent_recipe: dict | None = None,
    context: RecipeContext | None = None,
) -> str:
    modes = ["external", "hybrid"] if corpus.grounded else (
        ["external", "hybrid", "synthetic"] if external_data else ["synthetic"]
    )
    if context is None:
        context = RecipeContext(parent={"recipe": parent_recipe} if parent_recipe else None)
    rendered_context = json.dumps(
        context.to_dict(), ensure_ascii=False, sort_keys=True, indent=2
    )
    source_tool_rule = (
        "- external/hybrid 模式下，由你使用 dataset-search/data-discovery MCP 搜索、检查、预览并下载来源，\n"
        "  然后把工具返回的绝对本地路径、license 和 train split 写进 sources。PipelineAgent 不具备这些 MCP。"
        if external_data
        else "- 本轮未开放外部数据 MCP；external/hybrid 的真实来源只能使用 Input corpus，并在 sources 中记录其入口。"
    )
    return f"""在编写任何流水线代码之前，先规划数据生成策略。
本阶段只能写入 `{node_dir}/recipe.json`；禁止创建或修改 `pipeline.py`、`decision.json`
和 `operators/`。配方将由框架的确定性校验器验证，不由你自行验收。

目标任务：{profile.task_description}
必需输出 schema：{profile.target_schema}
输入语料：{corpus.prompt_block()}
允许的 mode：{modes}
目标样本数：{target_samples}
RecipeContext（只据此做数据策略决策；不要把 sibling/global pipeline 作为输入）：
```json
{rendered_context}
```

决策纪律：
- parent.compliance 决定 recipe 是否被真实执行；合规不足时，不得把下游分数直接归因给 recipe。
- parent.quality/downstream 用于调整 bucket、strategy、来源比例和 verification。
- siblings 是紧凑 outcome，用于避免重复相同 recipe 变异；不要索要其完整 pipeline。
- global_memory 只包含全树摘要：高质量 recipe、Pareto recipe、失败 pattern 与已观测变更收益。
{source_tool_rule}

`recipe.json` 必须是符合以下 contract 的 JSON object：
{{
  "version": 1,
  "mode": "external|hybrid|synthetic",
  "target_samples": {target_samples},
  "candidate_multiplier": 1.5,
  "sources": [{{
    "source_id": "stable-source-id", "kind": "input_corpus|external_dataset",
    "location": "absolute local path", "license": "license name", "split": "train"
  }}],
  "mixing_policy": {{
    "external_min_fraction": 0.3, "synthetic_min_fraction": 0.5,
    "require_seed_link": true, "preserve_external_rows": true
  }},
  "coverage_buckets": [{{
    "name": "stable unique name", "skills": ["..."], "operations": ["..."],
    "contexts": ["..."], "difficulty": "...", "answer_type": "...", "quota": 100
  }}],
  "proposal_strategies": {{"program_first": 0.5, "constraint_composition": 0.5}},
  "verification": {{"methods": ["symbolic_check", "independent_recompute"]}},
  "selection": {{"normalized_dedup": true, "coverage_aware": true}}
}}

对于 hybrid/synthetic：各 bucket 的 quota 总和不得低于 target_samples；必须先过量生成候选，
再执行验证、规范化去重和按覆盖配额选择。synthetic 至少需要两种实质不同的 proposal strategy。
严禁用固定 prompt 仅追加 variant/index number 来伪造多样性。
mode 特定来源契约：
- external：sources 至少一个；只使用真实来源，必须记录 location/license/train split。
- hybrid：sources 至少一个且必须提供 mixing_policy；真实行必须保留，以便 synthetic 行通过 seed_id 追溯。
- synthetic：sources 必须严格为 []，不得读取真实或外部数据。
所有 mode 都必须配置 verification.methods 和 selection.normalized_dedup=true。
这是无头运行：不要提问，不要使用 git/gh，只写上述 `recipe.json`。
"""


def build_recipe_repair_prompt(node_dir: str, errors: list[str]) -> str:
    listed = "\n".join(f"- {error}" for error in errors)
    return f"""框架的确定性校验器拒绝了 `{node_dir}/recipe.json`，错误如下：
{listed}
请逐条修复所有错误，并且只修改 `{node_dir}/recipe.json`。禁止编写 pipeline code、
`decision.json` 或 `operators/`；如果上一轮越权创建了这些实现产物，必须先删除它们。
不要提问，不要使用 git/gh。
"""


def build_selfcorrect_prompt(node_dir: str, traceback: str) -> str:
    return PIPELINE_SELFCORRECT_PROMPT.format(
        node_dir=node_dir,
        traceback=traceback,
        decision_sync_note=DECISION_SYNC_NOTE.format(node_dir=node_dir),
    )


PIPELINE_REVIEW_SELFCORRECT_PROMPT = """上一次生成的流水线 `{node_dir}/pipeline.py` 能跑通，\
但产出的数据被 Phase-1 评审判定【不达标】（未进入训练）。请修复数据质量问题，然后覆盖写回 \
`{node_dir}/pipeline.py` 或 `{node_dir}/operators/` 下的算子文件（保持流水线目标不变）。

评审反馈的问题：
{issues}
{signals_block}

【关键定位线索】DataFlow 的 FileStorage 会在 `{node_dir}/cache/` 下为**每个算子**按执行顺序落一个\
中间 jsonl（step1 = 第 1 个算子产出，step2 = 第 2 个算子产出，依此类推；可对照 `pipeline.py` 里 \
`forward()` 的算子调用顺序）。请用 Glob / Read 工具**逐个查看这些中间文件里的实际数据**，\
逐 step 追踪数据质量是从哪一步（哪个算子）开始变坏的——既包括行数骤减/变空，也包括行数没变但\
**内容劣化**（某字段从有值变 null/空、schema 不符、跑题等）——据此判断是哪个算子的逻辑或 prompt 有问题并修复它。

【DataFlow 源码参考】如果不确定 API 用法，可以查阅 DataFlow 源码确认：
- LLM Serving API 签名: `dataflow/serving/api_llm_serving_request.py`
- FileStorage API: `dataflow/utils/storage.py`
- Operator 基类: `dataflow/core/operator.py`
- 已有算子实现参考: `dataflow/operators/reasoning/` (数学推理类算子)

常见数据质量问题修复方向：
1. **字段为空/null**：检查对应算子的 LLM prompt 是否明确要求输出该字段
2. **LLM 返回 None**：检查是否正确处理 API 失败（`[r if r is not None else '' for r in responses]`）
3. **内容跑题**：检查 system_prompt 和 prompt 模板是否约束足够强
4. **格式不符**：检查响应解析逻辑（如 JSON 提取、正则匹配）

修复后运行 `DF_COMPILE_ONLY=1 python {node_dir}/pipeline.py` 验证 compile 通过。
{history_block}
{decision_sync_note}

约束：无头运行，禁止交互式询问（AskUserQuestion）与 git/gh 子流程。只修复数据质量、保持目标不变。"""

# 自修正只被要求覆盖写回 pipeline.py / operators/，于是 decision.json 会停留在初次生成那一版：
# 同一个 PipelineConfig 里 code 是新的、operators/field_flow/rationale 是旧的。这份错配会传染到
# global memory 的 ops/flow、子节点继承的 parent_rationale、以及 checkpoint。
DECISION_SYNC_NOTE = """【务必同步】若本次修复**增删了算子、改变了字段流转、或调整了设计取舍**，\
请同步覆盖写回 `{node_dir}/decision.json`（键名仍为 `ops` / `field_flow` / `reason`）。\
框架按这份文件记录本节点的算子构成与设计意图，并据此渲染历史记忆、传给子节点作为变异依据；\
不更新会让下游看到一份已不存在的设计。"""


def build_review_selfcorrect_prompt(
    node_dir: str,
    issues: list[str],
    history: list[str] | None = None,
    failure_signals: list[str] | None = None,
) -> str:
    issues_text = "\n".join(f"  - {s}" for s in issues) if issues else "  - （评审未给出具体条目，请据中间产物自行判断）"
    # 失败信号在渲染时按编号查表拼出。ReviewResult 里信号与 issues 是分开的两个字段，
    # 需要信号文本的消费点自己组装即可——FAILURE_SIGNALS 就在本模块，零成本。
    signals_block = ""
    if failure_signals:
        listed = "\n".join(
            f"  - {sid}：{FAILURE_SIGNALS[sid]}"
            for sid in failure_signals
            if sid in FAILURE_SIGNALS
        )
        if listed:
            signals_block = f"\n命中的失败信号（标准化标签，指示改进方向）：\n{listed}\n"
    # 历史修复轮次：只给最新一轮 issues 会让 LLM 不知道某段逻辑是上一轮为修 X 才加的，
    # 从而把它改掉、下一轮 X 复现，在节点内部空转掉有限的自修正次数。
    history_block = ""
    if history:
        history_block = (
            "\n【本节点已尝试过的修复（勿推翻其中已见效的部分，除非确有必要）】\n"
            + "\n".join(f"  - {h}" for h in history)
            + "\n"
        )
    return PIPELINE_REVIEW_SELFCORRECT_PROMPT.format(
        node_dir=node_dir,
        issues=issues_text,
        signals_block=signals_block,
        history_block=history_block,
        decision_sync_note=DECISION_SYNC_NOTE.format(node_dir=node_dir),
    )


# ---------------------------------------------------------------------------
# ReviewAgent（Phase-1 快反馈）
# ---------------------------------------------------------------------------
# ANDES F1-F7 失败信号分类。让 Tier-1 评审在打分之外标注命中的信号编号，得到标准化、可解释的
# 问题标签，存于 `ReviewResult.failure_signals`（与具体证据 `issues` 分开保存，各渲染点按编号
# 查本表拼文本）。仅作观测/反馈，不改变 tier1_score 的打分公式（见 docs/posttrainbench_adaptation.md）。
FAILURE_SIGNALS: dict[str, str] = {
    "F1": "描述过窄：生成围绕极少数结构展开，能力维度单一",
    "F2": "华而不实：表面花哨但逻辑塌陷，缺乏实质数据多样性",
    "F3": "模板锁定：大规模近重复 / 模板化雷同",
    "F4": "覆盖塌陷：未覆盖目标任务相关的关键能力维度",
    "F5": "约束漂移：偏离目标任务/领域的要求或答案格式",
    "F6": "有形无实：形式满足但训练价值低（过易/无信息量）",
    "F7": "描述空泛：生成结果语义松散、逐渐跑题",
}


# Tier-1 五维（与 ReviewAgent 的加权维度一致）：标量 tier1 对生成者几乎无指导价值，
# 拆成五维并点名最弱维，才能把"往哪改"讲清楚。
_DIM_LABELS: tuple[tuple[str, str], ...] = (
    ("correctness_score", "正确性"),
    ("relevance_score", "相关性"),
    ("difficulty_score", "难度"),
    ("diversity_score", "多样性"),
    ("schema_score", "schema 符合度"),
)


def _weakest_dims(review: ReviewResult, k: int = 2) -> list[tuple[str, float]]:
    dims = [(label, float(getattr(review, attr, 0.0) or 0.0)) for attr, label in _DIM_LABELS]
    return sorted(dims, key=lambda kv: kv[1])[:k]


def _parent_quality_block(review: ReviewResult) -> str:
    """把父节点的 Tier-1 归因渲染成可行动的变异指引。"""
    dims = " | ".join(
        f"{label} {float(getattr(review, attr, 0.0) or 0.0):.2f}" for attr, label in _DIM_LABELS
    )
    weakest = "、".join(f"{label}({score:.2f})" for label, score in _weakest_dims(review))
    lines = [
        "\n【父节点数据质量归因】",
        f"  Tier-1 复合分：{review.tier1_score:.3f}（评审{'通过' if review.passed else '未通过'}）",
        f"  五维得分：{dims}",
        f"  **最弱维度：{weakest} ← 本次变异应优先攻这些维度**",
        f"  硬指标：重复率 {review.dup_rate:.2f} | 空值率 {review.null_rate:.2f} | "
        f"可解析率 {review.parse_valid_rate:.2f}",
    ]
    if review.failure_signals:
        lines.append("  命中的失败信号（针对性改进方向）：")
        lines.extend(
            f"    - {sid}：{FAILURE_SIGNALS[sid]}"
            for sid in review.failure_signals
            if sid in FAILURE_SIGNALS
        )
    detail = review.issues
    if detail:
        lines.append("  评审原文问题（具体证据，逐条对照修正）：")
        lines.extend(f"    - {s}" for s in detail)
    return "\n".join(lines) + "\n"


def _failure_signals_block() -> str:
    return "\n".join(f"  - {sid}：{desc}" for sid, desc in FAILURE_SIGNALS.items())


REVIEW_PROMPT = """你是数据质量评审员。下面是某条流水线生成数据集的抽样，\
以及目标任务画像。请从五个维度评估这批**合成数据**用于微调目标模型的价值（各维 0.0-1.0）。

目标任务：{task_description}
期望 schema（下游训练读取的必需字段→类型）：{target_schema}
质量要点：{quality_criteria}

数据抽样（每行一条 JSON）：
{samples}

评分维度说明：
- correctness_score：解答/标签是否正确、推理过程与最终答案是否自洽一致（错误答案、推理与答案矛盾都要扣分）。
- relevance_score：与目标任务/领域的相关性。
- difficulty_score：对目标模型是否**非平凡且可学**——中间难度带最高；过于简单（无学习价值）或超纲无法学都要扣分。
- diversity_score：抽样批内的多样性、是否存在大量近重复/模板化雷同。
- schema_score：必需字段的齐全度与类型符合度；额外元数据/溯源字段不应扣分。

常见失败信号（若明确命中，请在 failure_signals 中列出对应编号；没有则留空数组）：
{failure_signal_block}

请只输出一个 JSON 对象：
{{
  "correctness_score": 0.0-1.0,
  "relevance_score": 0.0-1.0,
  "difficulty_score": 0.0-1.0,
  "diversity_score": 0.0-1.0,
  "schema_score": 0.0-1.0,
  "failure_signals": ["F3", ...],
  "issues": ["发现的问题（尽量具体，可引用样本）", ...]
}}
不要输出 JSON 以外的内容。"""


def build_review_prompt(profile: TaskProfile, samples_text: str) -> str:
    return REVIEW_PROMPT.format(
        task_description=profile.task_description,
        target_schema=profile.target_schema,
        quality_criteria=profile.quality_criteria,
        samples=samples_text,
        failure_signal_block=_failure_signals_block(),
    )


# ---------------------------------------------------------------------------
# DiagnosticAgent（Phase-2 慢反馈）
# ---------------------------------------------------------------------------
DIAGNOSTIC_PROMPT = """你是模型失败归因专家。下面是某数据集训练出的模型在下游评估中的失败样例。\
请聚类失败模式，并给出"如何改进数据生成流水线"的可执行建议。

下游任务：{downstream_task}
失败样例（input / prediction=模型完整输出 / extracted_answer=解析出的最终答案 / ground_truth / error_type）：
{bad_cases}

请只输出一个 JSON 对象：
{{
  "summary": "一句话概括主要失败模式",
  "failure_patterns": ["失败模式1", ...],
  "actionable_suggestions": ["对数据流水线的具体改进建议（指向算子/参数/数据来源）", ...]
}}
不要输出 JSON 以外的内容。"""


def build_diagnostic_prompt(downstream_task: str, bad_cases_text: str) -> str:
    return DIAGNOSTIC_PROMPT.format(
        downstream_task=downstream_task, bad_cases=bad_cases_text
    )
