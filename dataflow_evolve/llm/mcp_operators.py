"""In-process MCP server for DataFlow operator discovery (claude_agent_sdk).

Uses the live OPERATOR_REGISTRY to dynamically discover all operators at import
time. Operator parameters are extracted via `inspect` (not static AST parsing).

Exposes 4 tools for hierarchical operator browsing:
  - list_operator_categories
  - recommend_operator_categories
  - list_operators
  - get_operator_detail

Usage:
    from dataflow_evolve.llm.mcp_operators import operator_mcp_server

    options = ClaudeAgentOptions(
        mcp_servers={"dataflow-operators": operator_mcp_server},
        ...
    )
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from typing import Annotated, Any

from claude_agent_sdk import create_sdk_mcp_server, tool

# ---------------------------------------------------------------------------
# Registry bootstrap — ensure dataflow package is importable from _agent_root
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = PROJECT_ROOT / "workspace" / "_agent_root"

if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from dataflow.utils.registry import OPERATOR_REGISTRY  # noqa: E402

OPERATOR_REGISTRY._init_loaders()

# Fault-tolerant full load: some operators have heavy deps (pandas, torch, etc.)
# that may not be installed. We load what we can and skip the rest.
_load_errors: list[str] = []
for _loader_name, _loader in OPERATOR_REGISTRY.loader_map.items():
    if _loader is None:
        continue
    for _cls_name in list(getattr(_loader, "__all__", [])):
        try:
            getattr(_loader, _cls_name)
        except Exception as _e:
            _load_errors.append(f"{_loader_name}/{_cls_name}: {type(_e).__name__}")


# ---------------------------------------------------------------------------
# inspect-based introspection helpers (ported from WebUI)
# ---------------------------------------------------------------------------


def _safe_json_val(val: Any) -> Any:
    if val is inspect.Parameter.empty:
        return None
    if isinstance(val, type) or callable(val):
        return str(val)
    try:
        json.dumps(val)
        return val
    except TypeError:
        return str(val)


def _param_to_dict(p: inspect.Parameter) -> dict[str, Any]:
    return {
        "name": p.name,
        "default_value": _safe_json_val(p.default),
        "kind": p.kind.name,
    }


def _get_method_params(method: Any, skip_self: bool = True) -> list[dict[str, Any]]:
    try:
        sig = inspect.signature(method)
        params = list(sig.parameters.values())
        if skip_self and params and params[0].name == "self":
            params = params[1:]
        return [_param_to_dict(p) for p in params]
    except Exception:
        return []


def _call_get_desc(cls: type, lang: str = "zh") -> str:
    func_obj = cls.__dict__.get("get_desc")
    if not isinstance(func_obj, staticmethod):
        doc = inspect.getdoc(cls)
        return (doc or "").split("\n")[0] if doc else ""

    fn = func_obj.__func__
    params = list(inspect.signature(fn).parameters)
    try:
        if params == ["lang"]:
            result = fn(lang)
        elif params == ["self", "lang"]:
            result = fn(None, lang)
        else:
            return ""
        if isinstance(result, list):
            return "\n".join(str(item) for item in result)
        return str(result) if result else ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Category guide (adopted from WebUI's operator_category_guide.py)
# ---------------------------------------------------------------------------

CATEGORY_GUIDE: dict[str, dict] = {
    "core_text": {
        "use_for": (
            "通用文本→QA / 多跳 QA / 摘要 / 长文本切块生成；prompt-driven 的 "
            "filter / refine / evaluator (PromptedFilter/PromptedRefiner/"
            "PromptedEvaluator)；表格/字段重写用 PandasOperator。"
            "首选这一类做『从原始文本生成 SFT 训练数据』。"
        ),
        "not_for": (
            "**不要**用 SFTGeneratorSeed —— 那个属于 text_sft，是从已有 SFT "
            "种子扩展，而不是从原始文本生成 QA。需要『基础 QA 生成』时直接选 "
            "Text2QAGenerator 或 Text2MultiHopQAGenerator。"
            "也不要用它做 PT 语料质量过滤（去 text_pt）或代码相关任务（去 code）。"
        ),
        "examples": [
            "Text2QAGenerator", "Text2MultiHopQAGenerator",
            "ChunkedPromptedGenerator", "PromptedFilter",
            "PromptedRefiner", "PandasOperator",
        ],
    },
    "general_text": {
        "use_for": (
            "纯文本表面层清洗与统计：基于规则的 Filter（长度、字符比例、"
            "html 实体、URL、PII），N-gram / Hash 去重，BLEU/CIDEr 等无 LLM "
            "评测。**所有不需要 LLM 调用的『通用文本质量过滤』都来这里找**。"
        ),
        "not_for": (
            "需要语义理解的过滤（如『基于 LLM 判断是否为优质问题』）应该用 "
            "core_text 的 PromptedFilter；代码质量过滤去 code；"
            "PT 语料 perplexity/edu-classifier 过滤去 text_pt。"
        ),
        "examples": [
            "CharNumberFilter", "NgramFilter", "HashDeduplicateFilter",
            "HtmlEntityRefiner", "LLMLanguageFilter",
            "ContentNullFilter", "BleuSampleEvaluator",
        ],
    },
    "text_sft": {
        "use_for": (
            "已经有 SFT 数据（instruction / response 列）后做的事："
            "用 Condor / Treeinstruct / Deita / Alpagasus / Superfiltering / "
            "RM / Instag 等方法对 SFT 样本打分、过滤、扩写。"
        ),
        "not_for": (
            "**不要**在『还没生成 QA / instruction』的阶段用这一类。"
            "如果输入只是原始文档，应该先去 core_text（Text2QAGenerator）"
            "或 reasoning（ReasoningQuestionGenerator）生成训练样本，"
            "再回来走 text_sft 做质量评估。SFTGeneratorSeed 也不是『从文本生成 QA』，"
            "它需要既有 SFT 种子。"
        ),
        "examples": [
            "CondorGenerator", "CondorRefiner", "DeitaQualityFilter",
            "AlpagasusFilter", "SuperfilteringFilter", "TreeinstructFilter",
        ],
    },
    "text_pt": {
        "use_for": (
            "**预训练（PT）语料**质量过滤：perplexity 过滤、FineWeb-Edu 分类器、"
            "DebertaV3 质量、CCNet 去重。Phi4QAGenerator 是 PT 风格的 QA 合成。"
        ),
        "not_for": (
            "SFT/QA 数据评估走 text_sft；通用规则去重走 general_text；"
            "推理/数学数据去 reasoning。"
        ),
        "examples": [
            "PerplexityFilter", "FineWebEduFilter", "DebertaV3Filter",
            "CCNetDeduplicateFilter", "Phi4QAGenerator",
        ],
    },
    "reasoning": {
        "use_for": (
            "数学/逻辑推理 CoT 数据：从问题生成 CoT 答案、答案抽取、"
            "题目难度/分类评估、答案 N-gram/format/judge 过滤、"
            "PT 格式转换。**只有当任务明确是 math / step-by-step CoT 时才选这里**。"
        ),
        "not_for": (
            "通用 QA 生成不在这里，去 core_text；代码推理不在这里，去 code；"
            "纯文本清洗不在这里，去 general_text。"
        ),
        "examples": [
            "ReasoningQuestionGenerator", "ReasoningAnswerGenerator",
            "ReasoningQuestionFilter", "ReasoningAnswerFormatterFilter",
            "ReasoningQuestionDifficultySampleEvaluator",
        ],
    },
    "code": {
        "use_for": (
            "代码数据：从代码生成指令、从指令生成代码、代码质量打分、"
            "auto-generated 代码检测、长度/编码异常过滤、sandbox 评估。"
        ),
        "not_for": (
            "自然语言 QA / refine 不在这里；代码相关 SFT 评估也不能直接套通用 "
            "DeitaQuality（去 text_sft 但参数要切换 instruction/response）。"
        ),
        "examples": [
            "CodeInstructionGenerator", "CodeCodeToInstructionGenerator",
            "CodeQualityScoreFilter", "CodeAutoGeneratedFilter",
            "CodeSandboxSampleEvaluator",
        ],
    },
    "conversations": {
        "use_for": (
            "多轮对话/智能体场景的合成与评估：场景抽取/扩写、原子任务/"
            "顺序任务生成、function-calling 对话生成与评估、组合任务过滤。"
        ),
        "not_for": (
            "单轮 QA 不属于这里（去 core_text）；单条 instruction 评估不属于这里"
            "（去 text_sft）。"
        ),
        "examples": [
            "MultiTurnConversationGenerator", "ConsistentChatGenerator",
            "ScenarioExtractGenerator", "FunctionGenerator",
            "FuncCallConversationSampleEvaluator",
        ],
    },
    "knowledge_cleaning": {
        "use_for": (
            "**文档/URL 摄取链路**：File/URL→Markdown 转换（API/Flash/Local 三种），"
            "KBC 切块、清洗、批量 QA 生成。把『PDF/HTML/网页 → 训练数据』的"
            "前几步全部放在这里。"
        ),
        "not_for": (
            "如果输入已经是干净纯文本，跳过这一类，直接去 core_text。"
            "API-only 部署中 KBCChunkGenerator 等可能因依赖缺失不可用，"
            "此时回退到 core_text 的 ChunkedPromptedGenerator。"
        ),
        "examples": [
            "FileOrURLToMarkdownConverterAPI", "FileOrURLToMarkdownConverterFlash",
            "KBCChunkGenerator", "KBCMultiHopQAGeneratorBatch",
            "MathBookQuestionExtract",
        ],
    },
    "agentic_rag": {
        "use_for": "Agentic / 多跳 RAG 训练数据合成：原子任务、宽度/深度 QA 生成与 F1 评估。",
        "not_for": "普通单跳 QA 走 core_text；多轮对话走 conversations。",
        "examples": [
            "AgenticRAGAtomicTaskGenerator", "AgenticRAGDepthQAGenerator",
            "AgenticRAGWidthQAGenerator", "AgenticRAGQAF1SampleEvaluator",
        ],
    },
    "text2sql": {
        "use_for": "Text2SQL 数据生成与评估：可执行性过滤、组件分类、CoT 投票生成。",
        "not_for": "通用 QA / 推理任务不要走这里。",
        "examples": [
            "Text2SQLCoTVotingGenerator", "SQLExecutabilityFilter",
            "SQLExecutionFilter", "SQLComponentClassifier",
        ],
    },
    "pdf2vqa": {
        "use_for": "PDF→视觉问答数据：MinerU 输入构造、PDF/QA 合并、VQA 格式化、LLM 输出解析。",
        "not_for": "纯文本 QA 走 core_text；普通 OCR/markdown 走 knowledge_cleaning。",
        "examples": [
            "MinerU2LLMInputOperator", "PDF_Merger", "QA_Merger",
            "VQAFormatter", "LLMOutputParser",
        ],
    },
    "core_vision": {
        "use_for": "通用视觉 QA 合成（PromptedVQAGenerator）。",
        "not_for": "PDF 专用流水线去 pdf2vqa。",
        "examples": ["PromptedVQAGenerator"],
    },
    "core_speech": {
        "use_for": "语音转文本（Speech2TextGenerator）。",
        "not_for": "其他模态不属于这里。",
        "examples": ["Speech2TextGenerator"],
    },
    "chemistry": {
        "use_for": "化学领域：从文本抽取 SMILES、SMILES 等价数据集评估。",
        "not_for": "其他领域不属于这里。",
        "examples": [
            "ExtractSmilesFromTextGenerator", "SmilesEquivalenceDatasetEvaluator",
        ],
    },
}


# ---------------------------------------------------------------------------
# Build operator catalog from live registry
# ---------------------------------------------------------------------------

def _build_catalog() -> dict[str, list[dict]]:
    """Build {category: [operator_info, ...]} from live OPERATOR_REGISTRY."""
    obj_map = OPERATOR_REGISTRY.get_obj_map()
    type_map = OPERATOR_REGISTRY.get_type_of_objects()

    catalog: dict[str, list[dict]] = {}
    for op_name, cls in obj_map.items():
        type_path = type_map.get(op_name, [])
        category = type_path[2] if len(type_path) > 2 else "unknown"
        subcategory = type_path[3] if len(type_path) > 3 else "unknown"

        description = _call_get_desc(cls)
        init_params = _get_method_params(getattr(cls, "__init__", None))
        run_params = _get_method_params(getattr(cls, "run", None))
        allowed_prompts = [p.__name__ for p in getattr(cls, "ALLOWED_PROMPTS", [])]

        info = {
            "name": op_name,
            "category": category,
            "subcategory": subcategory,
            "description": description,
            "allowed_prompts": allowed_prompts,
            "init_params": init_params,
            "run_params": run_params,
        }
        catalog.setdefault(category, []).append(info)

    return catalog


_catalog = _build_catalog()


# ---------------------------------------------------------------------------
# Recommendation engine (ported from WebUI operator_category_guide.py)
# ---------------------------------------------------------------------------

def _recommend_categories(
    task_description: str,
    dataset_columns: list[str] | None = None,
    max_categories: int = 2,
) -> dict:
    dataset_columns = dataset_columns or []
    task_lower = (task_description or "").lower()
    col_lower = [c.lower() for c in dataset_columns]

    score_map: dict[str, int] = {}
    reason_map: dict[str, list[str]] = {}
    avoid_map: dict[str, list[str]] = {}

    def add(cat: str, score: int, reason: str) -> None:
        score_map[cat] = score_map.get(cat, 0) + score
        reason_map.setdefault(cat, []).append(reason)

    def avoid(cat: str, score: int, reason: str) -> None:
        avoid_map.setdefault(cat, []).append(reason)
        score_map.setdefault(cat, 0)

    def has_any(*keywords: str) -> bool:
        return any(k in task_lower for k in keywords)

    # --- keyword scoring ---
    if has_any("multi-hop", "multihop", "qa pair", "question-answer", "question answer", "问答", "多跳", "qa对", "qa 对"):
        add("core_text", 8, "task explicitly asks for QA / multi-hop QA generation")
        avoid("text_sft", 3, "QA-from-text is not a text_sft construction task")

    if has_any("summary", "summarize", "摘要", "总结", "classify", "classification", "sentiment", "score", "rewrite", "paraphrase"):
        add("core_text", 4, "task is LLM-style text generation / evaluation / refinement")

    if has_any("filter", "threshold", ">=", "<=", "rename", "flatten", "derive", "字段重命名", "展平", "过滤", "新增字段"):
        add("core_text", 4, "task needs deterministic field-level transform or rule filter")

    if has_any("emoji", "html", "url", "whitespace", "dedup", "language detect", "punctuation", "去重", "清洗", "html实体", "标点", "空格"):
        add("general_text", 6, "task mentions surface-form cleanup / language / dedup / deterministic text filters")

    if has_any("pdf", "markdown", "网页", "url to markdown", "chunk", "切块", "文档摄取", "知识库"):
        add("knowledge_cleaning", 7, "task starts from documents / URLs / PDF and likely needs ingestion operators")

    if has_any("instruction", "sft", "alpaga", "deita", "superfiltering"):
        add("text_sft", 7, "task explicitly mentions SFT / instruction-quality processing")

    if has_any("reasoning", "cot", "math", "逻辑推理", "推理题", "数学"):
        add("reasoning", 7, "task explicitly mentions reasoning / CoT / math")

    if has_any("code", "program", "函数", "代码"):
        add("code", 7, "task explicitly mentions code data")

    if has_any("conversation", "dialog", "chat", "多轮对话", "对话"):
        add("conversations", 6, "task explicitly mentions conversations / multi-turn data")

    if has_any("sql", "text2sql", "数据库"):
        add("text2sql", 7, "task explicitly mentions SQL / database queries")

    if has_any("vqa", "vision", "图像问答", "图片"):
        add("core_vision", 6, "task explicitly mentions visual QA")

    if has_any("speech", "audio", "语音"):
        add("core_speech", 6, "task explicitly mentions speech/audio")

    if has_any("smiles", "chemistry", "化学"):
        add("chemistry", 6, "task explicitly mentions chemistry / smiles")

    # --- column-based scoring ---
    if any(col in {"chunk", "text", "doc", "review_text", "raw_content", "content"} for col in col_lower):
        add("core_text", 2, "dataset columns look like text-generation inputs")

    if any(col in {"url", "pdf_path", "markdown", "html"} for col in col_lower):
        add("knowledge_cleaning", 2, "dataset columns suggest document/url ingestion")

    if any(col in {"instruction", "response", "answer"} for col in col_lower):
        add("text_sft", 2, "dataset columns resemble SFT-format data")

    if any(col in {"code", "completion"} for col in col_lower):
        add("code", 2, "dataset columns resemble code-format data")

    # --- fallback ---
    if not score_map:
        add("core_text", 1, "default starting point for generic text pipelines")
        add("general_text", 1, "secondary fallback for deterministic cleanup/filtering")

    ranked = sorted(score_map.items(), key=lambda item: (-item[1], item[0]))
    chosen = ranked[:max_categories]

    recommended = [
        {
            "category": category,
            "score": score,
            "reason": "; ".join(reason_map.get(category, [])[:2]) or "closest available match",
        }
        for category, score in chosen
    ]

    avoid_ranked = sorted(avoid_map.items(), key=lambda item: item[0])
    avoid_items = [
        {
            "category": category,
            "score": 0,
            "reason": "; ".join(reasons[:2]),
        }
        for category, reasons in avoid_ranked
        if category not in {item["category"] for item in recommended}
    ]

    return {
        "recommended_categories": recommended,
        "avoid_categories": avoid_items[:3],
        "next_action": (
            "Call list_operators only for the recommended categories, then inspect "
            "specific candidates with get_operator_detail before use. Do not browse "
            "more than 2 categories unless the first pass clearly fails."
        ),
        "query_budget": {
            "max_category_queries": max_categories,
            "why": "Keep MCP context bounded and avoid broad catalog exploration.",
        },
    }


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

@tool(
    "list_operator_categories",
    "List all DataFlow operator categories with use_for/not_for guidance, examples, and live operator counts.",
    {},
)
async def list_operator_categories(args: dict[str, Any]) -> dict[str, Any]:
    categories = []
    for cat_name in sorted(CATEGORY_GUIDE.keys()):
        guide = CATEGORY_GUIDE[cat_name]
        ops = _catalog.get(cat_name, [])
        categories.append({
            "name": cat_name,
            "operator_count": len(ops),
            "use_for": guide["use_for"],
            "not_for": guide["not_for"],
            "examples": guide["examples"],
            "import_prefix": f"dataflow.operators.{cat_name}",
        })
    return {"content": [{"type": "text", "text": json.dumps(categories, indent=2, ensure_ascii=False)}]}


@tool(
    "recommend_operator_categories",
    "Given a task description (and optional dataset columns), recommend the top 2 most relevant operator categories with avoid-list and query budget.",
    {
        "task_description": Annotated[str, "Description of the data engineering task"],
        "dataset_columns": Annotated[str, "Comma-separated dataset column names (optional)"],
    },
)
async def recommend_operator_categories(args: dict[str, Any]) -> dict[str, Any]:
    task_description = args.get("task_description", "")
    if not task_description:
        return {"content": [{"type": "text", "text": "Error: task_description is required"}], "is_error": True}

    columns_str = args.get("dataset_columns", "")
    columns = [c.strip() for c in columns_str.split(",") if c.strip()] if columns_str else []

    result = _recommend_categories(task_description, columns)
    return {"content": [{"type": "text", "text": json.dumps(result, indent=2, ensure_ascii=False)}]}


@tool(
    "list_operators",
    "List all operators in a given category with name, subcategory, description, and allowed_prompts.",
    {"category": Annotated[str, "Category name (e.g., 'reasoning', 'text_sft', 'core_text')"]},
)
async def list_operators(args: dict[str, Any]) -> dict[str, Any]:
    category = args.get("category", "")
    if not category:
        return {"content": [{"type": "text", "text": "Error: category is required"}], "is_error": True}

    if category not in _catalog:
        available = sorted(_catalog.keys())
        return {"content": [{"type": "text", "text": f"Error: unknown category '{category}'. Available: {available}"}], "is_error": True}

    ops = _catalog[category]
    result = []
    for op in ops:
        result.append({
            "name": op["name"],
            "subcategory": op["subcategory"],
            "description": op["description"],
            "allowed_prompts": op["allowed_prompts"],
        })

    return {"content": [{"type": "text", "text": json.dumps(result, indent=2, ensure_ascii=False)}]}


@tool(
    "get_operator_detail",
    "Get detailed information about a specific operator by name, including full constructor and run() parameter signatures with defaults.",
    {"name": Annotated[str, "Operator class name (e.g., 'ReasoningAnswerGenerator')"]},
)
async def get_operator_detail(args: dict[str, Any]) -> dict[str, Any]:
    op_name = args.get("name", "")
    if not op_name:
        return {"content": [{"type": "text", "text": "Error: name is required"}], "is_error": True}

    for cat_name, ops in _catalog.items():
        for op in ops:
            if op["name"] == op_name:
                detail = {
                    "name": op["name"],
                    "category": op["category"],
                    "subcategory": op["subcategory"],
                    "description": op["description"],
                    "import_path": f"from dataflow.operators.{op['category']} import {op['name']}",
                    "allowed_prompts": op["allowed_prompts"],
                    "init_params": op["init_params"],
                    "run_params": op["run_params"],
                }
                return {"content": [{"type": "text", "text": json.dumps(detail, indent=2, ensure_ascii=False)}]}

    return {"content": [{"type": "text", "text": f"Error: operator '{op_name}' not found in any category"}], "is_error": True}


# ---------------------------------------------------------------------------
# Exported server config
# ---------------------------------------------------------------------------

operator_mcp_server = create_sdk_mcp_server(
    name="dataflow-operators",
    version="2.0.0",
    tools=[
        list_operator_categories,
        recommend_operator_categories,
        list_operators,
        get_operator_detail,
    ],
)
