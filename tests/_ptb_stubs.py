"""PostTrainBench 适配测试用的轻量 stub 与 bare Orchestrator 工厂（非测试文件，不被收集）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataflow_evolver.core import evaluation, input_corpus
from dataflow_evolver.core.schemas import (
    DiagnosticReport,
    PipelineConfig,
    ReviewResult,
    TrainResult,
)
from dataflow_evolver.core.tree_manager import TreeManager
from dataflow_evolver.engine.execution_wrapper import ExecutionResult
from dataflow_evolver.main import Orchestrator


class StubPipelineAgent:
    def __init__(self, code="print('x')"):
        self.code = code
        self.generate_calls = 0

    def generate(self, **kwargs):
        self.generate_calls += 1
        return PipelineConfig(operators=[], field_flow="f", code=self.code, rationale="r")

    def generate_initial(self, **kwargs):
        self.generate_calls += 1
        return PipelineConfig(operators=[], field_flow="f", code=self.code, rationale="r")

    def self_correct(self, node_dir, tb):
        return None

    def self_correct_review(self, node_dir, issues, **kwargs):
        return None


class StubExecutor:
    """记录每次 run 的 env_overrides；产出一个真实存在的 jsonl 文件。"""

    def __init__(self, base_dir: Path, rows=1):
        self.base_dir = Path(base_dir)
        self.rows = rows
        self.calls = []  # list of (node_dir, env_overrides)

    def run(self, config, node_dir, env_overrides=None):
        self.calls.append((node_dir, env_overrides))
        self.base_dir.mkdir(parents=True, exist_ok=True)
        ds = self.base_dir / f"{node_dir}.jsonl"
        ds.write_text(
            "".join('{"instruction":"q%d","answer":"a%d"}\n' % (i, i) for i in range(self.rows)),
            encoding="utf-8",
        )
        return ExecutionResult(True, dataset_path=str(ds))


class StubReviewer:
    """按调用顺序返回预置 ReviewResult。"""

    def __init__(self, results):
        self._results = list(results)
        self._i = 0
        self.calls = 0

    def review(self, dataset_path, profile, **kwargs):
        self.calls += 1
        r = self._results[min(self._i, len(self._results) - 1)]
        self._i += 1
        return r


class StubTrainer:
    """按调用顺序返回预置 TrainResult；记录调用次数与收到的数据集路径。"""

    def __init__(self, results=None):
        self._results = list(results or [])
        self._i = 0
        self.calls = 0
        self.datasets = []  # 训练脚本实际拿到的数据集（断言切齐是否生效）

    def run(self, dataset_path, output_dir):
        self.calls += 1
        self.datasets.append(dataset_path)
        r = self._results[min(self._i, len(self._results) - 1)]
        self._i += 1
        return r


class StubDiagnostician:
    def __init__(self):
        self.calls = 0

    def analyze(self, bad_cases, downstream_task):
        self.calls += 1
        return DiagnosticReport(summary="ok")


def make_review_result(tier1=0.5, passed=True, dup=0.0, null=0.0, parse=1.0, composite=None):
    r = ReviewResult(schema_score=1.0, relevance_score=1.0, passed=passed)
    r.correctness_score = 1.0
    r.difficulty_score = 1.0
    r.diversity_score = 1.0
    r.llm_composite = composite if composite is not None else tier1
    r.dup_rate = dup
    r.null_rate = null
    r.parse_valid_rate = parse
    r.tier1_score = tier1
    return r


def bare_orchestrator(tmp_path, reward_source="review", **overrides):
    """构造一个只填必要属性的 Orchestrator（不触发真实 serving / SDK / 子进程）。

    `reward_source` 单独作参数而非 setattr 覆盖项：它要建出评分策略，而策略在运行时才从
    编排器读 trainer 等协作者，所以测试仍可在返回后自由替换 o.trainer / o.reviewer。
    """
    o = Orchestrator.__new__(Orchestrator)
    o.strategy = evaluation.build(reward_source)
    o.probe_samples = 200
    o.target_samples = 4000
    o.tree = TreeManager(max_children_per_node=4)
    o.self_correction_retries = 2
    o.review_self_correction_retries = 2
    o.agent_root = Path(tmp_path)
    o.workspace = Path(tmp_path)
    o.pipeline_agent = StubPipelineAgent()
    o.corpus = input_corpus.discover("", Path(tmp_path) / "empty.jsonl")
    o.init_reference_code = None
    o.init_instruction = ""
    o.dev_set = None                        # 未物化 dev 集：评测脚本沿用自身默认
    o.dataset_schema = {}                   # 未配下游契约：沿用 TaskProfiler 的推断
    o.data_links_dir = ""                   # 外部数据工具未挂载（溯源扫描仍会跑）
    o.executor = StubExecutor(Path(tmp_path) / "ds")
    o.reviewer = StubReviewer([make_review_result()])
    o.trainer = StubTrainer([TrainResult(score=0.5, metric_name="gsm8k")])
    o.diagnostician = StubDiagnostician()
    # 时间预算 / 终选相关
    o.time_budget_sec = 0.0
    o.tier2_top_k = 3
    o.tier2_per_run_sec = 100.0
    o.final_eval_reserve_sec = 100.0
    # 安全阀阈值
    o.collapse_dup = 0.5
    o.collapse_null = 0.3
    o.collapse_parse = 0.7
    o.collapse_tier1_frac = 0.6
    for k, v in overrides.items():
        setattr(o, k, v)
    return o
