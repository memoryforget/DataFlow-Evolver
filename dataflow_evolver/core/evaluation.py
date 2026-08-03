"""评分策略：一个节点的奖励从哪来，以及这一口径带来的连带纪律。

两种口径的差别不止"最后取哪个数"，而是一整套互相牵连的选择：

- `train`：搜索期就跑真实 SFT + dev 评测，奖励即下游真分。分数一步到位，但每个节点都要占
  GPU、按 target_samples 出全量数据，也因此不需要任何收尾兑现。
- `review`：搜索期只产 probe_samples 条探针数据交 ReviewAgent 打 Tier-1 复合分作奖励，
  不训练不占 GPU；代价是这个分只是**代理指标**，搜索结束必须对 top-K 放大重产并做真实
  SFT+评测把它兑现成真分（编排器的终选阶段），所以时间预算里要为收尾预留。

把这些绑在一起放进策略对象，是因为它们必须同时成立：注入了探针规模却按训练分打分，或者
按 Tier-1 打分却不做收尾兑现，都会得到一个看起来能跑、结论却不成立的运行。散落成若干处
`if reward_source == ...` 时，加第三种口径就意味着去每一处补一个分支，漏一处不会报错。

策略本身**无状态**，协作者（trainer/diagnostician/规模配置）通过 `ctx` 传入：编排器始终是
这些对象的唯一持有者，策略只表达"这一口径怎么做"。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol

from dataflow_evolver.core.tree_node import PipelineNode
from dataflow_evolver.utils.logging import get_logger

logger = get_logger("core.evaluation")


def samples_env(n: int) -> dict[str, str] | None:
    """产出规模的执行期环境变量（n<=0 = 不指定，交由 pipeline 内的缺省值决定）。

    生成的 pipeline.py 从 DF_TARGET_SAMPLES 读条数，故同一份代码可复用探针/全量两档规模，
    放大重产时无需再调一次 Agent SDK 重写代码。
    """
    return {"DF_TARGET_SAMPLES": str(int(n))} if n and n > 0 else None


class EvaluationContext(Protocol):
    """策略从编排器读取的协作者与配置（只读）。"""

    trainer: Any
    diagnostician: Any
    agent_root: Path
    probe_samples: int
    target_samples: int
    tier2_top_k: int
    tier2_per_run_sec: float
    final_eval_reserve_sec: float


class EvaluationStrategy(ABC):
    """把「奖励口径」的全部连带决定收在一处。"""

    name: str

    #: 搜索期的分数是否只是代理指标、需要在收尾阶段放大重产兑现成真分。
    needs_scale_up_selection: bool = False

    def search_env(self, ctx: EvaluationContext) -> dict[str, str] | None:
        """搜索期执行流水线时注入的环境变量。"""
        return None

    def default_prompt_samples(self, ctx: EvaluationContext) -> int:
        """写进生成 prompt 的「DF_TARGET_SAMPLES 缺省回退值」。

        脱离编排器单独运行 pipeline.py 时用的就是这个值，因此应当取该口径下的常规规模。
        """
        return ctx.target_samples

    def reserve_sec(self, ctx: EvaluationContext) -> float:
        """需要从总时间预算里为收尾阶段预留的墙钟秒数。"""
        return 0.0

    @abstractmethod
    def score(
        self, ctx: EvaluationContext, node: PipelineNode, profile, dataset_path: str
    ) -> float | None:
        """已通过评审的节点的最终得分；None = 该节点失败（由调用方判 dead 并回传惩罚）。"""


class TrainScore(EvaluationStrategy):
    """搜索期直接跑真实训练 + dev 评测，奖励就是下游真分。"""

    name = "train"

    def score(self, ctx, node, profile, dataset_path):
        train_out = str(Path(ctx.agent_root) / node.node_dir_name / "train_out")
        train = ctx.trainer.run(dataset_path, train_out)
        node.train_result = train
        if train.failed:
            logger.warning("训练系统级失败：%s，判定节点失败。", train.error_msg)
            return None
        # 诊断只在有真实 bad case 时才有内容可分析，因此绑定在 train 口径上。
        node.diagnostic_report = ctx.diagnostician.analyze(
            train.bad_cases, profile.downstream_task
        )
        logger.info("节点 %s 完成：dev=%.4f", node.node_dir_name, train.score)
        return train.score


class ReviewScore(EvaluationStrategy):
    """搜索期只用 Tier-1 复合分作代理奖励，不训练；真分留到收尾阶段兑现。"""

    name = "review"
    needs_scale_up_selection = True

    def search_env(self, ctx):
        return samples_env(ctx.probe_samples)

    def default_prompt_samples(self, ctx):
        # 探针规模：这一口径下单跑一次 pipeline 也该是便宜的。
        return ctx.probe_samples if ctx.probe_samples > 0 else ctx.target_samples

    def reserve_sec(self, ctx):
        return ctx.tier2_top_k * ctx.tier2_per_run_sec + ctx.final_eval_reserve_sec

    def score(self, ctx, node, profile, dataset_path):
        review = node.review_result
        logger.info(
            "节点 %s 完成（Tier-1）：tier1=%.4f（composite=%.3f, dup=%.2f null=%.2f parse=%.2f）",
            node.node_dir_name, review.tier1_score, review.llm_composite,
            review.dup_rate, review.null_rate, review.parse_valid_rate,
        )
        return review.tier1_score


STRATEGIES: dict[str, type[EvaluationStrategy]] = {
    TrainScore.name: TrainScore,
    ReviewScore.name: ReviewScore,
}


def build(reward_source: str) -> EvaluationStrategy:
    """按配置名建策略；名字不认识直接抛。

    拼错的 `reward_source` 若静默回退成默认口径，一次跑几小时的实验会安静地跑成另一种
    实验——只有对着日志逐行看才可能发现。宁可启动即失败。
    """
    key = (reward_source or "").strip().lower()
    try:
        return STRATEGIES[key]()
    except KeyError:
        raise ValueError(
            f"未知的 search.reward_source={reward_source!r}；可选：{sorted(STRATEGIES)}"
        ) from None
