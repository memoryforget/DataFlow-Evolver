"""DataFlow-Evolve 主事件环：MCTS 驱动的双反馈环编排器（TODO.md §5.8）。

    1. TaskProfilerAgent → TaskProfile（一次）
    1b. Init node：产出第一条可跑通的流水线与第一个分数，作为树根（node_0）
    2. node = TreeManager.select()
    3. config = PipelineAgent.generate(profile, parent.diagnostic_report, global_memory, parent_code)
    4. dataset = ExecutionWrapper.run(config)
         └ 崩溃 → 把栈喂回 PipelineAgent 自修正（限次），仍失败则惩罚并回 2
    5. review = ReviewAgent.review(dataset)              # Phase-1 快反馈
         └ 未过 → 惩罚节点，不训练，回 2
    6. train = TrainerWrapper.run(dataset)               # Phase-2 真实训练（dev 评测，驱动 MCTS）
       report = DiagnosticAgent.analyze(train.bad_cases)
    7. TreeManager.backpropagate(node, reward=节点分数)；回到 2

奖励即节点分数本身（review 模式取 tier1 复合分，train 模式取 dev 真分），失败节点取
PENALTY_REWARD。Init node 与其余节点同口径评分、不享特殊地位，故无需任何奖励平移；
"相对初始流水线提升了多少"在报告层用 `node_score − node_0_score` 算即可。
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path

from dataflow_evolver.agents.diagnostic_agent import DiagnosticAgent
from dataflow_evolver.agents.pipeline_agent import PipelineAgent
from dataflow_evolver.agents.review_agent import ReviewAgent
from dataflow_evolver.agents.task_profiler import TaskProfilerAgent
from dataflow_evolver.core import (
    checkpoint,
    decontamination,
    dev_set,
    evaluation,
    input_corpus,
    provenance,
)
from dataflow_evolver.core.schemas import ReviewResult
from dataflow_evolver.core.tree_manager import TreeManager
from dataflow_evolver.core.tree_node import PipelineNode
from dataflow_evolver.engine.execution_wrapper import SubprocessExecutionWrapper
from dataflow_evolver.engine.trainer_wrapper import (
    CappedTrainer,
    DownstreamTrainerWrapper,
)
from dataflow_evolver.llm.agent_sdk import AgentInfraError
from dataflow_evolver.llm.serving import build_serving
from dataflow_evolver.utils.config import Config, load_config
from dataflow_evolver.utils.logging import get_logger, setup_logging

logger = get_logger("main")

# 失败节点（生成无产出 / 执行崩溃 / 评审不过 / 训练失败）回传的奖励。
# 两种奖励口径的分数都非负（tier1 复合分与 dev 准确率均 ∈[0,1]），故 0 恒为最差。
PENALTY_REWARD = 0.0


def _resolve_pipeline_llm(cfg_llm: Config) -> dict[str, dict]:
    """解析【pipeline 算子运行时】LLM 配置：`pipeline_llm` 段按用途区分多个 serving。

    格式：pipeline_llm.<purpose> 子段（如 generation / reasoning），
    每个用途省略的字段逐字段回退到 `api` 段。
    返回 {"purpose": {api_url, model_name, api_key, max_workers}}。
    """
    api = cfg_llm.api
    pl = cfg_llm.get("pipeline_llm")

    def _resolve_one(section) -> dict:
        def pick(key: str, default=None):
            if section is not None:
                v = section.get(key, None) if hasattr(section, "get") else None
                if v is not None:
                    return v
            return api.get(key, default)
        return {
            "api_url": pick("api_url"),
            "model_name": pick("model_name"),
            "api_key": pick("api_key", ""),
            "max_workers": pick("max_workers", 100),
        }

    if pl is None:
        return {"default": _resolve_one(None)}

    result = {}
    pl_data = pl._data if hasattr(pl, "_data") else pl
    for purpose, section_data in pl_data.items():
        section = Config(section_data) if isinstance(section_data, dict) else None
        result[purpose] = _resolve_one(section)
    return result


class Orchestrator:
    """组装所有组件并跑双反馈 MCTS 主循环。"""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        project = cfg.project
        self.workspace = Path(project.workspace_dir).resolve()
        self.agent_root = Path(project.agent_root).resolve()

        serving = build_serving(cfg.llm)

        # 辅助 Agent（非-SDK）
        self.profiler = TaskProfilerAgent(serving)
        review_cfg = cfg.review
        review_weights = _plain_dict(review_cfg.get("weights", None))
        # 去污染索引：未配 reference_files 时是个空索引，扫描恒返回干净（不改变任何行为）。
        decon_cfg = review_cfg.get("decontamination") or Config({})
        self.contamination = decontamination.build(
            reference_files=list(decon_cfg.get("reference_files", []) or []),
            n=int(decon_cfg.get("ngram", decontamination.DEFAULT_NGRAM)),
            max_rate=float(decon_cfg.get("max_rate", decontamination.DEFAULT_MAX_RATE)),
            fields=list(decon_cfg.get("fields", []) or []) or None,
        )
        self.reviewer = ReviewAgent(
            serving,
            contamination=self.contamination,
            schema_threshold=review_cfg.get("schema_threshold", 0.6),
            relevance_threshold=review_cfg.get("relevance_threshold", 0.6),
            sample_size=review_cfg.get("sample_size", 80),
            weights=review_weights,
            hard_metrics=bool(review_cfg.get("hard_metrics", True)),
            seed=review_cfg.get("sample_seed", None),
            null_threshold=float(review_cfg.get("null_threshold", 0.5)),
        )
        # 终选安全阀阈值：4000 放大重产后若质量塌陷则跳过用次优（见 _quality_collapsed）。
        self.collapse_dup = float(review_cfg.get("collapse_dup_rate", 0.5))
        self.collapse_null = float(review_cfg.get("collapse_null_rate", 0.3))
        self.collapse_parse = float(review_cfg.get("collapse_parse_valid_rate", 0.7))
        self.collapse_tier1_frac = float(review_cfg.get("collapse_tier1_frac", 0.6))
        self.diagnostician = DiagnosticAgent(
            serving, max_bad_cases=cfg.diagnostic.get("max_bad_cases", 30)
        )

        # 构建者（Agent SDK 驱动）——把【pipeline 算子运行时】LLM 配置注入生成的流水线。
        # 该 LLM 与辅助 Agent（Profiler/Review/Diagnostic，走上面 build_serving(cfg.llm) 的 api 段）
        # 解耦：算子做海量合成可用便宜模型，辅助 Agent 可用更强模型。见 _resolve_pipeline_llm。
        self.pipeline_llm = _resolve_pipeline_llm(cfg.llm)
        # 为每个用途分配独立的环境变量名注入密钥
        self.pipeline_env_keys: dict[str, str] = {}
        for purpose in self.pipeline_llm:
            self.pipeline_env_keys[purpose] = f"DF_PIPELINE_API_KEY_{purpose.upper()}"
        llm_block_lines = ["   按用途区分的 LLM Serving 配置（pipeline 中声明多个 serving 实例）："]
        for purpose, llm_cfg in self.pipeline_llm.items():
            env_key = self.pipeline_env_keys[purpose]
            llm_block_lines.append(
                f"   [{purpose}] model={llm_cfg['model_name']}, "
                f"api_url={llm_cfg['api_url']}, "
                f"key_name_of_api_key={env_key}, "
                f"max_workers={llm_cfg['max_workers']}"
            )
        llm_block_lines.append(
            "   说明：key_name_of_api_key 是【环境变量名】，运行时框架已在子进程注入该变量，"
            "流水线只需引用此名字，切勿写死密钥明文。"
            "每个算子根据功能绑定对应用途的 serving 实例。"
        )
        llm_block = "\n".join(llm_block_lines)
        # 输入契约：input/ 里有语料就走 corpus-grounded 合成（入口=语料），空则走从零合成
        # （入口=0 字节 empty.jsonl，仅满足 DataFlow 的 first_entry 形参）。见 core/input_corpus.py。
        # 同一份 corpus 供三处使用：pipeline prompt 的入口与预览、TaskProfiler 的样例、日志。
        empty_entry_path = self.agent_root / "empty.jsonl"
        empty_entry_path.parent.mkdir(parents=True, exist_ok=True)
        empty_entry_path.write_text("", encoding="utf-8")
        self.corpus = input_corpus.discover(
            project.get("input_dir", "") or "",
            empty_entry_path,
            entry_name=str(project.get("input_entry", "") or ""),
        )

        # 目标产出规模（注入生成 prompt 指导算子产出量；0 或缺省=不指定）。
        # 两档：probe_samples=搜索期探针小规模；target_samples=终选重产大规模。
        # 生成的 pipeline.py 从环境变量 DF_TARGET_SAMPLES 读取条数，故同一份代码可复用两档规模。
        gen_cfg = cfg.get("generation")
        self.target_samples = int(gen_cfg.get("target_samples", 0)) if gen_cfg else 0
        self.probe_samples = int(gen_cfg.get("probe_samples", 0)) if gen_cfg else 0
        # target_samples 对 pipeline 只是建议值（算子可以不读、过滤会少产、增强会多产），
        # 而训练配方冻结意味着总步数正比于条数——不切齐就等于让节点用数据量买算力。默认强制。
        self.enforce_target_samples = (
            bool(gen_cfg.get("enforce_target_samples", True)) if gen_cfg else True
        )

        # 奖励口径（train=搜索期真实训练取真分 / review=Tier-1 代理分 + 收尾兑现）。
        # 口径带来的连带纪律——执行规模、是否训练、是否需要放大重产兑现、预算预留——都由策略
        # 一处给出，见 core/evaluation.py。策略无状态，运行时从 self 读取协作者与规模配置。
        self.strategy = evaluation.build(cfg.search.get("reward_source", "train"))
        logger.info("奖励口径：%s", self.strategy.name)

        # 外部数据搜索（默认关闭）：开启后 Agent 可去 HuggingFace 找/取真实数据集，
        # 统一落到共享的 data_links/。是否允许引入外部数据属于实验设定，故须显式打开。
        # 惰性导入：这条链路会拉起 claude_agent_sdk 并注册 MCP server，关闭时不该有任何代价。
        self.data_links_dir = ""
        if cfg.agent_sdk.get("external_data", False):
            from dataflow_evolver.llm import mcp_datasets

            from dataflow_evolver.llm import mcp_discovery

            self.data_links_dir = str(
                mcp_datasets.configure(
                    project.get("data_links_dir", "") or self.workspace / "data_links",
                    cfg.get("dataset_search"),
                )
            )
            mcp_discovery.configure(cfg.get("discovery"))

        self.pipeline_agent = PipelineAgent(
            agent_cfg=cfg.agent_sdk,
            agent_root=str(self.agent_root),
            corpus=self.corpus,
            llm_block=llm_block,
            target_samples=self.strategy.default_prompt_samples(self),
            data_links_dir=self.data_links_dir,
        )

        # Init node 的启动方式（三档，对应 docs/design-decisions.md §5.2 的独立性消融）：
        #   给参考实现 → 忠实改写；只给指令 → 按指令构建；都不给 → 从零自主设计。
        # 参考实现在这里就读盘，配错路径要在启动时报错，而不是烧掉一次 Agent SDK 调用才发现。
        init_cfg = cfg.get("init") or Config({})
        self.init_instruction = str(init_cfg.get("instruction", "") or "")
        ref_path = str(init_cfg.get("reference_pipeline", "") or "")
        self.init_reference_code = _read_reference_pipeline(ref_path) if ref_path else None

        # 引擎——执行流水线的子进程里把【pipeline 算子 LLM】密钥注入约定的环境变量名
        self.pipeline_extra_env: dict[str, str] = {}
        for purpose, llm_cfg in self.pipeline_llm.items():
            env_key = self.pipeline_env_keys[purpose]
            if llm_cfg["api_key"]:
                self.pipeline_extra_env[env_key] = llm_cfg["api_key"]
        agent_pipeline_filename = cfg.agent_sdk.get("pipeline_filename", "pipeline.py")
        self.pipeline_filename = agent_pipeline_filename
        self.executor = SubprocessExecutionWrapper(
            agent_root=str(self.agent_root),
            pipeline_filename=agent_pipeline_filename,
            timeout_sec=cfg.trainer.timeout_sec,
            extra_env=self.pipeline_extra_env or None,
        )
        # dev 评测驱动 MCTS：trainer.dev_eval_cmd（见设计文档 §3/§6）
        dev_cmd = cfg.trainer.dev_eval_cmd
        # dev 题集由框架物化并显式下传，不再由评测脚本各自抽样：奖励是节点之间比出来的，
        # 「所有节点评在同一批题上」必须是事实而非约定（见 core/dev_set.py）。
        self.dev_set = dev_set.build(cfg.trainer.get("dev_set"), self.workspace)
        dev_set_args = ["--dev-set", self.dev_set.path] if self.dev_set else []
        if self.dev_set:
            # dev 是搜索直接优化的目标，把 dev 题混进合成数据是最省力的刷分路径——比混 test
            # 更有效，因为 test 全程不参与 loop。它必须在去污染参考集里，且不能指望用户记得配。
            self.contamination.add_file(self.dev_set.path)
        # 下游训练脚本实际读取的必需字段。它决定流水线至少该产出什么、评审按什么打分，是**契约**而非
        # 建议——不配则退回 TaskProfiler 让 LLM 现编字段名，而编出来的名字（problem/solution
        # 之类都很合理）与脚本的取字段逻辑对不上时，评审照样满分放行，直到训练阶段才炸，还会
        # 被记成「训练脚本非零退出」。
        self.dataset_schema = _plain_dict(cfg.trainer.get("dataset_schema"))
        if self.dataset_schema:
            logger.info("下游 schema 契约：%s", self.dataset_schema)
        else:
            logger.warning(
                "未配 trainer.dataset_schema：产出字段由 TaskProfiler 自行设计，"
                "与训练脚本对不上要到训练阶段才会暴露。"
            )
        train_python = cfg.trainer.get("train_python", None)
        eval_python = cfg.trainer.get("eval_python", None)
        train_timeout = cfg.trainer.train_timeout_sec
        logger.info("dev 评测命令：%s（超时 %ds）", dev_cmd, train_timeout)
        if train_python:
            logger.info("train 环境 python：%s", train_python)
        if eval_python:
            logger.info("eval 环境 python：%s", eval_python)

        def _make_trainer(cmd: str):
            if not cmd:
                return None
            inner = _make_bridge(cmd)
            if not (self.enforce_target_samples and self.target_samples > 0):
                return inner
            return CappedTrainer(inner, cap=self.target_samples)

        def _make_bridge(cmd: str):
            return DownstreamTrainerWrapper(
                train_eval_cmd=cmd,
                timeout_sec=train_timeout,
                python_path=train_python,
                eval_python_path=eval_python,
                extra_args=dev_set_args,
            )

        self.trainer = _make_trainer(dev_cmd)
        if isinstance(self.trainer, CappedTrainer):
            logger.info("训练集切齐：至多 %d 条/节点（同算力预算）", self.target_samples)
        elif dev_cmd:
            logger.warning(
                "训练集未切齐：节点间数据量不同将带来不同的总优化步数，分数不完全可比。"
            )
        self.dev_eval_cmd = dev_cmd
        self.train_python = train_python
        self.eval_python = eval_python
        self.train_timeout = train_timeout
        # test 终评命令：train 模式下沿用旧纪律（手动跑，不由编排器驱动）；review 模式下
        # _final_selection 选出 final_model 后由编排器**自动跑一次**官方全量评测（见 §终评）。
        # 缺省时由 dev_eval_cmd 把 "--mode dev" 换成 "--mode test" 派生。
        self.test_eval_cmd = cfg.trainer.get("test_eval_cmd", "") or dev_cmd.replace(
            "--mode dev", "--mode test"
        )

        # 树搜索
        search = cfg.search
        self.tree = TreeManager(
            exploration_constant=float(search.get("exploration_constant", 1.414)),
            max_children_per_node=int(search.get("max_children_per_node", 4)),
            exploration_decay=float(search.get("exploration_decay", 1.0)),
            exploration_min=float(search.get("exploration_min", 0.0)),
            memory_top_k=int(search.get("memory_top_k", 5)),
            memory_max_issues=int(search.get("memory_max_issues", 3)),
            memory_issue_maxlen=int(search.get("memory_issue_maxlen", 120)),
        )
        self.max_iterations = int(search.get("max_iterations", 10))
        # 预算感知（review 两段式专用；0=关闭，仅按 max_iterations 停）：
        #   time_budget_sec 为整轮（搜索+终选+终评）总墙钟预算。搜索会在预留出「终选+终评」
        #   所需时间后停止扩展；预留不足时 _final_selection 会自动缩小 K（极端 K=1）。
        self.time_budget_sec = float(search.get("time_budget_sec", 0) or 0)
        self.tier2_top_k = int(search.get("tier2_top_k", 3))
        # 单次 Tier-2（4000 重产 + 全量 SFT + dev 评测）的时间估计，用于预算预留与 K 缩减；
        # 缺省取 train_timeout（最坏情况）。final_eval_reserve_sec 为终评预留。
        self.tier2_per_run_sec = float(search.get("tier2_per_run_sec", train_timeout))
        self.final_eval_reserve_sec = float(
            search.get("final_eval_reserve_sec", train_timeout)
        )
        self.self_correction_retries = int(search.get("self_correction_retries", 5))
        self.review_self_correction_retries = int(search.get("review_self_correction_retries", 5))
        # 基础设施（Agent SDK）失败熔断：连续失败达到阈值则主动终止；每次失败退避后重试本轮
        self.max_consecutive_infra = int(search.get("max_consecutive_infra_failures", 5))
        self.infra_backoff_sec = float(search.get("infra_backoff_sec", 30))
        # 断点续跑（默认关闭）：非空路径则每轮落盘树状态，启动时若存在则载入续跑
        self.checkpoint_path = search.get("checkpoint", "")

    # ------------------------------------------------------------------
    def run(self, task_description: str) -> PipelineNode | None:
        self._run_start = time.time()
        self.final_result: dict | None = None
        # 0. 若有 checkpoint → 载入续跑；否则正常起步（Profiler + Init node）
        state = checkpoint.read(self.checkpoint_path) if self.checkpoint_path else None
        if state:
            checkpoint.verify_dev_set(
                state, self.dev_set.recipe() if self.dev_set else None
            )
            profile = checkpoint.restore_profile(state)
            # 续跑同样受契约约束：checkpoint 里的 schema 出自上一轮的配置，中途改了
            # dataset_schema 而不覆盖，续跑的节点会按旧字段产数据、评分，训练时才炸。
            if self.dataset_schema and profile.target_schema != self.dataset_schema:
                logger.warning(
                    "checkpoint 里的 schema 与当前 trainer.dataset_schema 不符，按配置覆盖：%s → %s",
                    profile.target_schema, self.dataset_schema,
                )
                profile.target_schema = dict(self.dataset_schema)
            checkpoint.restore_tree(
                state, self.tree,
                agent_root=str(self.agent_root),
                pipeline_filename=self.pipeline_filename,
            )
            start_it = state["completed_iterations"] + 1
            logger.info(
                "从 checkpoint 续跑：已完成 %d 轮，从第 %d 轮继续。",
                state["completed_iterations"], start_it,
            )
        else:
            profile = self.profiler.run(
                task_description,
                self.corpus.sample_text(),
                fixed_schema=self.dataset_schema,
            )
            if profile is None:
                logger.error("无法生成 TaskProfile，终止。")
                return None
            # 1b. Init node：整棵树的变异起点，跑不出分就没有可搜索的空间
            if not self._run_init_node(profile):
                return None
            start_it = 1

        it = start_it
        consecutive_infra = 0
        while it <= self.max_iterations:
            if self._search_should_stop():
                break
            logger.info("==== 迭代 %d/%d ====", it, self.max_iterations)
            parent = self.tree.select()
            status = self._step(parent, profile)

            if status == "infra_fatal":
                logger.error(
                    "Agent 基础设施发生不可重试错误（如 CLI 缺失/鉴权失败），主动终止。"
                    "已完成 %d 轮；请修复后重启。", it - 1,
                )
                break
            if status == "infra_retryable":
                consecutive_infra += 1
                if consecutive_infra >= self.max_consecutive_infra:
                    logger.error(
                        "连续 %d 次基础设施失败（代理/网络疑似不可用），主动终止。已完成 %d 轮；"
                        "请确认上游服务恢复后重启（可续跑）。",
                        consecutive_infra, it - 1,
                    )
                    break
                logger.warning(
                    "基础设施失败（连续第 %d/%d 次），本轮不计入迭代预算，%.0fs 后重试本轮。",
                    consecutive_infra, self.max_consecutive_infra, self.infra_backoff_sec,
                )
                time.sleep(self.infra_backoff_sec)
                continue

            # 正常完成（含被惩罚的 dead 节点）：计入预算、落盘 checkpoint、推进迭代
            consecutive_infra = 0
            self._save_checkpoint(it, profile)
            it += 1

        # 代理分口径：搜索结束后对 top-K 放大重产做真实 SFT+评测选模，并对 final_model 跑一次官方全量终评。
        if self.strategy.needs_scale_up_selection:
            self._final_selection(profile)
            # 终选把 tier2_review_result 与 train_result 回写到候选节点，而循环内最后一次落盘
            # 早于此。不在这里补写，这两份数据就只剩 run.log 里的日志行——而它们正是检验
            # 「tier1 能否预测下游真分」的唯一可比样本对，落不进结构化终态等于白留字段。
            self._save_checkpoint(it - 1, profile)

        best = self.tree.best_node()
        if best is not None:
            logger.info(
                "最优节点：%s（mean_reward=%.4f）", best.node_dir_name, best.mean_reward
            )
        return best

    # ------------------------------------------------------------------
    def _final_reserve_sec(self) -> float:
        """为搜索后的「终选 + 终评」预留的墙钟时间（只有需要兑现代理分的口径才预留）。"""
        return self.strategy.reserve_sec(self)

    def _search_should_stop(self) -> bool:
        """时间预算触发则停止扩展（预留终选+终评时间）。time_budget_sec<=0 时永不触发。"""
        if self.time_budget_sec <= 0:
            return False
        elapsed = time.time() - self._run_start
        reserve = self._final_reserve_sec()
        if elapsed >= self.time_budget_sec - reserve:
            logger.info(
                "时间预算触发：已用 %.0fs / 预算 %.0fs（预留终选+终评 %.0fs），停止扩展进入终选。",
                elapsed, self.time_budget_sec, reserve,
            )
            return True
        return False

    # ------------------------------------------------------------------
    def _save_checkpoint(self, completed_iterations: int, profile) -> None:
        """每完成一轮覆盖写一次 checkpoint（关闭时为 no-op）。"""
        if not self.checkpoint_path:
            return
        try:
            state = checkpoint.serialize(
                self.tree, completed_iterations, profile,
                dev_set_recipe=self.dev_set.recipe() if self.dev_set else None,
            )
            checkpoint.write(self.checkpoint_path, state)
        except Exception as exc:  # 落盘失败不应拖垮主循环
            logger.warning("checkpoint 落盘失败（继续跑）：%s", exc)

    # ------------------------------------------------------------------
    def _run_init_node(self, profile) -> bool:
        """Init node（node_0 = 树根）：产出第一条可跑通的流水线与第一个分数。

        它和其余节点走**完全同一条**评分路径（`_evaluate_node`），分数没有任何特殊地位——
        整棵树都从它变异而来，所以"相对初始流水线提升了多少"直接是 `node_score − root_score`，
        不需要奖励平移，也不需要单独的锚点模块。

        三种启动方式（对应独立性消融的三档，见 docs/design-decisions.md §5.2）：
        参考实现改写 / 只给自然语言指令 / 从零自主设计。均由 `init` 配置段决定。
        """
        root = self.tree.root
        node_dir = root.node_dir_name
        logger.info("==== Init node（%s）====", node_dir)
        config = self.pipeline_agent.generate_initial(
            profile=profile,
            node_dir_name=node_dir,
            reference_code=self.init_reference_code,
            instruction=self.init_instruction,
        )
        if config is None:
            logger.error("Init node 生成失败：Agent 跑完但未产出有效流水线。")
            return False
        root.config = config

        outcome = self._evaluate_node(root, profile)
        if outcome is None:
            logger.error("Init node 未能跑出分数；整棵树没有可变异的起点，终止。")
            return False

        # 根节点不经 backpropagate（它没有父链），直接落账：一次访问、一份奖励。
        root.visits = 1
        root.total_reward = outcome
        self.tree.global_memory.update(root)
        logger.info("Init node 完成：%s score=%.4f", node_dir, outcome)
        return True

    # ------------------------------------------------------------------
    def _step(self, parent: PipelineNode, profile) -> str:
        """跑一轮扩展并返回状态：
        - "ok"：正常完成（含被惩罚的 dead 节点），计入迭代预算；
        - "infra_retryable"：Agent 基础设施瞬时失败（429/超时/连接），撤销预留节点、不惩罚、不计入预算，主循环退避后重试本轮；
        - "infra_fatal"：Agent 基础设施永久失败（CLI 缺失/鉴权等），主循环终止运行。
        """
        # 预留子节点 → 拿到稳定的 node_dir 名
        child = self.tree.reserve_child(parent)
        node_dir = child.node_dir_name
        try:
            self._run_step(child, parent, profile)
            return "ok"
        except AgentInfraError as exc:
            logger.warning(
                "Agent 基础设施失败（retryable=%s）：%s；撤销预留节点 %s，不惩罚、不计入本轮迭代。",
                exc.retryable, exc, node_dir,
            )
            self.tree.discard_child(child)
            self._cleanup_empty_node_dir(node_dir)
            return "infra_retryable" if exc.retryable else "infra_fatal"

    def _review_evidence_path(self, node_dir_name: str) -> str:
        """节点评审证据的落盘位置（追加式 JSONL，每行一次评审）。

        结构化终态在 checkpoint、过程在 run.log，但**证据**（抽样与 LLM 原始响应）此前无处可寻，
        导致无法事后核验 tier1 是否可信。
        """
        return str(self.agent_root / node_dir_name / "review.jsonl")

    def _append_fix_history(
        self, node_dir_name: str, round_idx: int, targets: list[str], rendered: str
    ) -> None:
        """把一轮自修正的意图与渲染结果落盘到 `node_N/fix_history.jsonl`。

        review.jsonl 里有各轮的原始评审，逐轮 delta 可以事后重算；但"这一轮是针对哪几条 issue
        去改的"以及"prompt 里实际喂回去的那段文字"（受阈值影响）无法从中还原。落盘失败只告警。
        """
        record = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "round": round_idx,
            "targets": targets,
            "rendered": rendered,
        }
        path = self.agent_root / node_dir_name / "fix_history.jsonl"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("自修正历史落盘失败（不影响主流程）：%s", exc)

    def _cleanup_empty_node_dir(self, node_dir_name: str) -> None:
        """撤销基础设施失败的预留节点后，若其 workspace 目录为空则清理，避免留下空目录。"""
        node_dir_abs = self.agent_root / node_dir_name
        try:
            if node_dir_abs.is_dir() and not any(node_dir_abs.iterdir()):
                node_dir_abs.rmdir()
        except OSError:
            pass

    def _run_step(self, child: PipelineNode, parent: PipelineNode, profile) -> None:
        """一轮扩展的主体：生成 → 评估 → 回传。

        SDK 基础设施失败会以 AgentInfraError 向上抛出（由 _step 统一处理）；其余失败
        （生成无产出/执行崩溃/评审不过/训练失败）置 dead 并回传 PENALTY_REWARD。
        """
        node_dir = child.node_dir_name

        # 3. 生成。父节点信息分三路注入：已验证代码作变异起点、设计意图避免误改取舍、Tier-1 归因
        #    指明该攻哪一维；再加同父兄弟的负面清单促使分散探索。
        #    父节点与已列出的兄弟在 prompt 里都有专属块，故从全局记忆里排除：父是为了避免同一份
        #    issues 重复注入，兄弟是因为两处角色相反（记忆"参考它" vs 兄弟清单"别跟它一样"）。
        siblings = _sibling_summaries(parent, exclude=child)
        memory_summary = self.tree.global_memory.render(
            exclude_node_ids={parent.id, *(sid for sid, _ in siblings)}
        )
        config = self.pipeline_agent.generate(
            profile=profile,
            node_dir_name=node_dir,
            diagnostic=parent.diagnostic_report,
            memory_summary=memory_summary,
            parent_code=parent.config.code if parent.config else None,
            parent_rationale=parent.config.rationale if parent.config else None,
            parent_review=parent.review_result,
            parent_train_result=parent.train_result,
            siblings=[line for _, line in siblings],
            parent_recipe=parent.config.recipe if parent.config else None,
        )
        if config is None:
            logger.warning("PipelineAgent 生成失败（agent 跑完但未产出有效流水线），惩罚该扩展并跳过。")
            child.dead = True
            self.tree.backpropagate(child, PENALTY_REWARD)
            return
        child.config = config

        reward = self._evaluate_node(child, profile)
        self.tree.backpropagate(child, PENALTY_REWARD if reward is None else reward)

    def _evaluate_node(self, node: PipelineNode, profile) -> float | None:
        """对**已写好 config** 的节点求分：执行(自修正) → 评审(自修正) → 按模式取分。

        返回该节点的奖励；任一环节最终失败则置 `node.dead` 并返回 None（由调用方决定
        如何记账——子节点回传惩罚，Init node 则直接终止运行）。

        Init node 与变异节点共用本方法，是"初始流水线不享受特殊评分口径"这条设计的落点。

        无论成败都在收尾扫一遍外部数据溯源：失败节点用了什么数据同样是审计对象——"拉了不该拉
        的数据集导致污染扣分"恰恰只在失败节点上发生，而那正是最需要查清来源的时候。
        """
        try:
            return self._score_node(node, profile)
        finally:
            try:
                provenance.scan_node(
                    self.agent_root / node.node_dir_name, data_links_dir=self.data_links_dir
                )
            except Exception as exc:  # 旁路记录绝不能盖掉真实的返回值或异常
                logger.warning("外部数据溯源扫描失败（%s）：%s", node.node_dir_name, exc)

    def _score_node(self, node: PipelineNode, profile) -> float | None:
        """`_evaluate_node` 的主体，见其文档。"""
        node_dir = node.node_dir_name
        config = node.config
        assert config is not None, "_evaluate_node 要求节点已持有 config"
        # mode 不能只靠最终行自报来源：执行前先静态审计代码中的数据入口。synthetic 若引用
        # data_links/ 或直连远程数据立即拒绝；external/hybrid 禁止绕过 MCP 的 unmanaged 来源，
        # 且声明 external_dataset 时 pipeline 必须实际引用已审计的 data_links 路径。
        mode = config.recipe.get("mode") if config.recipe else None
        if mode in {"external", "hybrid", "synthetic"}:
            source_record = provenance.scan_node(
                self.agent_root / node_dir, data_links_dir=self.data_links_dir, write=False
            )
            source_issues = _mode_source_preflight_issues(
                config.recipe, corpus_grounded=self.corpus.grounded, source_record=source_record
            )
            if source_issues:
                logger.warning("%s mode 来源预检失败：%s", node_dir, source_issues)
                node.dead = True
                return None
        search_env = self.strategy.search_env(self)
        # 无 env 要注入时不传该参数：保持对旧执行器/测试桩的调用签名完全不变。
        exec_kwargs = {} if search_env is None else {"env_overrides": search_env}

        # 4. 执行（含自修正）
        exec_result = self.executor.run(config, node_dir, **exec_kwargs)
        retries = 0
        while not exec_result.success and retries < self.self_correction_retries:
            retries += 1
            logger.info("执行崩溃，自修正第 %d/%d 次。", retries, self.self_correction_retries)
            fixed = self.pipeline_agent.self_correct(node_dir, exec_result.traceback or "")
            if fixed is None:
                break
            node.config = fixed
            exec_result = self.executor.run(fixed, node_dir, **exec_kwargs)

        if not exec_result.success or exec_result.dataset_path is None:
            logger.warning("执行最终失败，判定节点失败。")
            node.dead = True
            return None

        # 5. Phase-1 快反馈（含立即自修正：失败则看中间 step 产物定位、修复、重审，限次）
        evidence_path = self._review_evidence_path(node_dir)
        review = self.reviewer.review(
            exec_result.dataset_path, profile, evidence_path=evidence_path, phase="search",
            recipe=node.config.recipe,
        )
        node.review_result = review
        r_retries = 0
        # 各轮修复的目标与成效。只把最新一轮 issues 喂回去，LLM 不知道某段逻辑是上一轮为修 X
        # 才加的，容易改掉它导致 X 复现，在节点内部震荡空转掉有限的自修正次数。
        fix_history: list[str] = []
        while not review.passed and r_retries < self.review_self_correction_retries:
            r_retries += 1
            logger.info(
                "数据评审未通过，自修正第 %d/%d 次。issues=%s",
                r_retries, self.review_self_correction_retries, review.issues,
            )
            fixed = self.pipeline_agent.self_correct_review(
                node_dir,
                review.issues,
                history=fix_history,
                failure_signals=review.failure_signals,
            )
            if fixed is None:
                break
            node.config = fixed
            exec_result = self.executor.run(fixed, node_dir, **exec_kwargs)
            # 评审自修正后执行崩溃：内嵌执行修复循环（复用 self_correction_retries 次数）
            exec_fix_retries = 0
            while not exec_result.success and exec_fix_retries < self.self_correction_retries:
                exec_fix_retries += 1
                logger.info(
                    "评审自修正后执行崩溃，执行修复第 %d/%d 次。",
                    exec_fix_retries, self.self_correction_retries,
                )
                exec_fixed = self.pipeline_agent.self_correct(node_dir, exec_result.traceback or "")
                if exec_fixed is None:
                    break
                node.config = exec_fixed
                exec_result = self.executor.run(exec_fixed, node_dir, **exec_kwargs)
            if not exec_result.success or exec_result.dataset_path is None:
                logger.warning("评审自修正后执行最终失败，停止重审。")
                break
            new_review = self.reviewer.review(
                exec_result.dataset_path,
                profile,
                evidence_path=evidence_path,
                phase=f"search_recheck_{r_retries}",
                recipe=node.config.recipe,
            )
            summary = _fix_round_summary(r_retries, review, new_review)
            fix_history.append(summary)
            self._append_fix_history(node_dir, r_retries, review.issues, summary)
            review = new_review
            node.review_result = review
        if not review.passed:
            logger.info("数据评审最终未通过，不训练，判定节点失败。issues=%s", review.issues)
            node.dead = True
            return None

        # 6. 按奖励口径取分：train 口径在此跑真实训练+诊断，review 口径直接用 Tier-1 复合分。
        score = self.strategy.score(self, node, profile, exec_result.dataset_path)
        if score is None:
            node.dead = True
        return score

    def _quality_collapsed(self, orig: "ReviewResult", new: "ReviewResult") -> bool:
        """大规模重产后的质量塌陷判定（安全阀）：命中任一即视为塌陷，跳过该候选。"""
        if not new.passed:
            return True
        if new.dup_rate > self.collapse_dup:
            return True
        if new.null_rate > self.collapse_null:
            return True
        if new.parse_valid_rate < self.collapse_parse:
            return True
        if (
            orig is not None and orig.tier1_score > 0
            and new.tier1_score < self.collapse_tier1_frac * orig.tier1_score
        ):
            return True
        return False

    def _final_selection(self, profile) -> None:
        """搜索结束后的 Tier-2 选模 + 终评（仅需要兑现代理分的口径会走到，见 EvaluationStrategy）。

        1) 按 tier1 取 top-K 候选；
        2) 各自用 target_samples（4000）重跑**同一** pipeline.py（env 注入，不重新生成代码）；
        3) 安全阀：重跑 ReviewAgent + 硬指标，质量塌陷则跳过用次优（backfill）；
        4) 对存活候选跑 dev_eval_cmd（单卡全量 SFT + 官方 evaluate.py --limit dev-limit），取真分最高者；
        5) 其 sft_model 落为 final_model/，再跑一次官方 evaluate.py --limit -1 终评。
        """
        if self.trainer is None:
            logger.warning("未配置 dev_eval_cmd（trainer 为空），跳过终选/终评。")
            return
        candidates = [
            n for n in self.tree.all_nodes()
            if (not n.dead) and n.review_result is not None
            and n.config is not None and n.visits > 0
        ]
        candidates.sort(key=lambda n: n.review_result.tier1_score, reverse=True)
        if not candidates:
            logger.warning("无可终选的候选节点。")
            return

        k = self.tier2_top_k
        if self.time_budget_sec > 0:
            remaining = (
                self.time_budget_sec - (time.time() - self._run_start)
                - self.final_eval_reserve_sec
            )
            affordable = max(1, int(remaining // max(self.tier2_per_run_sec, 1.0)))
            k = max(1, min(self.tier2_top_k, affordable))
        logger.info(
            "终选：%d 个候选，目标跑通 top-%d 个 Tier-2（4000 重产 + 单卡全量 SFT + dev 评测）。",
            len(candidates), k,
        )

        best: tuple[float, str, PipelineNode] | None = None
        successes = 0
        for node in candidates:
            if successes >= k:
                break
            if self.time_budget_sec > 0 and (
                (time.time() - self._run_start)
                >= self.time_budget_sec - self.final_eval_reserve_sec
            ):
                logger.info("预算耗尽，停止 Tier-2 尝试。")
                break

            node_dir = node.node_dir_name
            logger.info(
                "Tier-2 候选：%s（tier1=%.4f）", node_dir, node.review_result.tier1_score
            )
            # 1) 大规模重产同一 pipeline.py
            exec_result = self.executor.run(
                node.config, node_dir,
                env_overrides=evaluation.samples_env(self.target_samples),
            )
            if not exec_result.success or exec_result.dataset_path is None:
                logger.warning("候选 %s 放大重产失败，改用次优。", node_dir)
                continue
            # 2) 安全阀。重评结果回写节点：它与随后的 train.score 出自同一份放大数据，
            #    是事后检验「tier1 是否真能预测下游真分」的唯一可比样本对。
            re_review = self.reviewer.review(
                exec_result.dataset_path,
                profile,
                evidence_path=self._review_evidence_path(node_dir),
                phase="tier2",
                recipe=node.config.recipe,
            )
            node.tier2_review_result = re_review
            if self._quality_collapsed(node.review_result, re_review):
                logger.warning(
                    "候选 %s 放大后质量塌陷（dup=%.2f null=%.2f parse=%.2f tier1=%.3f passed=%s），改用次优。",
                    node_dir, re_review.dup_rate, re_review.null_rate,
                    re_review.parse_valid_rate, re_review.tier1_score, re_review.passed,
                )
                continue
            # 3) 单卡全量 SFT + dev 评测
            out_dir = str(self.agent_root / node_dir / "tier2_out")
            train = self.trainer.run(exec_result.dataset_path, out_dir)
            node.train_result = train
            if train.failed:
                logger.warning("候选 %s Tier-2 失败：%s，改用次优。", node_dir, train.error_msg)
                continue
            successes += 1
            sft_model = str(Path(out_dir) / "sft_model")
            logger.info("候选 %s Tier-2 dev=%.4f（%s）", node_dir, train.score, train.metric_name)
            if best is None or train.score > best[0]:
                best = (train.score, sft_model, node)

        if best is None:
            logger.error("终选未产出任何有效模型（候选全失败/塌陷）。")
            return
        dev_score, sft_model, node = best
        final_model = self._materialize_final_model(sft_model)
        logger.info(
            "终选完成：最优候选 %s，dev=%.4f，final_model=%s",
            node.node_dir_name, dev_score, final_model,
        )
        # 5) 终评：官方 evaluate.py --limit -1（只评测不训练；不计入搜索预算）
        test_score = self._run_final_eval(final_model)
        self.final_result = {
            "final_model": final_model,
            "best_node": node.node_dir_name,
            "dev_score": dev_score,
            "test_score": test_score,
            "metric_name": node.train_result.metric_name if node.train_result else "score",
        }

    def _materialize_final_model(self, sft_model: str) -> str:
        """把最优候选的 sft_model 落到 workspace/final_model（符号链接；失败则回退原路径）。"""
        target = self.workspace / "final_model"
        src = Path(sft_model)
        try:
            if target.is_symlink() or target.exists():
                if target.is_dir() and not target.is_symlink():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(src, target_is_directory=True)
            return str(target)
        except OSError as exc:
            logger.warning("创建 final_model 符号链接失败（%s），回退用原路径。", exc)
            return str(src)

    def _run_final_eval(self, final_model: str) -> float | None:
        """对 final_model 跑一次官方 evaluate.py --limit -1（复用桥接 --mode test，只评不训）。"""
        out_dir = self.workspace / "final_eval"
        out_dir.mkdir(parents=True, exist_ok=True)
        tokens = shlex.split(self.test_eval_cmd)
        if tokens and tokens[0] in ("python", "python3") and self.train_python:
            tokens[0] = self.train_python
        cmd = [*tokens, "--model-dir", str(final_model), "--output", str(out_dir)]
        env = {**os.environ, **self.pipeline_extra_env}
        if self.eval_python:
            env["SFT_EVAL_PYTHON"] = self.eval_python
        logger.info("终评命令：%s", " ".join(cmd))
        try:
            proc = subprocess.run(
                cmd, env=env, timeout=self.train_timeout,
                capture_output=True, text=True,
            )
        except subprocess.TimeoutExpired:
            logger.error("终评超时（>%ds）。", self.train_timeout)
            return None
        results_path = out_dir / "results.json"
        if proc.returncode != 0 or not results_path.exists():
            logger.error(
                "终评失败（rc=%s，results.json 存在=%s）。stderr 尾部：\n%s",
                proc.returncode, results_path.exists(), (proc.stderr or "")[-1500:],
            )
            return None
        try:
            data = json.loads(results_path.read_text(encoding="utf-8"))
            score = float(data.get("score", 0.0))
            logger.info("终评分数：%s=%.4f", data.get("metric_name", "score"), score)
            return score
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            logger.error("终评结果解析失败：%s", exc)
            return None


# 两类指标的可信度差一个量级，阈值必须分开：
# - 硬指标扫**全份文件**得出，两轮之间的 delta 是确定量，0.02 已足够滤掉浮点噪声；
# - 五维与 tier1 是 LLM 对两批**不同抽样**打的分，judge 在不同样本上本身就有 ±0.05~0.1 的
#   波动，阈值若取 0.02 等于不过滤，会把纯抽样噪声渲染成"多样性 0.20→0.60"，再配上因果语气
#   喂回 LLM，反而让它确信某个无效改动见效并在后续轮次里保护它——比什么都不给更糟。
_HARD_DELTA_MIN = 0.02
_LLM_DELTA_MIN = 0.10


def _fix_round_summary(
    round_idx: int, before: ReviewResult, after: ReviewResult, max_targets: int = 3
) -> str:
    """一轮评审自修正的「针对什么 → 观察到什么变化」摘要，喂回下一轮 prompt 作为锚。

    确定量与参考量分开表述，让 LLM 知道哪些 delta 能当证据、哪些不能。
    """
    targets = "；".join(before.issues[:max_targets]) or "（未给出具体条目）"
    hard_dims = (
        ("重复率", before.dup_rate, after.dup_rate),
        ("空值率", before.null_rate, after.null_rate),
        ("可解析率", before.parse_valid_rate, after.parse_valid_rate),
    )
    llm_dims = (
        ("正确性", before.correctness_score, after.correctness_score),
        ("相关性", before.relevance_score, after.relevance_score),
        ("难度", before.difficulty_score, after.difficulty_score),
        ("多样性", before.diversity_score, after.diversity_score),
        ("schema", before.schema_score, after.schema_score),
        ("Tier-1 复合分", before.tier1_score, after.tier1_score),
    )
    moved_hard = [f"{n} {b:.2f}→{a:.2f}" for n, b, a in hard_dims if abs(a - b) > _HARD_DELTA_MIN]
    moved_llm = [f"{n} {b:.2f}→{a:.2f}" for n, b, a in llm_dims if abs(a - b) > _LLM_DELTA_MIN]

    lines = [f"第 {round_idx} 轮针对：{targets}"]
    if moved_hard:
        lines.append("    确定变化（全量扫描，可据此判断改动是否见效）：" + "、".join(moved_hard))
    if moved_llm:
        lines.append(
            "    参考变化（两轮评审看的是不同抽样，judge 波动本身约 ±0.1，"
            "不能据此断定是哪处改动见效）：" + "、".join(moved_llm)
        )
    if not moved_hard and not moved_llm:
        lines.append("    未观察到超出噪声量级的变化")
    lines.append(f"    重评结论：{'通过' if after.passed else '仍未通过'}")
    return "\n".join(lines)


def _sibling_summaries(
    parent: PipelineNode, exclude: PipelineNode | None = None, max_items: int = 6
) -> list[tuple[int, str]]:
    """同父兄弟已试过的方向摘要（供 prompt 做负面清单），返回 (node_id, 摘要行)。

    同一父节点下要长出 max_children_per_node 个子节点，若每次 prompt 完全一致，多次昂贵扩展
    很可能收敛到同一种变异。这里给出"谁试过什么、得分多少"，让后续子节点另择方向。
    被 exclude 的是本轮正在生成的预留节点（它已在 parent.children 里但还没有内容）。
    连 node_id 一起返回，是为了让调用方按**实际渲染出的**兄弟去排除全局记忆里的重复条目——
    若两处各自判断该渲染谁，规则迟早分叉。
    """
    out: list[tuple[int, str]] = []
    for sib in parent.children:
        if sib is exclude or sib.config is None:
            continue
        ops = ", ".join(str(op.get("name", "?")) for op in sib.config.operators) or "未记录算子"
        parts = [f"{sib.node_dir_name}: ops=[{ops}]"]
        review = sib.review_result
        if review is not None:
            parts.append(f"tier1={review.tier1_score:.3f}")
        if sib.train_result is not None:
            parts.append(f"dev={sib.train_result.score:.3f}")
        if sib.dead:
            parts.append("已判失败")
        if sib.config.rationale:
            intent = sib.config.rationale.replace("\n", " ")
            parts.append(f"意图: {intent[:80]}{'…' if len(intent) > 80 else ''}")
        out.append((sib.id, " | ".join(parts)))
        if len(out) >= max_items:
            break
    return out


def _mode_source_preflight_issues(
    recipe: dict,
    *,
    corpus_grounded: bool,
    source_record: dict,
) -> list[str]:
    """用代码引用审计交叉验证 recipe.mode，阻止只在输出元数据里伪造来源。"""
    mode = recipe.get("mode")
    linked = source_record.get("data_links", [])
    unmanaged = bool(source_record.get("unmanaged"))
    declared_external = any(
        isinstance(src, dict) and src.get("kind") == "external_dataset"
        for src in recipe.get("sources", [])
    )
    issues: list[str] = []
    if mode == "synthetic" and (linked or unmanaged):
        issues.append("synthetic mode 的 pipeline 引用了真实/外部数据")
    if mode in {"external", "hybrid"} and unmanaged:
        issues.append("external/hybrid mode 检出绕过 External Data MCP 的 unmanaged 数据引用")
    if mode in {"external", "hybrid"} and declared_external and not linked:
        issues.append("recipe 声明了 external_dataset，但 pipeline 未引用任何已审计 data_links 文件")
    if mode in {"external", "hybrid"} and not corpus_grounded and not linked:
        issues.append("无输入语料的 external/hybrid mode 必须实际引用 External Data MCP 落盘文件")
    return issues


def _read_reference_pipeline(path: str) -> str:
    """读取 Init node 的参考实现源码。

    读不到直接抛：这是启动配置错误，静默降级成"从零自主设计"会让一次消融实验跑成
    另一档模式，而日志里只有一行 warning——事后无从分辨结果属于哪一档。
    """
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"init.reference_pipeline 指向的文件不存在：{p}")
    return p.read_text(encoding="utf-8")


def _override(cfg: Config, section: str, **values) -> None:
    """把非 None 的命令行值写回配置的某一段（None = 该项未指定，沿用 YAML）。"""
    provided = {k: v for k, v in values.items() if v is not None}
    if provided:
        cfg.to_dict().setdefault(section, {}).update(provided)


def _plain_dict(value) -> dict:
    """把配置里的嵌套段取成普通 dict；缺省/空值统一成 `{}`。

    `Config` 不能直接 `dict()`——它没有 `keys()`，`dict()` 会退化成整数索引并抛 `KeyError: 0`。
    而同一个键在测试里常常直接塞普通 dict，两种形态都得吃下。
    """
    if value is None:
        return {}
    return dict(value.to_dict() if hasattr(value, "to_dict") else value)


def main() -> None:
    parser = argparse.ArgumentParser(description="DataFlow-Evolve 双反馈 MCTS 主循环")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML 配置路径")
    parser.add_argument("--task", required=True, help="高层任务描述")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--init-code",
        default=None,
        help="Init node 的参考实现路径（覆盖 init.reference_pipeline）：给了就走忠实改写模式",
    )
    parser.add_argument(
        "--init-instruction",
        default=None,
        help="Init node 的构建指令（覆盖 init.instruction）",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="输入语料目录（覆盖 project.input_dir）；有语料=grounded 合成，空=从零合成",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    # 命令行覆盖：消融实验要在同一份 YAML 上换档（换启动模式、切输入范式），
    # 每次改配置文件既容易漏改，也让"这次跑的到底是哪档"不可追溯。
    _override(cfg, "init", reference_pipeline=args.init_code, instruction=args.init_instruction)
    _override(cfg, "project", input_dir=args.input_dir)
    log_file = Path(cfg.project.workspace_dir) / "run.log"
    setup_logging(level=args.log_level, log_file=log_file)

    orchestrator = Orchestrator(cfg)
    best = orchestrator.run(args.task)
    if best and best.config:
        print(f"\n最优流水线位于：{orchestrator.agent_root / best.node_dir_name}")
        print(f"field_flow: {best.config.field_flow}")
        print(f"mean_reward: {best.mean_reward:.4f}")
    else:
        print("\n未产出有效流水线。")

    final = getattr(orchestrator, "final_result", None)
    if final:
        print("\n==== PostTrainBench 终选/终评 ====")
        print(f"最优候选节点：{final['best_node']}")
        print(f"final_model：{final['final_model']}")
        print(f"dev 分（Tier-2 选模，metric={final['metric_name']}, --limit dev）：{final['dev_score']:.4f}")
        test_score = final.get("test_score")
        if test_score is not None:
            print(f"终评分（官方 evaluate.py --limit -1 全量）：{test_score:.4f}")
        else:
            print("终评分：未获得（终评失败，见日志）")
        # 口径提示：PTB 的 dev 与终评取自同一题池，--limit N 是确定性前缀，故 dev 题**包含在**
        # 终评题里。选模在终评集的子集上进行，终评分因此偏乐观——对外比较时必须一并说明。
        print(
            "注：dev 题集是终评题集的前缀子集（官方 evaluate.py 的 --limit 取确定性前 N 条），"
            "选模与终评并非独立样本，终评分偏乐观；见 docs/posttrainbench_adaptation.md。"
        )


if __name__ == "__main__":
    main()
