"""DataFlow-Evolve 主事件环：MCTS 驱动的双反馈环编排器（TODO.md §5.8）。

    1. TaskProfilerAgent → TaskProfile（一次）
    1b.（暖启动 baseline.enabled=true）跑一次强基线得 baseline_dev_score，basline 作 root
    2. node = TreeManager.select()
    3. config = PipelineAgent.generate(profile, parent.diagnostic_report, global_memory, parent_code)
    4. dataset = ExecutionWrapper.run(config)
         └ 崩溃 → 把栈喂回 PipelineAgent 自修正（限次），仍失败则惩罚并回 2
    5. review = ReviewAgent.review(dataset)              # Phase-1 快反馈
         └ 未过 → 惩罚节点，不训练，回 2
    6. train = TrainerWrapper.run(dataset)               # Phase-2 真实训练（dev 评测，驱动 MCTS）
       report = DiagnosticAgent.analyze(train.bad_cases)
    7. TreeManager.backpropagate(node, reward=RewardPolicy.reward(train.score))
    8. （可选）DataPool.stage(当前节点数据)；回到 2。运行结束 dump() 统一筛/去重/裁剪

奖励/惩罚随 `baseline.enabled` 切换：
    - 冷启动（false）：虚拟空根 + 绝对分奖励 + penalty=0（行为与历史一致）。
    - 暖启动（true）：baseline 作 root + Relative Gain 奖励 + penalty=−baseline_dev（见 §2.8）。
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
from dataflow_evolver.core import checkpoint
from dataflow_evolver.core.baseline import (
    REWARD_ABSOLUTE,
    REWARD_RELATIVE,
    RewardPolicy,
    adapt_baseline_source,
    summarize_baseline_operators,
)
from dataflow_evolver.core.data_pool import DataPool, PoolConfig
from dataflow_evolver.core.schemas import PipelineConfig, ReviewResult
from dataflow_evolver.core.tree_manager import TreeManager
from dataflow_evolver.core.tree_node import PipelineNode
from dataflow_evolver.engine.execution_wrapper import SubprocessExecutionWrapper
from dataflow_evolver.engine.trainer_wrapper import DownstreamTrainerWrapper, SSHTrainerWrapper
from dataflow_evolver.llm.agent_sdk import AgentInfraError
from dataflow_evolver.llm.serving import build_serving
from dataflow_evolver.utils.config import Config, load_config
from dataflow_evolver.utils.logging import get_logger, setup_logging

logger = get_logger("main")


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
        self.seed_path = str(Path(project.seed_data).resolve())

        serving = build_serving(cfg.llm)

        # 辅助 Agent（非-SDK）
        self.profiler = TaskProfilerAgent(serving)
        review_cfg = cfg.review
        _w = review_cfg.get("weights", None)
        review_weights = _w.to_dict() if hasattr(_w, "to_dict") else _w
        self.reviewer = ReviewAgent(
            serving,
            schema_threshold=review_cfg.get("schema_threshold", 0.6),
            relevance_threshold=review_cfg.get("relevance_threshold", 0.6),
            sample_size=review_cfg.get("sample_size", 80),
            weights=review_weights,
            hard_metrics=bool(review_cfg.get("hard_metrics", True)),
            seed=review_cfg.get("sample_seed", None),
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
        # 从零合成设计：流水线不读输入语料，first_entry 指向一个共享的 0 字节 empty.jsonl
        # （仅满足 DataFlow 的 first_entry 形参）；seed.jsonl 只供 TaskProfiler 建模，不进流水线。
        self.empty_entry_path = self.agent_root / "empty.jsonl"
        self.empty_entry_path.parent.mkdir(parents=True, exist_ok=True)
        self.empty_entry_path.write_text("", encoding="utf-8")

        # reward 来源开关（向后兼容）：
        #   "train"（默认，数学 SFT 路径不变）：每节点跑 dev_eval_cmd（SFT+评测）取真分作 reward。
        #   "review"（PostTrainBench 两段式）：搜索期每节点只产 probe_samples 条交 ReviewAgent 打
        #            tier1 复合分作 reward（不训练、不占 GPU）；搜索结束再对 top-K 重产 target_samples
        #            条做真实 SFT+评测（见 _final_selection）。
        self.reward_source = str(cfg.search.get("reward_source", "train")).lower()

        # 目标产出规模（注入生成 prompt 指导算子产出量；0 或缺省=不指定）。
        # 两档：probe_samples=搜索期探针小规模；target_samples=终选重产大规模。
        # 生成的 pipeline.py 从环境变量 DF_TARGET_SAMPLES 读取条数，故同一份代码可复用两档规模。
        gen_cfg = cfg.get("generation")
        self.target_samples = int(gen_cfg.get("target_samples", 0)) if gen_cfg else 0
        self.probe_samples = int(gen_cfg.get("probe_samples", 0)) if gen_cfg else 0
        # 注入生成 prompt 的「env 缺省回退值」：review 模式用探针值（脱离编排器单跑也便宜），
        # train 模式沿用 target_samples（与旧行为一致）。
        prompt_default_samples = (
            self.probe_samples if (self.reward_source == "review" and self.probe_samples > 0)
            else self.target_samples
        )
        self.pipeline_agent = PipelineAgent(
            agent_cfg=cfg.agent_sdk,
            agent_root=str(self.agent_root),
            entry_path=str(self.empty_entry_path),
            llm_block=llm_block,
            target_samples=prompt_default_samples,
        )

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
        train_python = cfg.trainer.get("train_python", None)
        eval_python = cfg.trainer.get("eval_python", None)
        train_timeout = cfg.trainer.train_timeout_sec
        # 训练调度：local=本机拉起子进程（单机）；ssh=本机 rank0 + SSH 到 node1 拉起 rank1（双机）
        dispatch = cfg.trainer.get("dispatch", "local")
        ssh_cfg = cfg.trainer.get("ssh") if dispatch == "ssh" else None
        logger.info("dev 评测命令：%s（超时 %ds，dispatch=%s）", dev_cmd, train_timeout, dispatch)
        if train_python:
            logger.info("train 环境 python：%s", train_python)
        if eval_python:
            logger.info("eval 环境 python：%s", eval_python)

        def _make_trainer(cmd: str):
            if not cmd:
                return None
            if dispatch == "ssh":
                return SSHTrainerWrapper(
                    train_eval_cmd=cmd,
                    node1_host=ssh_cfg.node1_host,
                    master_addr=ssh_cfg.master_addr,
                    gpus_per_node=int(ssh_cfg.get("gpus_per_node", 8)),
                    nnodes=int(ssh_cfg.get("nnodes", 2)),
                    master_port=int(ssh_cfg.get("master_port", 29500)),
                    remote_cwd=ssh_cfg.get("remote_cwd", None) or None,
                    ssh_opts=ssh_cfg.get("ssh_opts", ""),
                    timeout_sec=train_timeout,
                    python_path=train_python,
                    eval_python_path=eval_python,
                )
            return DownstreamTrainerWrapper(
                train_eval_cmd=cmd,
                timeout_sec=train_timeout,
                python_path=train_python,
                eval_python_path=eval_python,
            )

        self.trainer = _make_trainer(dev_cmd)
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

        # 强基线锚点（warm/cold 开关，见 docs/design-decisions.md §2）
        baseline_cfg = cfg.get("baseline") or Config({})
        self.baseline_enabled = bool(baseline_cfg.get("enabled", False))
        self.baseline_margin = float(baseline_cfg.get("margin", 0.0))
        self.baseline_seed = int(baseline_cfg.get("seed", 42))
        self.baseline_pipeline_path = baseline_cfg.get(
            "baseline_pipeline",
            "workspace/_agent_root/dataflow/statics/pipelines/api_pipelines/reasoning_math_baseline_pipeline.py",
        )
        # 奖励策略：默认绝对分（冷启动）；暖启动跑完基线后切换为 Relative Gain。
        self.reward_policy = RewardPolicy(reward_mode=REWARD_ABSOLUTE)

        # 数据池（可选副产品，与奖励解耦；默认关闭）
        dp_cfg = cfg.get("data_pool")
        pool_config = PoolConfig.from_dict(dp_cfg.to_dict() if dp_cfg else None)
        # 行数据暂存到 workspace 下的稳定 sidecar 目录（跨进程续跑可复用），
        # 内存/checkpoint 只留元数据，避免整份数据集常驻/反复写盘（见 data_pool.StagedNode）。
        self.pool: DataPool | None = (
            DataPool(pool_config, staging_dir=str(self.workspace / "pool_staging"))
            if pool_config.enabled else None
        )

        # 树搜索
        search = cfg.search
        self.tree = TreeManager(
            exploration_constant=search.get("exploration_constant", 1.414),
            max_children_per_node=search.get("max_children_per_node", 4),
            exploration_decay=search.get("exploration_decay", 1.0),
            exploration_min=search.get("exploration_min", 0.0),
            memory_top_k=search.get("memory_top_k", 5),
            memory_max_issues=search.get("memory_max_issues", 3),
            memory_issue_maxlen=search.get("memory_issue_maxlen", 120),
        )
        self.max_iterations = search.get("max_iterations", 10)
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
        self.self_correction_retries = search.get("self_correction_retries", 5)
        self.review_self_correction_retries = search.get("review_self_correction_retries", 5)
        # 基础设施（Agent SDK）失败熔断：连续失败达到阈值则主动终止；每次失败退避后重试本轮
        self.max_consecutive_infra = int(search.get("max_consecutive_infra_failures", 5))
        self.infra_backoff_sec = float(search.get("infra_backoff_sec", 30))
        # 断点续跑（默认关闭）：非空路径则每轮落盘树状态，启动时若存在则载入续跑
        self.checkpoint_path = search.get("checkpoint", "")

    # ------------------------------------------------------------------
    def run(self, task_description: str) -> PipelineNode | None:
        self._run_start = time.time()
        self.final_result: dict | None = None
        # 0. 若有 checkpoint → 载入续跑；否则正常起步（含暖启动基线）
        state = checkpoint.read(self.checkpoint_path) if self.checkpoint_path else None
        if state:
            profile = checkpoint.restore_profile(state)
            checkpoint.restore_tree(
                state, self.tree,
                agent_root=str(self.agent_root),
                pipeline_filename=self.pipeline_filename,
            )
            self.reward_policy = checkpoint.restore_reward_policy(state)
            if self.pool is not None:
                checkpoint.restore_pool(state, self.pool)
            start_it = state["completed_iterations"] + 1
            logger.info(
                "从 checkpoint 续跑：已完成 %d 轮，从第 %d 轮继续（reward_mode=%s）",
                state["completed_iterations"], start_it, self.reward_policy.reward_mode,
            )
        else:
            profile = self.profiler.run(task_description, seed_path=self.seed_path)
            if profile is None:
                logger.error("无法生成 TaskProfile，终止。")
                return None
            # 1b. 暖启动：跑一次强基线，basline 作 root + 切换到 Relative Gain 奖励
            if self.baseline_enabled:
                if not self._run_baseline():
                    logger.error("基线流程失败，暖启动无法继续。终止。")
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

        if self.pool is not None:
            self.pool.dump()

        # review 两段式：搜索结束后对 top-K 做真实 SFT+评测选模，并对 final_model 跑一次官方全量终评。
        if self.reward_source == "review":
            self._final_selection(profile)
            # 终选把 tier2_review_result 与 train_result 回写到候选节点，而循环内最后一次落盘
            # 早于此。不在这里补写，这两份数据就只剩 run.log 里的日志行——而它们正是检验
            # 「tier1 能否预测下游真分」的唯一可比样本对，落不进结构化终态等于白留字段。
            self._save_checkpoint(it - 1, profile)

        best = self.tree.best_leaf()
        if best is not None:
            logger.info(
                "最优节点：%s（mean_reward=%.4f）", best.node_dir_name, best.mean_reward
            )
        return best

    # ------------------------------------------------------------------
    def _final_reserve_sec(self) -> float:
        """为搜索后的「终选 + 终评」预留的墙钟时间（review 模式才预留）。"""
        if self.reward_source != "review":
            return 0.0
        return self.tier2_top_k * self.tier2_per_run_sec + self.final_eval_reserve_sec

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
                self.tree, self.reward_policy, self.pool, completed_iterations, profile
            )
            checkpoint.write(self.checkpoint_path, state)
        except Exception as exc:  # 落盘失败不应拖垮主循环
            logger.warning("checkpoint 落盘失败（继续跑）：%s", exc)

    # ------------------------------------------------------------------
    def _run_baseline(self) -> bool:
        """暖启动专用：用 seed 固定的强基线流水线跑一次，得 baseline_dev_score 并作 root。

        基线源码硬编码了 openai.com/gpt-4o 与 empty.jsonl；这里以最小可行方式（正则替换）
        把它适配到代理 API（复用执行子进程注入的 DF_PIPELINE_API_KEY）、本地空种子与节点缓存。
        见 docs/design-decisions.md §2.2/§2.4。
        """
        src_path = Path(self.baseline_pipeline_path).resolve()
        if not src_path.exists():
            logger.error("基线流水线源码不存在：%s", src_path)
            return False

        root_dir_name = self.tree.root.node_dir_name
        node_dir_abs = self.agent_root / root_dir_name
        node_dir_abs.mkdir(parents=True, exist_ok=True)

        # 自带一个 0 字节空种子（Condor 从零合成，输入内容被忽略，仅需文件存在）
        empty_jsonl = node_dir_abs / "empty.jsonl"
        empty_jsonl.write_text("", encoding="utf-8")

        # 基线用 generation 用途的 LLM（或第一个可用的）
        baseline_purpose = "generation" if "generation" in self.pipeline_llm else next(iter(self.pipeline_llm))
        baseline_llm = self.pipeline_llm[baseline_purpose]
        baseline_env_key = self.pipeline_env_keys[baseline_purpose]

        adapted = adapt_baseline_source(
            src_path.read_text(encoding="utf-8"),
            api_url=baseline_llm["api_url"],
            model_name=baseline_llm["model_name"],
            key_env=baseline_env_key,
            empty_jsonl_path=str(empty_jsonl),
            cache_path=str(node_dir_abs / "cache"),
        )
        pipeline_path = node_dir_abs / self.pipeline_filename
        pipeline_path.write_text(adapted, encoding="utf-8")

        # 算子清单 / field_flow 从适配后的源码解析得出（而非手写），保证与实际执行一致：
        # 基线行为完全由 code 决定，产出规模等参数以源码为准（不再谎报 num_samples 等元数据）。
        operators, field_flow = summarize_baseline_operators(adapted)
        baseline_config = PipelineConfig(
            operators=operators,
            field_flow=field_flow or "(baseline pipeline; 详见 code)",
            code=adapted,
            rationale=(
                f"强基线：导入并适配自 {src_path.name}"
                "（算子/field_flow 由源码解析得出，与实际执行一致）。"
            ),
        )

        logger.info("跑强基线（seed=%d）：%s", self.baseline_seed, pipeline_path)
        exec_result = self.executor.run(baseline_config, root_dir_name)
        if not exec_result.success or exec_result.dataset_path is None:
            logger.error("基线流水线执行失败：%s", exec_result.traceback)
            return False

        train_out = str(node_dir_abs / "train_out")
        train = self.trainer.run(exec_result.dataset_path, train_out)
        if train.failed:
            logger.error("基线 dev 评测失败：%s", train.error_msg)
            return False

        baseline_dev = train.score
        logger.info("baseline_dev_score=%.4f", baseline_dev)

        # 切换奖励策略 → Relative Gain；basline 作 root；池门槛改为基线自标定（严格 > baseline+margin）
        self.reward_policy = RewardPolicy(
            reward_mode=REWARD_RELATIVE, baseline_dev_score=baseline_dev
        )
        root = self.tree.set_baseline_root(baseline_config, baseline_dev)
        root.train_result = train
        if self.pool is not None:
            self.pool.set_warm_gate(baseline_dev, self.baseline_margin)
            logger.info("数据池门槛（暖启动）：node_dev > %.4f", baseline_dev + self.baseline_margin)
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
        """一轮扩展的主体：生成→执行(自修正)→评审(自修正)→训练→诊断→回传。

        SDK 基础设施失败会以 AgentInfraError 向上抛出（由 _step 统一处理）；其余失败
        （生成无产出/执行崩溃/评审不过/训练失败）按既有逻辑置 dead 并回传 penalty。
        """
        node_dir = child.node_dir_name
        penalty = self.reward_policy.penalty
        # 搜索期执行环境：review 模式注入探针规模，让生成的 pipeline 只产 probe_samples 条（便宜）。
        search_env = self._gen_env(self.probe_samples) if self.reward_source == "review" else None
        # train 模式不注入任何 env（保持对旧执行器/测试桩的调用签名完全不变）。
        exec_kwargs = {} if search_env is None else {"env_overrides": search_env}

        # 3. 生成。父节点信息分三路注入：已验证代码作变异起点、设计意图避免误改取舍、Tier-1 归因
        #    指明该攻哪一维；再加同父兄弟的负面清单促使分散探索。
        #    父节点与已列出的兄弟在 prompt 里都有专属块，故从全局记忆里排除：父是为了避免同一份
        #    issues 重复注入，兄弟是因为两处角色相反（记忆"参考它" vs 兄弟清单"别跟它一样"）。
        siblings = _sibling_summaries(parent, exclude=child)
        memory_summary = self.tree.global_memory.render(
            exclude_node_ids={parent.id, *(sid for sid, _ in siblings)}
        )
        parent_review = parent.review_result
        parent_code = parent.config.code if parent.config else None
        parent_rationale = parent.config.rationale if parent.config else None
        config = self.pipeline_agent.generate(
            profile=profile,
            node_dir_name=node_dir,
            diagnostic=parent.diagnostic_report,
            memory_summary=memory_summary,
            parent_code=parent_code,
            parent_rationale=parent_rationale,
            parent_review=parent_review,
            siblings=[line for _, line in siblings],
        )
        if config is None:
            logger.warning("PipelineAgent 生成失败（agent 跑完但未产出有效流水线），惩罚该扩展并跳过。")
            child.dead = True
            self.tree.backpropagate(child, penalty)
            return
        child.config = config

        # 4. 执行（含自修正）
        exec_result = self.executor.run(config, node_dir, **exec_kwargs)
        retries = 0
        while not exec_result.success and retries < self.self_correction_retries:
            retries += 1
            logger.info("执行崩溃，自修正第 %d/%d 次。", retries, self.self_correction_retries)
            fixed = self.pipeline_agent.self_correct(node_dir, exec_result.traceback or "")
            if fixed is None:
                break
            child.config = fixed
            exec_result = self.executor.run(fixed, node_dir, **exec_kwargs)

        if not exec_result.success or exec_result.dataset_path is None:
            logger.warning("执行最终失败，惩罚节点。")
            child.dead = True
            self.tree.backpropagate(child, penalty)
            return

        # 5. Phase-1 快反馈（含立即自修正：失败则看中间 step 产物定位、修复、重审，限次）
        evidence_path = self._review_evidence_path(node_dir)
        review = self.reviewer.review(
            exec_result.dataset_path, profile, evidence_path=evidence_path, phase="search"
        )
        child.review_result = review
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
            child.config = fixed
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
                child.config = exec_fixed
                exec_result = self.executor.run(exec_fixed, node_dir, **exec_kwargs)
            if not exec_result.success or exec_result.dataset_path is None:
                logger.warning("评审自修正后执行最终失败，停止重审。")
                break
            new_review = self.reviewer.review(
                exec_result.dataset_path,
                profile,
                evidence_path=evidence_path,
                phase=f"search_recheck_{r_retries}",
            )
            summary = _fix_round_summary(r_retries, review, new_review)
            fix_history.append(summary)
            self._append_fix_history(node_dir, r_retries, review.issues, summary)
            review = new_review
            child.review_result = review
        if not review.passed:
            logger.info("数据评审最终未通过，不训练，惩罚节点。issues=%s", review.issues)
            child.dead = True
            self.tree.backpropagate(child, penalty)
            return

        # ---- reward_source=review：Tier-1 复合分即 reward，搜索期不训练/不诊断/不占 GPU ----
        if self.reward_source == "review":
            reward = review.tier1_score
            self.tree.backpropagate(child, reward)
            logger.info(
                "节点 %s 完成（Tier-1）：tier1=%.4f reward=%.4f（composite=%.3f, "
                "dup=%.2f null=%.2f parse=%.2f）",
                child.node_dir_name, review.tier1_score, reward, review.llm_composite,
                review.dup_rate, review.null_rate, review.parse_valid_rate,
            )
            return

        # 6. Phase-2 真实训练（dev 评测）+ 诊断
        train_out = str(self.agent_root / node_dir / "train_out")
        train = self.trainer.run(exec_result.dataset_path, train_out)
        child.train_result = train
        if train.failed:
            logger.warning("训练系统级失败：%s，惩罚节点。", train.error_msg)
            child.dead = True
            self.tree.backpropagate(child, penalty)
            return

        report = self.diagnostician.analyze(train.bad_cases, profile.downstream_task)
        child.diagnostic_report = report

        # 7. 回传奖励（RewardPolicy：绝对分 / Relative Gain 随模式切换）
        reward = self.reward_policy.reward(train.score)
        self.tree.backpropagate(child, reward)
        logger.info(
            "节点 %s 完成：dev=%.4f reward=%.4f（mode=%s）",
            child.node_dir_name, train.score, reward, self.reward_policy.reward_mode,
        )

        # 8. 数据池收割（可选）：运行期只暂存，门槛/去重/占比裁剪留到 run() 末尾 dump 统一处理
        if self.pool is not None:
            rows = _read_jsonl_rows(exec_result.dataset_path)
            self.pool.stage(
                node_id=child.id,
                rows=rows,
                dev_score=train.score,
                review_passed=review.passed,
            )


    # ------------------------------------------------------------------
    def _gen_env(self, n: int) -> dict[str, str] | None:
        """构造注入 pipeline 执行子进程的产出规模环境变量（n<=0 时不注入）。"""
        return {"DF_TARGET_SAMPLES": str(int(n))} if n and n > 0 else None

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
        """搜索结束后的 Tier-2 选模 + 终评（仅 reward_source=review）。

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
                node.config, node_dir, env_overrides=self._gen_env(self.target_samples)
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


def _read_jsonl_rows(path: str) -> list[dict]:
    """读取 jsonl 文件为 dict 列表（跳过空行/坏行），供数据池收割。"""
    rows: list[dict] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
    except OSError as exc:
        logger.warning("读取数据集失败（池收割跳过）：%s", exc)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="DataFlow-Evolve 双反馈 MCTS 主循环")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML 配置路径")
    parser.add_argument("--task", required=True, help="高层任务描述")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    cfg = load_config(args.config)
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


if __name__ == "__main__":
    main()
