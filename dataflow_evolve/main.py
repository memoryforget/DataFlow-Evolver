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
from pathlib import Path

from dataflow_evolve.agents.diagnostic_agent import DiagnosticAgent
from dataflow_evolve.agents.pipeline_agent import PipelineAgent
from dataflow_evolve.agents.review_agent import ReviewAgent
from dataflow_evolve.agents.task_profiler import TaskProfilerAgent
from dataflow_evolve.core import checkpoint
from dataflow_evolve.core.baseline import (
    REWARD_ABSOLUTE,
    REWARD_RELATIVE,
    RewardPolicy,
    adapt_baseline_source,
)
from dataflow_evolve.core.data_pool import DataPool, PoolConfig
from dataflow_evolve.core.schemas import PipelineConfig
from dataflow_evolve.core.tree_manager import TreeManager
from dataflow_evolve.core.tree_node import PipelineNode
from dataflow_evolve.engine.execution_wrapper import SubprocessExecutionWrapper
from dataflow_evolve.engine.trainer_wrapper import DownstreamTrainerWrapper
from dataflow_evolve.llm.serving import build_serving
from dataflow_evolve.utils.config import Config, load_config
from dataflow_evolve.utils.logging import get_logger, setup_logging

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
        self.reviewer = ReviewAgent(
            serving,
            schema_threshold=review_cfg.get("schema_threshold", 0.6),
            relevance_threshold=review_cfg.get("relevance_threshold", 0.6),
            sample_size=review_cfg.get("sample_size", 20),
        )
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

        # 目标产出规模（注入生成 prompt 指导算子产出量；0 或缺省=不指定）
        gen_cfg = cfg.get("generation")
        target_samples = int(gen_cfg.get("target_samples", 0)) if gen_cfg else 0
        self.pipeline_agent = PipelineAgent(
            agent_cfg=cfg.agent_sdk,
            agent_root=str(self.agent_root),
            entry_path=str(self.empty_entry_path),
            llm_block=llm_block,
            target_samples=target_samples,
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
            timeout_sec=cfg.trainer.get("timeout_sec", 1800),
            extra_env=self.pipeline_extra_env or None,
        )
        # dev 评测驱动 MCTS：trainer.dev_eval_cmd（见设计文档 §3/§6）
        dev_cmd = cfg.trainer.dev_eval_cmd
        train_python = cfg.trainer.get("train_python", None)
        logger.info("dev 评测命令：%s", dev_cmd)
        if train_python:
            logger.info("train 环境 python：%s", train_python)
        self.trainer = DownstreamTrainerWrapper(
            train_eval_cmd=dev_cmd,
            timeout_sec=cfg.trainer.get("timeout_sec", 1800),
            python_path=train_python,
        )
        # test 终评（全程对 MCTS loop 不可见，仅终评一次）
        test_cmd = cfg.trainer.get("test_eval_cmd", "")
        self.test_trainer = DownstreamTrainerWrapper(
            train_eval_cmd=test_cmd,
            timeout_sec=cfg.trainer.get("timeout_sec", 1800),
            python_path=train_python,
        ) if test_cmd else None

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
        self.pool: DataPool | None = DataPool(pool_config) if pool_config.enabled else None

        # 树搜索
        search = cfg.search
        self.tree = TreeManager(
            exploration_constant=search.get("exploration_constant", 1.414),
            max_children_per_node=search.get("max_children_per_node", 4),
            exploration_decay=search.get("exploration_decay", 1.0),
            exploration_min=search.get("exploration_min", 0.0),
        )
        self.max_iterations = search.get("max_iterations", 20)
        self.self_correction_retries = search.get("self_correction_retries", 2)
        self.review_self_correction_retries = search.get("review_self_correction_retries", 5)
        # 断点续跑（默认关闭）：非空路径则每轮落盘树状态，启动时若存在则载入续跑
        self.checkpoint_path = search.get("checkpoint", "")

    # ------------------------------------------------------------------
    def run(self, task_description: str) -> PipelineNode | None:
        # 0. 若有 checkpoint → 载入续跑；否则正常起步（含暖启动基线）
        state = checkpoint.read(self.checkpoint_path) if self.checkpoint_path else None
        if state:
            profile = checkpoint.restore_profile(state)
            checkpoint.restore_tree(state, self.tree)
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

        for it in range(start_it, self.max_iterations + 1):
            logger.info("==== 迭代 %d/%d ====", it, self.max_iterations)
            parent = self.tree.select()
            self._step(parent, profile)
            self._save_checkpoint(it, profile)

        if self.pool is not None:
            self.pool.dump()

        best = self.tree.best_leaf()
        if best is not None:
            logger.info(
                "最优节点：%s（mean_reward=%.4f）", best.node_dir_name, best.mean_reward
            )
        return best

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

        baseline_config = PipelineConfig(
            operators=[
                {"name": "CondorGenerator", "params": {"num_samples": 50, "prompt_template": "MathCondorPrompt"}},
                {"name": "ReasoningQuestionFilter", "params": {}},
                {"name": "ReasoningAnswerGenerator", "params": {}},
                {"name": "ReasoningAnswerFormatterFilter", "params": {}},
                {"name": "ReasoningAnswerTokenLengthFilter", "params": {"max_answer_token_length": 8192}},
                {"name": "ReasoningAnswerNgramFilter", "params": {"min_score": 0.1, "max_score": 1.0, "ngrams": 5}},
            ],
            field_flow="empty -> condor_generate(math) -> question_filter -> answer_generate(cot) -> format_filter -> token_filter -> ngram_filter",
            code=adapted,
            rationale="数学推理基线：CondorGenerator(math prompt) 从零合成 + reasoning filters 保质量 + CoT 答案生成",
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
    def _step(self, parent: PipelineNode, profile) -> None:
        # 预留子节点 → 拿到稳定的 node_dir 名
        child = self.tree.reserve_child(parent)
        node_dir = child.node_dir_name
        penalty = self.reward_policy.penalty

        # 3. 生成（沿用 parent 的诊断作为变异指引；若 parent 上轮被评审打回，把 issues 也喂回去；
        #    并把父节点/基线已验证的流水线代码作为变异起点注入）
        memory_summary = self.tree.global_memory.render()
        parent_review = parent.review_result
        review_issues = (
            parent_review.issues
            if parent_review is not None and not parent_review.passed
            else None
        )
        parent_code = parent.config.code if parent.config else None
        config = self.pipeline_agent.generate(
            profile=profile,
            node_dir_name=node_dir,
            diagnostic=parent.diagnostic_report,
            memory_summary=memory_summary,
            review_issues=review_issues,
            parent_code=parent_code,
        )
        if config is None:
            logger.warning("PipelineAgent 生成失败，惩罚该扩展并跳过。")
            child.dead = True
            self.tree.backpropagate(child, penalty)
            return
        child.config = config

        # 4. 执行（含自修正）
        exec_result = self.executor.run(config, node_dir)
        retries = 0
        while not exec_result.success and retries < self.self_correction_retries:
            retries += 1
            logger.info("执行崩溃，自修正第 %d/%d 次。", retries, self.self_correction_retries)
            fixed = self.pipeline_agent.self_correct(node_dir, exec_result.traceback or "")
            if fixed is None:
                break
            child.config = fixed
            exec_result = self.executor.run(fixed, node_dir)

        if not exec_result.success or exec_result.dataset_path is None:
            logger.warning("执行最终失败，惩罚节点。")
            child.dead = True
            self.tree.backpropagate(child, penalty)
            return

        # 5. Phase-1 快反馈（含立即自修正：失败则看中间 step 产物定位、修复、重审，限次）
        review = self.reviewer.review(exec_result.dataset_path, profile)
        child.review_result = review
        r_retries = 0
        while not review.passed and r_retries < self.review_self_correction_retries:
            r_retries += 1
            logger.info(
                "数据评审未通过，自修正第 %d/%d 次。issues=%s",
                r_retries, self.review_self_correction_retries, review.issues,
            )
            fixed = self.pipeline_agent.self_correct_review(node_dir, review.issues)
            if fixed is None:
                break
            child.config = fixed
            exec_result = self.executor.run(fixed, node_dir)
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
                exec_result = self.executor.run(exec_fixed, node_dir)
            if not exec_result.success or exec_result.dataset_path is None:
                logger.warning("评审自修正后执行最终失败，停止重审。")
                break
            review = self.reviewer.review(exec_result.dataset_path, profile)
            child.review_result = review
        if not review.passed:
            logger.info("数据评审最终未通过，不训练，惩罚节点。issues=%s", review.issues)
            child.dead = True
            self.tree.backpropagate(child, penalty)
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


if __name__ == "__main__":
    main()
