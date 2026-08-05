"""ExecutionWrapper：运行 PipelineAgent 产出的 DataFlow 流水线代码并捕获崩溃栈。"""
from __future__ import annotations

import subprocess
import sys
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from dataflow_evolver.core.schemas import PipelineConfig
from dataflow_evolver.utils.logging import get_logger

logger = get_logger("engine.execution")


@dataclass
class ExecutionResult:
    """流水线执行结果。"""

    success: bool
    dataset_path: str | None = None        # 成功时：生成的数据集 jsonl 路径
    traceback: str | None = None           # 失败时：崩溃栈，喂回 PipelineAgent 自修正
    stdout: str = ""
    stderr: str = ""
    artifacts: dict[str, str] = field(default_factory=dict)


class ExecutionWrapperABC(ABC):
    """流水线执行器接口。"""

    @abstractmethod
    def run(
        self,
        config: PipelineConfig,
        node_dir: str,
        env_overrides: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """在 `node_dir` 下执行 `config.code`，返回数据集路径或崩溃栈。

        `env_overrides` adds or overrides variables for this iteration, such as the
        optional ``DF_TARGET_SAMPLES`` value.
        """
        raise NotImplementedError


class SubprocessExecutionWrapper(ExecutionWrapperABC):
    """在隔离子进程里运行生成的 pipeline.py，捕获崩溃栈。

    约定：PipelineAgent 已把 `pipeline.py` 写到 `node_dir`（位于 agent_root 下）。
    子进程 cwd 设为 `agent_root`（使 `import dataflow` 与相对路径生效）。
    成功后扫描 `node_dir` 下执行期间新增/更新的 .jsonl，取最新者作为产出数据集。
    """

    def __init__(
        self,
        agent_root: str,
        pipeline_filename: str = "pipeline.py",
        timeout_sec: int = 1800,
        python_exe: str | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        self.agent_root = Path(agent_root).resolve()
        self.pipeline_filename = pipeline_filename
        self.timeout_sec = timeout_sec
        self.python_exe = python_exe or sys.executable
        self.extra_env = extra_env or {}

    def run(
        self,
        config: PipelineConfig,
        node_dir: str,
        env_overrides: dict[str, str] | None = None,
    ) -> ExecutionResult:
        node_path = (self.agent_root / node_dir).resolve()
        pipeline_path = node_path / self.pipeline_filename
        if not pipeline_path.exists():
            return ExecutionResult(False, traceback=f"流水线文件不存在：{pipeline_path}")

        import os
        env = {**os.environ, **self.extra_env}
        # `python <node_dir>/pipeline.py` 的 sys.path[0] 是**脚本所在目录**（node_dir），
        # cwd=agent_root 并不会进 sys.path，因此 vendored 的 agent_root/dataflow 导入不到。
        # 不补这一项，每个节点首跑必定 ModuleNotFoundError('dataflow')，只能靠自纠回合补
        # sys.path，白烧一次 Agent 往返与一次自纠额度。
        prev_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(
            [str(self.agent_root), *(p for p in prev_pp.split(os.pathsep) if p)]
        )
        if env_overrides:
            env.update(env_overrides)
        start = time.time()
        try:
            proc = subprocess.run(
                [self.python_exe, str(pipeline_path)],
                cwd=str(self.agent_root),
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            logger.error("流水线执行超时（>%ds）：%s", self.timeout_sec, node_dir)
            return ExecutionResult(
                False, traceback=f"TimeoutExpired: 执行超过 {self.timeout_sec}s",
                stdout=exc.stdout or "", stderr=exc.stderr or "",
            )

        if proc.returncode != 0:
            logger.error("流水线执行失败（exit=%d）：%s", proc.returncode, node_dir)
            if proc.stderr:
                logger.error("stderr:\n%s", proc.stderr[-2000:])
            return ExecutionResult(
                False,
                traceback=proc.stderr or f"non-zero exit code {proc.returncode}",
                stdout=proc.stdout, stderr=proc.stderr,
            )

        dataset_path = _find_fresh_jsonl(node_path, since=start)
        if dataset_path is None:
            logger.warning("流水线执行成功但未发现输出 jsonl：%s", node_dir)
            return ExecutionResult(
                False, traceback="执行成功但未在 node_dir 找到输出 .jsonl 数据集",
                stdout=proc.stdout, stderr=proc.stderr,
            )
        logger.info("流水线产出数据集：%s", dataset_path)
        return ExecutionResult(
            True, dataset_path=str(dataset_path), stdout=proc.stdout, stderr=proc.stderr,
        )


_STEP_RE = re.compile(r"_step(\d+)\.jsonl$", re.IGNORECASE)


def _find_fresh_jsonl(node_path: Path, since: float) -> Path | None:
    """返回 node_path 下执行期间产生的输出 jsonl（流水线的最终产物）。

    DataFlow 的 FileStorage 把每个算子产物**确定性**命名为
    `{file_name_prefix}_step{N}.{cache_type}`（见 storage.py `_get_cache_file_path`），
    最后一个算子的 step 号最大。因此优先**按 step 号取最大的非空文件**来定位最终产物，
    而不是靠 mtime 猜（旧实现会被中间步/输入种子的写入时间干扰，见 review M2）。

    若最大 step 文件为空，则依次回退到更小 step 的非空文件；若完全没有 `_step{N}` 命名
    的文件（自定义产物名），再退回"最新非空 → 最新"的 mtime 启发式。
    """
    fresh = [
        p for p in node_path.rglob("*.jsonl")
        if p.is_file() and p.stat().st_mtime >= since - 1.0
    ]
    if not fresh:
        return None

    # 优先：按 _step{N} 命名取最大 step 的非空文件
    stepped = []
    for p in fresh:
        m = _STEP_RE.search(p.name)
        if m:
            stepped.append((int(m.group(1)), p))
    for _, p in sorted(stepped, key=lambda t: t[0], reverse=True):
        if _jsonl_nonempty(p):
            return p

    # 回退：无 step 命名 → 最新的非空；若全为空 → 最新文件（交由下游 Review 判其不合格）
    fresh.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for p in fresh:
        if _jsonl_nonempty(p):
            return p
    return fresh[0]


def _jsonl_nonempty(path: Path) -> bool:
    """是否至少有一行非空内容。"""
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    return True
    except OSError:
        return False
    return False
