"""DownstreamTrainerWrapper：外部训练/评估脚本的子进程桥接。

I/O 契约（TODO.md §5.6）：
  框架调用： python <train_eval_cmd> --dataset <ds.jsonl> --output <out_dir>
  脚本回写： <out_dir>/results.json = {score, metric_name, bad_cases:[{input,prediction,ground_truth,error_type}]}
"""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path

from dataflow_evolver.core import train_set
from dataflow_evolver.core.schemas import BadCase, TrainResult
from dataflow_evolver.utils.logging import get_logger

logger = get_logger("engine.trainer")


class TrainerWrapperABC(ABC):
    """下游训练/评估桥接接口。系统级失败必须降级为 TrainResult(failed=True)，不抛出。"""

    @abstractmethod
    def run(self, dataset_path: str, output_dir: str) -> TrainResult:
        """对 `dataset_path` 运行外部训练脚本，解析 results.json 为 TrainResult。"""
        raise NotImplementedError


class CappedTrainer(TrainerWrapperABC):
    """在数据进训练脚本前切齐到固定条数，再交给内层 trainer。

    做成一层包装而不是在调用点各切一次：搜索期打分与终选 Tier-2 是两条独立的调用路径，
    任何一条漏掉切齐，那批节点就在不同算力预算下打分，而这种偏差不会报错。包在
    `self.trainer` 外面，则"凡是训练都受约束"由构造处一次保证。

    切齐的理由与它管不到的那一半（产出不足），见 `core/train_set`。
    """

    def __init__(self, inner: TrainerWrapperABC, cap: int, seed: int = train_set.DEFAULT_SEED):
        self.inner = inner
        self.cap = int(cap)
        self.seed = int(seed)

    def run(self, dataset_path: str, output_dir: str) -> TrainResult:
        ts = train_set.materialize(dataset_path, output_dir, self.cap, self.seed)
        return self.inner.run(ts.path, output_dir)


class DownstreamTrainerWrapper(TrainerWrapperABC):
    """子进程调用用户提供的"训练+评估"脚本。

    复用 DataMaster 的 timeout/非零退出捕获模式：任何系统级失败都返回
    TrainResult(failed=True, ...)，绝不把异常抛到主编排器。
    """

    def __init__(
        self,
        train_eval_cmd: str,
        timeout_sec: int = 1800,
        python_path: str | None = None,
        eval_python_path: str | None = None,
        extra_args: list[str] | None = None,
    ) -> None:
        self.cmd_parts = _split_cmd(train_eval_cmd)
        # 允许配置独立的 train 环境 python（解决 agent/train 依赖冲突）
        if python_path and self.cmd_parts and self.cmd_parts[0] in ("python", "python3"):
            self.cmd_parts[0] = python_path
        self.timeout_sec = timeout_sec
        # 编排器决定、需逐次传给脚本的参数（如 --dev-set）。放在 --dataset/--output 之后，
        # 使脚本侧的位置约定不受影响。
        self.extra_args = list(extra_args or [])
        self.launch_env: dict[str, str] = {}
        if eval_python_path:
            self.launch_env["SFT_EVAL_PYTHON"] = eval_python_path

    def run(self, dataset_path: str, output_dir: str) -> TrainResult:
        out_dir = Path(output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        results_path = out_dir / "results.json"
        log_path = out_dir / "train.log"
        run_id = _make_run_id(out_dir)

        cmd = [
            *self.cmd_parts,
            "--dataset", str(Path(dataset_path).resolve()),
            "--output", str(out_dir),
            *self.extra_args,
        ]
        logger.info("启动下游训练：%s", " ".join(cmd))

        # 使用 Popen 以便在超时时杀死整个进程树（包括孙进程）
        try:
            env = {**os.environ, **self.launch_env, "DF_TRAIN_RUN_ID": run_id}
            _write_run_manifest(out_dir, cmd, str(Path(dataset_path).resolve()), env)
            if sys.platform == "win32":
                # Windows: 创建新进程组
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                )
            else:
                # Unix: 使用 setsid 创建新会话
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                    preexec_fn=os.setsid,
                )
        except FileNotFoundError as exc:
            msg = f"训练脚本不可执行：{exc}"
            logger.error(msg)
            return TrainResult(0.0, "unknown", [], str(log_path), True, msg)

        # 等待进程完成或超时
        try:
            stdout, stderr = proc.communicate(timeout=self.timeout_sec)
        except subprocess.TimeoutExpired:
            logger.warning("训练脚本超时（>%ds），正在终止进程树...", self.timeout_sec)
            # 杀死整个进程树
            _kill_process_tree(proc.pid)
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
            msg = f"训练脚本超时（>{self.timeout_sec}s）"
            logger.error(msg)
            return TrainResult(0.0, "unknown", [], str(log_path), True, msg)

        log_path.write_text(
            (stdout or "") + "\n--- STDERR ---\n" + (stderr or ""),
            encoding="utf-8",
        )

        if proc.returncode != 0:
            msg = f"训练脚本非零退出（{proc.returncode}）"
            logger.error("%s；详见 %s", msg, log_path)
            return TrainResult(0.0, "unknown", [], str(log_path), True, msg)

        if not results_path.exists():
            msg = f"训练脚本未回写 results.json：{results_path}"
            logger.error(msg)
            return TrainResult(0.0, "unknown", [], str(log_path), True, msg)

        return _parse_results(results_path, log_path)


def _looks_like_python(token: str) -> bool:
    """判断命令首 token 是否为 python 解释器（"python"/"python3" 或解释器路径）。"""
    base = Path(token).name.lower()
    return base in ("python", "python3") or base.startswith("python")


def _split_cmd(cmd: str) -> list[str]:
    """跨平台地把命令行拆成 token。

    Windows 上 `shlex.split` 的 POSIX 模式会把路径反斜杠当转义吃掉，故用 posix=False
    并手动剥离 token 两端的成对引号。
    """
    if os.name == "nt":
        parts = shlex.split(cmd, posix=False)
        return [
            p[1:-1] if len(p) >= 2 and p[0] == p[-1] and p[0] in "\"'" else p
            for p in parts
        ]
    return shlex.split(cmd)


def _make_run_id(output_dir: Path) -> str:
    return f"{int(time.time() * 1000)}-{os.getpid()}-{output_dir.name}"


# 训练配置以命令行 + 环境变量的形式散在框架外，节点分数却全靠它可比。跑到一半改了 YAML
# 或动了脚本，前后节点就在不同训练配方下打分，而树里看不出任何差别。这些前缀覆盖训练规模、
# 基座、评测并行度等会改变分数的量。
_ENV_PREFIXES = ("SFT_", "EVAL_", "MATH_", "PTB_", "DF_")


def _write_run_manifest(
    out_dir: Path, cmd: list[str], dataset_path: str, env: dict[str, str]
) -> None:
    """把这次训练的配方落一份 `run_manifest.json`，供事后核验跨节点可比性。

    尽力而为：留痕失败不该影响训练本身，只告警。
    """
    try:
        manifest = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "cmd": cmd,
            "dataset": dataset_path,
            "env": {
                k: v for k, v in sorted(env.items()) if k.startswith(_ENV_PREFIXES)
            },
            "scripts": _fingerprint_scripts(cmd),
        }
        (out_dir / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError as exc:
        logger.warning("run_manifest 落盘失败（不影响训练）：%s", exc)


def _fingerprint_scripts(cmd: list[str]) -> dict[str, str]:
    """对命令里出现的 .py 文件取内容摘要——脚本被改动后同一条命令不再等价。"""
    out: dict[str, str] = {}
    for token in cmd:
        if not token.endswith(".py"):
            continue
        p = Path(token)
        if not p.is_file():
            continue
        try:
            out[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        except OSError:
            continue
    return out


def _parse_results(results_path: Path, log_path: Path) -> TrainResult:
    try:
        data = json.loads(results_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"results.json 解析失败：{exc}"
        logger.error(msg)
        return TrainResult(0.0, "unknown", [], str(log_path), True, msg)

    bad_cases = [
        BadCase(
            input=str(bc.get("input", "")),
            prediction=str(bc.get("prediction", "")),
            ground_truth=str(bc.get("ground_truth", "")),
            error_type=bc.get("error_type"),
            extracted_answer=(
                None if bc.get("extracted_answer") is None
                else str(bc.get("extracted_answer"))
            ),
        )
        for bc in data.get("bad_cases", [])
    ]
    result = TrainResult(
        score=float(data.get("score", 0.0)),
        metric_name=str(data.get("metric_name", "score")),
        bad_cases=bad_cases,
        raw_log_path=str(log_path),
        failed=False,
        error_msg=None,
    )
    logger.info("训练完成：%s=%.4f，bad_cases=%d", result.metric_name, result.score, len(bad_cases))
    return result


def _kill_process_tree(pid: int) -> None:
    """跨平台地杀死进程及其所有子孙进程。"""
    if sys.platform == "win32":
        # Windows: 使用 taskkill /T 杀死进程树
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
                timeout=10,
            )
        except Exception as e:
            logger.warning("无法杀死 Windows 进程树 PID=%d: %s", pid, e)
    else:
        # Unix: 杀死整个进程组
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            time.sleep(2)
            # 如果进程仍存活，发送 SIGKILL
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except ProcessLookupError:
                pass  # 进程已终止
        except ProcessLookupError:
            pass  # 进程已不存在
        except Exception as e:
            logger.warning("无法杀死 Unix 进程组 PID=%d: %s", pid, e)
