"""DownstreamTrainerWrapper：外部训练/评估脚本的子进程桥接。

I/O 契约（TODO.md §5.6）：
  框架调用： python <train_eval_cmd> --dataset <ds.jsonl> --output <out_dir>
  脚本回写： <out_dir>/results.json = {score, metric_name, bad_cases:[{input,prediction,ground_truth,error_type}]}
"""
from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path

from dataflow_evolve.core.schemas import BadCase, TrainResult
from dataflow_evolve.utils.logging import get_logger

logger = get_logger("engine.trainer")


class TrainerWrapperABC(ABC):
    """下游训练/评估桥接接口。系统级失败必须降级为 TrainResult(failed=True)，不抛出。"""

    @abstractmethod
    def run(self, dataset_path: str, output_dir: str) -> TrainResult:
        """对 `dataset_path` 运行外部训练脚本，解析 results.json 为 TrainResult。"""
        raise NotImplementedError


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
    ) -> None:
        self.cmd_parts = _split_cmd(train_eval_cmd)
        # 允许配置独立的 train 环境 python（解决 agent/train 依赖冲突）
        if python_path and self.cmd_parts and self.cmd_parts[0] in ("python", "python3"):
            self.cmd_parts[0] = python_path
        self.timeout_sec = timeout_sec

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
        ]
        logger.info("启动下游训练：%s", " ".join(cmd))

        # 使用 Popen 以便在超时时杀死整个进程树（包括孙进程）
        try:
            env = {**os.environ, "DF_TRAIN_RUN_ID": run_id}
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


class SSHTrainerWrapper(TrainerWrapperABC):
    """双机训练：框架只在 node0 启动一次脚本，脚本内部负责拉起 node1 worker。

    前提：两机共享文件系统（dataset/output 路径两机一致）+ 免密 SSH。框架只做：
      1. 在 node0 本机启动一次训练脚本；
      2. 注入分布式与远端信息（NODE1_HOST / MASTER_ADDR / NNODES / GPU 数等）；
      3. 阻塞等待脚本完成并解析 results.json。

    具体的 convert_dataset / node1 SSH 拉起 / rank0 训练 / rank0 评测全部在脚本内部完成，
    框架本身只产出 pipeline 和数据集路径。
    """

    def __init__(
        self,
        train_eval_cmd: str,
        *,
        node1_host: str,
        master_addr: str,
        gpus_per_node: int,
        nnodes: int = 2,
        master_port: int = 29500,
        remote_cwd: str | None = None,
        ssh_opts: str = "",
        timeout_sec: int = 25200,
        python_path: str | None = None,
    ) -> None:
        self.cmd_parts = _split_cmd(train_eval_cmd)
        # rank1 远端解释器与 rank0 一致（默认沿用 python_path 或本机 python）
        self.python_exe = python_path or sys.executable
        if self.cmd_parts and _looks_like_python(self.cmd_parts[0]):
            self.cmd_parts[0] = self.python_exe
        self.node1_host = node1_host
        self.master_addr = master_addr
        self.gpus_per_node = gpus_per_node
        self.nnodes = nnodes
        self.master_port = master_port
        self.remote_cwd = remote_cwd or os.getcwd()
        self.ssh_opts = ssh_opts
        self.timeout_sec = timeout_sec

    def _launcher_env(self, run_id: str) -> dict[str, str]:
        """构造 node0 launcher 环境。真正的 node1 拉起由脚本内部完成。"""
        return {
            "NNODES": str(self.nnodes),
            "NODE_RANK": "0",
            "MASTER_ADDR": self.master_addr,
            "MASTER_PORT": str(self.master_port),
            "SFT_NUM_GPUS": str(self.gpus_per_node * self.nnodes),
            "FORCE_TORCHRUN": "1",
            "DF_TRAIN_RUN_ID": run_id,
            "NODE1_HOST": self.node1_host,
            "REMOTE_CWD": self.remote_cwd,
            "SSH_OPTS": self.ssh_opts,
            "SFT_REMOTE_PYTHON": self.python_exe,
        }

    def run(self, dataset_path: str, output_dir: str) -> TrainResult:
        out_dir = Path(output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        results_path = out_dir / "results.json"
        log_path = out_dir / "train.log"
        ds_abs = str(Path(dataset_path).resolve())
        tail = ["--dataset", ds_abs, "--output", str(out_dir)]
        run_id = _make_run_id(out_dir)

        # 只在 node0 启动一次脚本；脚本内部负责 convert + SSH 拉 node1 + rank0/rank1 训练 + rank0 评测。
        try:
            rc, stdout, stderr, timed_out = self._run_local(tail, timeout=self.timeout_sec, run_id=run_id)
        except FileNotFoundError as exc:
            msg = f"训练脚本不可执行：{exc}"
            logger.error(msg)
            return TrainResult(0.0, "unknown", [], str(log_path), True, msg)

        log_path.write_text(
            (stdout or "") + "\n--- STDERR ---\n" + (stderr or ""), encoding="utf-8"
        )

        if timed_out:
            msg = f"训练超时（>{self.timeout_sec}s）"
            logger.error(msg)
            return TrainResult(0.0, "unknown", [], str(log_path), True, msg)
        if rc != 0:
            msg = f"rank0 训练非零退出（{rc}）"
            logger.error("%s；详见 %s", msg, log_path)
            return TrainResult(0.0, "unknown", [], str(log_path), True, msg)
        if not results_path.exists():
            msg = f"训练脚本未回写 results.json：{results_path}"
            logger.error(msg)
            return TrainResult(0.0, "unknown", [], str(log_path), True, msg)
        return _parse_results(results_path, log_path)

    def _run_local(self, tail: list[str], timeout: int, run_id: str) -> tuple[int, str, str, bool]:
        """本机拉起 node0 launcher，阻塞等待。返回 (returncode, stdout, stderr, timed_out)。"""
        env = {**os.environ, **self._launcher_env(run_id=run_id)}
        cmd = [*self.cmd_parts, *tail]
        logger.info("本机拉起训练脚本（node0 launcher）：%s", " ".join(cmd))
        if sys.platform == "win32":
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, env=env,
                                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        else:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, env=env, preexec_fn=os.setsid)
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            return proc.returncode, stdout, stderr, False
        except subprocess.TimeoutExpired:
            logger.warning("node0 launcher 超时（>%ds），终止进程树...", timeout)
            _kill_process_tree(proc.pid)
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
            return proc.returncode, stdout, stderr, True


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
