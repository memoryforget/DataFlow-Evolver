"""DownstreamTrainerWrapper：外部训练/评估脚本的子进程桥接。

I/O 契约（TODO.md §5.6）：
  框架调用： python <train_eval_cmd> --dataset <ds.jsonl> --output <out_dir>
  脚本回写： <out_dir>/results.json = {score, metric_name, bad_cases:[{input,prediction,ground_truth,error_type}]}
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
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

    def __init__(self, train_eval_cmd: str, timeout_sec: int = 1800) -> None:
        self.cmd_parts = _split_cmd(train_eval_cmd)
        self.timeout_sec = timeout_sec

    def run(self, dataset_path: str, output_dir: str) -> TrainResult:
        out_dir = Path(output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        results_path = out_dir / "results.json"
        log_path = out_dir / "train.log"

        cmd = [
            *self.cmd_parts,
            "--dataset", str(Path(dataset_path).resolve()),
            "--output", str(out_dir),
        ]
        logger.info("启动下游训练：%s", " ".join(cmd))
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.timeout_sec
            )
        except subprocess.TimeoutExpired:
            msg = f"训练脚本超时（>{self.timeout_sec}s）"
            logger.error(msg)
            return TrainResult(0.0, "unknown", [], str(log_path), True, msg)
        except FileNotFoundError as exc:
            msg = f"训练脚本不可执行：{exc}"
            logger.error(msg)
            return TrainResult(0.0, "unknown", [], str(log_path), True, msg)

        log_path.write_text(
            (proc.stdout or "") + "\n--- STDERR ---\n" + (proc.stderr or ""),
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
