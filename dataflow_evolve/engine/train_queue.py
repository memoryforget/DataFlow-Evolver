"""双机训练文件队列协议：编排器与常驻 worker 共享的目录布局、原子写、job spec 读写。

设计见 docs / 计划文件。核心不变量：MCTS 同步阻塞 → 任意时刻至多一个活跃训练任务，
故 claiming 逻辑极简（rank0 靠 os.replace 原子认领，无需分布式锁）。

目录布局（queue_dir 位于共享 FS）：
    train_queue/
    ├── pending/    <job_id>.json     编排器投递，rank0 认领前
    ├── running/    <job_id>.json     rank0 认领后（非 rank0 worker 等此标记再启动）
    │               <job_id>.cancel   编排器超时写入，worker 检测后杀训练进程树
    ├── done/       <job_id>.json     rank0 训练结束写完成标记 {status, results_path, error}
    ├── archive/    <job_id>.json     完成/作废的 job spec 归档，便于排查
    └── worker_state/ <rank>.seen     worker 已处理 job_id 列表（重启后跳过）
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

# 子目录名（编排器与 worker 必须一致）
PENDING = "pending"
RUNNING = "running"
DONE = "done"
ARCHIVE = "archive"
WORKER_STATE = "worker_state"

_SUBDIRS = (PENDING, RUNNING, DONE, ARCHIVE, WORKER_STATE)


def ensure_queue_dirs(queue_dir: str | Path) -> Path:
    """创建（幂等）队列根及所有子目录，返回解析后的根路径。"""
    root = Path(queue_dir).resolve()
    for sub in _SUBDIRS:
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def atomic_write_json(path: str | Path, obj: dict) -> None:
    """先写同目录 .tmp 再 os.replace 到目标名，保证读者不会看到半截 JSON。"""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + f".tmp.{uuid.uuid4().hex}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: str | Path) -> dict | None:
    """读 JSON；文件不存在或半截/损坏返回 None（供轮询容错）。"""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def new_job_id(prefix: str) -> str:
    """生成带时间戳的唯一 job_id，如 node_7-dev-1712345678901-ab12cd。"""
    import time

    return f"{prefix}-{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}"


def job_spec(
    *,
    job_id: str,
    cmd: list[str],
    dataset_path: str,
    output_dir: str,
    run_token: str,
) -> dict:
    """构造 job spec。cmd 为基础命令（不含 --dataset/--output，由 worker append）。"""
    import time

    return {
        "job_id": job_id,
        "cmd": cmd,
        "dataset_path": str(Path(dataset_path).resolve()),
        "output_dir": str(Path(output_dir).resolve()),
        "results_path": str((Path(output_dir).resolve() / "results.json")),
        "created_at": time.time(),
        "run_token": run_token,
    }
