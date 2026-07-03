"""dual_node_worker.py：双机训练的常驻 worker 守护进程。

在参与训练的每台机器上各起一个（rank0 / rank1）。worker 轮询共享 FS 上的文件队列
（dataflow_evolve/engine/train_queue.py 协议），发现新训练任务后带正确的分布式 env
（NODE_RANK / MASTER_ADDR / NNODES / ...）拉起 job spec 里的训练命令，torchrun 便能
在两机间会合。rank0 负责认领任务（原子 rename pending→running）并在训练结束后写 done
标记；非 rank0 只等 running 标记出现即启动同一命令。

用法（两机各执行一次，MASTER_ADDR 填 rank0 机器的可路由 IP）：
    # node0（主）
    python scripts/dual_node_worker.py --rank 0 --nnodes 2 \
        --master-addr <IP0> --master-port 29500 --gpus 8 \
        --queue-dir <shared>/workspace/train_queue --python <train_env_python>
    # node1（从）
    python scripts/dual_node_worker.py --rank 1 --nnodes 2 \
        --master-addr <IP0> --master-port 29500 --gpus 8 \
        --queue-dir <shared>/workspace/train_queue --python <train_env_python>

单机自测（验证队列机制，不跑真 torchrun）：--rank 0 --nnodes 1 --gpus 0，配合
scripts/reference_train_eval.py 作为 job cmd 即可端到端跑通。
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# 允许以 `python scripts/dual_node_worker.py` 直接运行时能 import 包
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataflow_evolve.engine import train_queue  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [worker] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 已处理任务追踪（重启后跳过已完成 job）
# ---------------------------------------------------------------------------
def _seen_path(queue_root: Path, rank: int) -> Path:
    return queue_root / train_queue.WORKER_STATE / f"{rank}.seen"


def _load_seen(queue_root: Path, rank: int) -> set[str]:
    p = _seen_path(queue_root, rank)
    if not p.exists():
        return set()
    try:
        return {ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()}
    except OSError:
        return set()


def _mark_seen(queue_root: Path, rank: int, job_id: str) -> None:
    p = _seen_path(queue_root, rank)
    try:
        with p.open("a", encoding="utf-8") as f:
            f.write(job_id + "\n")
    except OSError as e:
        logger.warning("无法写入 seen 记录（继续）：%s", e)


# ---------------------------------------------------------------------------
# 进程树 kill（供任务超时 / 收到 cancel 标记时清理孙进程，避免僵尸 GPU 进程）
# ---------------------------------------------------------------------------
def _kill_process_tree(pid: int) -> None:
    if sys.platform == "win32":
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                           capture_output=True, timeout=10)
        except Exception as e:  # noqa: BLE001
            logger.warning("无法杀死 Windows 进程树 PID=%d: %s", pid, e)
    else:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            time.sleep(2)
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        except ProcessLookupError:
            pass
        except Exception as e:  # noqa: BLE001
            logger.warning("无法杀死 Unix 进程组 PID=%d: %s", pid, e)


# ---------------------------------------------------------------------------
# 分布式 env 构造
# ---------------------------------------------------------------------------
def _build_env(args: argparse.Namespace) -> dict[str, str]:
    """为训练子进程构造分布式 env。修掉 eval 脚本原本 NODE_RANK/MASTER_ADDR 缺失的 bug。

    llamafactory_math_sft_eval.py 内部 NPROC_PER_NODE = SFT_NUM_GPUS // NNODES，故
    SFT_NUM_GPUS 须传【集群总卡数】= 本机卡数 × nnodes。
    """
    env = os.environ.copy()
    env["NNODES"] = str(args.nnodes)
    env["NODE_RANK"] = str(args.rank)
    env["MASTER_ADDR"] = args.master_addr
    env["MASTER_PORT"] = str(args.master_port)
    env["FORCE_TORCHRUN"] = "1"
    if args.gpus > 0:
        env["SFT_NUM_GPUS"] = str(args.gpus * args.nnodes)
    return env


# ---------------------------------------------------------------------------
# 训练命令拼装与执行
# ---------------------------------------------------------------------------
def _build_cmd(spec: dict, python_exe: str) -> list[str]:
    """job spec 的基础命令 + --dataset/--output。基础命令首 token 为脚本路径。"""
    base = list(spec["cmd"])
    return [python_exe, *base, "--dataset", spec["dataset_path"], "--output", spec["output_dir"]]


def _run_training(spec: dict, args: argparse.Namespace, cancel_path: Path) -> tuple[bool, str]:
    """拉起训练子进程并等待。返回 (成功, 错误信息)。期间轮询 cancel 标记以便杀进程树。"""
    python_exe = args.python or sys.executable
    cmd = _build_cmd(spec, python_exe)
    env = _build_env(args)
    out_dir = Path(spec["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "worker_train.log"

    logger.info("拉起训练（rank=%d）：%s", args.rank, " ".join(cmd))
    try:
        if sys.platform == "win32":
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, env=env, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        else:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, env=env, preexec_fn=os.setsid)
    except FileNotFoundError as exc:
        return False, f"训练命令不可执行：{exc}"

    deadline = time.time() + args.job_timeout
    while True:
        try:
            proc.wait(timeout=args.poll_interval)
            break  # 进程结束
        except subprocess.TimeoutExpired:
            pass
        if cancel_path.exists():
            logger.warning("检测到 cancel 标记，终止训练进程树：%s", spec["job_id"])
            _kill_process_tree(proc.pid)
            return False, "编排器取消（超时）"
        if time.time() > deadline:
            logger.error("worker 侧任务超时（>%ds），终止：%s", args.job_timeout, spec["job_id"])
            _kill_process_tree(proc.pid)
            return False, f"worker 侧任务超时（>{args.job_timeout}s）"

    stdout = proc.stdout.read() if proc.stdout else ""
    try:
        log_path.write_text(stdout or "", encoding="utf-8")
    except OSError:
        pass
    if proc.returncode != 0:
        return False, f"训练命令非零退出（{proc.returncode}），详见 {log_path}"
    return True, ""


# ---------------------------------------------------------------------------
# 主 poll 循环
# ---------------------------------------------------------------------------
def _claim_rank0(queue_root: Path, job_id: str) -> dict | None:
    """rank0 原子认领 pending→running。返回 spec；认领失败（已被处理）返回 None。"""
    pending = queue_root / train_queue.PENDING / f"{job_id}.json"
    running = queue_root / train_queue.RUNNING / f"{job_id}.json"
    try:
        os.replace(pending, running)  # 原子；失败=已被认领或已消失
    except OSError:
        return None
    return train_queue.read_json(running)


def _poll_once(args: argparse.Namespace, queue_root: Path, seen: set[str]) -> bool:
    """扫一轮队列，处理至多一个任务。返回是否处理了任务。"""
    if args.rank == 0:
        pending_dir = queue_root / train_queue.PENDING
        candidates = sorted(pending_dir.glob("*.json"))
        for p in candidates:
            job_id = p.stem
            if job_id in seen:
                continue
            spec = _claim_rank0(queue_root, job_id)
            if spec is None:
                continue
            _process(args, queue_root, seen, job_id, spec)
            return True
    else:
        running_dir = queue_root / train_queue.RUNNING
        candidates = sorted(running_dir.glob("*.json"))
        for p in candidates:
            job_id = p.stem
            if job_id in seen:
                continue
            spec = train_queue.read_json(p)
            if spec is None:
                continue
            _process(args, queue_root, seen, job_id, spec)
            return True
    return False


def _process(args: argparse.Namespace, queue_root: Path, seen: set[str],
             job_id: str, spec: dict) -> None:
    """执行一个任务：拉起训练；rank0 结束后写 done 标记 + 归档。"""
    cancel_path = queue_root / train_queue.RUNNING / f"{job_id}.cancel"
    logger.info("开始处理任务 %s（rank=%d）", job_id, args.rank)
    ok, err = _run_training(spec, args, cancel_path)
    _mark_seen(queue_root, args.rank, job_id)
    seen.add(job_id)

    if args.rank != 0:
        logger.info("任务 %s 训练结束（rank=%d，非 rank0 不写 done）：ok=%s", job_id, args.rank, ok)
        return

    # rank0 独占写 done + 归档
    done_path = queue_root / train_queue.DONE / f"{job_id}.json"
    done = {
        "job_id": job_id,
        "status": "ok" if ok else "failed",
        "results_path": spec.get("results_path"),
        "error": None if ok else err,
        "at": time.time(),
    }
    train_queue.atomic_write_json(done_path, done)
    # 归档 running spec，清理 cancel 标记
    try:
        os.replace(queue_root / train_queue.RUNNING / f"{job_id}.json",
                   queue_root / train_queue.ARCHIVE / f"{job_id}.json")
    except OSError:
        pass
    try:
        cancel_path.unlink()
    except OSError:
        pass
    logger.info("任务 %s 完成，已写 done：status=%s", job_id, done["status"])


def main() -> None:
    parser = argparse.ArgumentParser(description="双机训练常驻 worker")
    parser.add_argument("--rank", type=int, required=True, help="NODE_RANK（0=主，写 done）")
    parser.add_argument("--nnodes", type=int, default=2)
    parser.add_argument("--master-addr", default="localhost", help="rank0 机器可路由 IP")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--gpus", type=int, default=0, help="本机 GPU 数（0=不设 SFT_NUM_GPUS）")
    parser.add_argument("--queue-dir", required=True, help="共享 FS 队列根目录")
    parser.add_argument("--python", default=None, help="训练环境解释器（默认当前）")
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--job-timeout", type=int, default=25200, help="单任务上限秒数（≥编排器 timeout）")
    args = parser.parse_args()

    queue_root = train_queue.ensure_queue_dirs(args.queue_dir)
    seen = _load_seen(queue_root, args.rank)
    logger.info("worker 启动：rank=%d nnodes=%d master=%s:%d queue=%s（已知 %d 个历史任务）",
                args.rank, args.nnodes, args.master_addr, args.master_port, queue_root, len(seen))

    while True:
        try:
            handled = _poll_once(args, queue_root, seen)
        except Exception as exc:  # noqa: BLE001 — 单任务异常不应拖垮守护进程
            logger.error("poll 循环异常（继续）：%s", exc, exc_info=True)
            handled = False
        if not handled:
            time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
