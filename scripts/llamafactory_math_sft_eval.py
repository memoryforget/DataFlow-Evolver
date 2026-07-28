"""LlamaFactory Full SFT + Math Benchmark 评测脚本（LIMO 配置：Qwen2.5-32B 双机16卡）。

契约：
    python scripts/llamafactory_math_sft_eval.py --dataset <ds.jsonl> --output <out_dir> [--mode dev|test]

模式：
    dev（默认）：MATH train 随机抽样 500 条评测（MATH_TRAIN_SAMPLE_SIZE 控制，=0 跑全量 7500），驱动 MCTS 奖励信号。
    test：GSM8K + AIME2024 + MATH500 + AMC23 + OlympiadBench + CHMath +
          Gaokao + Kaoyan + GradeSchool + MinervaMath + GPQA，仅终评用。

流程：
    1. 数据格式转换（DataFlow JSONL → LlamaFactory sharegpt 格式）
    2. LlamaFactory Full SFT（Qwen2.5-32B, DeepSpeed ZeRO-3, 双机16卡 A100 80G）
    3. vLLM 推理（直接加载 fine-tuned 模型）对选定 bench 评测
    4. bench 取平均 accuracy → results.json

LIMO 论文核心配置（2502.03387）：
    - 基础模型: Qwen2.5-32B-Instruct
    - 序列长度: 16384 (所有样本均低于此长度)
    - 学习率: 5e-6 (极低，论文无 warmup)
    - 训练轮数: 15 epochs
    - 全局批次大小: 64 (16卡 × batch_size=1 × grad_accum=4)
    - DeepSpeed ZeRO-3 + FlashAttention-2
    - 数据: 800 条高质量推理样本

双机启动（16卡 A100 80G）：
    硬件要求：双机必须有 100G+ 网络互连（RoCE/InfiniBand），否则通信成为严重瓶颈。

    推荐：只在主节点 (Node 0) 启动，由脚本内部 SSH 拉起 Node 1 worker:
        export NNODES=2
        export NODE_RANK=0
        export NODE1_HOST=<node1 ssh目标>
        export MASTER_ADDR=<主节点IP>
        export MASTER_PORT=29500
        python scripts/llamafactory_math_sft_eval.py --dataset data.jsonl --output output/

    兼容手动模式：也可以在从节点 (Node 1) 单独启动 worker:
        export NNODES=2
        export NODE_RANK=1
        export MASTER_ADDR=<主节点IP>
        export MASTER_PORT=29500
        python scripts/llamafactory_math_sft_eval.py --dataset data.jsonl --output output/

    注意：
    1. 两个节点必须使用相同的 --dataset 和 --output 路径（共享文件系统）
    2. 显存占用估算：ZeRO-3 将 32B 模型状态均摊至 ~32GB/卡，激活值约 20-25GB，总占用 55-65GB
    3. 全局批次大小 = NNODES × (NUM_GPUS/NNODES) × BATCH_SIZE × GRAD_ACCUM = 2 × 8 × 1 × 4 = 64

评测环境依赖：
    - vLLM: 0.19.1 (DP>1 通过 vllm serve + OpenAI API 模式)
    - transformers: 4.57.x (必须 <5，transformers 5.x 删除了 all_special_tokens_extended)
    - torch: 2.10.0+cu128
    - 环境变量: 必须设置 VLLM_WORKER_MULTIPROC_METHOD=spawn (避免 CUDA fork 错误)
    - DP>1 时独立脚本自动启动 vllm serve + OpenAI API 模式推理
    - 推荐使用独立评测环境，避免和训练环境冲突

输出：<out_dir>/results.json
    {"score": float, "metric_name": "math_*_accuracy", "bad_cases": [...], "benchmarks": [...]}
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

# 设置多进程启动方式为 spawn，避免 CUDA fork 问题
multiprocessing.set_start_method('spawn', force=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 配置常量（可通过环境变量覆盖）— LIMO 论文配置
# ---------------------------------------------------------------------------
BASE_MODEL = os.environ.get("SFT_BASE_MODEL", "/vepfs-mlp2/c20250602/500050/models/Qwen2.5-32B-Instruct")
NUM_EPOCHS = float(os.environ.get("SFT_NUM_EPOCHS", "15"))
LEARNING_RATE = float(os.environ.get("SFT_LEARNING_RATE", "5.0e-6"))
BATCH_SIZE = int(os.environ.get("SFT_BATCH_SIZE", "1"))
GRAD_ACCUM = int(os.environ.get("SFT_GRAD_ACCUM", "4"))
NUM_GPUS = int(os.environ.get("SFT_NUM_GPUS", "16"))  # 训练：双机总卡数（torchrun 分布式）
CUTOFF_LEN = int(os.environ.get("SFT_CUTOFF_LEN", "16384"))  # LIMO 序列长度
WARMUP_STEPS = int(os.environ.get("SFT_WARMUP_STEPS", "5"))  # 论文无 warmup，保留5步防初期震荡
# 评测：使用独立 vLLM 脚本，支持 TP（tensor parallel）+ DP（data parallel）。
# 32B 权重 bf16≈64GB，使用 TP=4（每卡 16GB）在 4×A100 80G 上运行。
# vLLM 0.19.1 DP>1 时自动启动 vllm serve + OpenAI API 模式推理（8 GPU）。
TP_SIZE = int(os.environ.get("EVAL_TP_SIZE", "4"))
DP_SIZE = int(os.environ.get("EVAL_DP_SIZE", "2"))  # 默认 DP=2，配合 TP=4 使用全部 8 张 GPU
MAX_MODEL_LEN = int(os.environ.get("EVAL_MAX_MODEL_LEN", "32768"))  # Qwen2.5-32B 上下文窗口,与LIMO一致
MAX_NEW_TOKENS = int(os.environ.get("EVAL_MAX_NEW_TOKENS", "32768"))  # 评测生成长度上限，与 LIMO 一致(实际受 MAX_MODEL_LEN-prompt 挤占)
HF_CACHE_DIR = os.environ.get("HF_CACHE_DIR", "./hf_cache")  # HuggingFace 缓存目录（vLLM tokenizer）
MAX_BAD_CASES_PER_BENCH = 30
SFT_TRAIN_TIMEOUT_SEC = int(os.environ.get("SFT_TRAIN_TIMEOUT_SEC", "21600"))
TRAIN_RUN_ID = os.environ.get("DF_TRAIN_RUN_ID", "manual")


def _node_rank() -> int:
    """本进程的节点 rank（NODE_RANK；单机/未设时为 0）。用于双机下只让 rank0 做评测。"""
    try:
        return int(os.environ.get("NODE_RANK", "0"))
    except ValueError:
        return 0


def _nnodes() -> int:
    try:
        return int(os.environ.get("NNODES", "1"))
    except ValueError:
        return 1


def _node1_host() -> str:
    return os.environ.get("NODE1_HOST", "").strip()


def _ssh_opts() -> str:
    return os.environ.get("SSH_OPTS", "").strip()


def _remote_cwd() -> str:
    return os.environ.get("REMOTE_CWD", "").strip() or os.getcwd()


def _remote_python() -> str:
    return os.environ.get("SFT_REMOTE_PYTHON", sys.executable)


def _eval_python() -> str:
    return os.environ.get("SFT_EVAL_PYTHON", "").strip()


def _should_launch_remote_worker() -> bool:
    return _node_rank() == 0 and _nnodes() > 1


def _read_sysfs_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _detect_nccl_ib_hca_and_gid_index(socket_ifname: str) -> tuple[str | None, str | None]:
    base = Path("/sys/class/infiniband")
    if not socket_ifname or not base.exists():
        return None, None

    candidates: list[tuple[int, str, str]] = []
    for ndev_path in base.glob("*/ports/*/gid_attrs/ndevs/*"):
        if _read_sysfs_text(ndev_path) != socket_ifname:
            continue

        gid_index = ndev_path.name
        port_dir = ndev_path.parents[2]
        gid_value = _read_sysfs_text(port_dir / "gids" / gid_index)
        gid_type = _read_sysfs_text(ndev_path.parents[1] / "types" / gid_index)
        if not gid_value or gid_value == "0000:0000:0000:0000:0000:0000:0000:0000":
            continue

        is_link_local = gid_value.lower().startswith("fe80:")
        if gid_type == "RoCE v2" and not is_link_local:
            score = 0
        elif gid_type == "IB/RoCE v1" and not is_link_local:
            score = 1
        elif gid_type == "RoCE v2":
            score = 2
        elif gid_type == "IB/RoCE v1":
            score = 3
        else:
            score = 9

        hca = ndev_path.parents[4].name
        port = port_dir.name
        candidates.append((score, f"{hca}:{port}", gid_index))

    if not candidates:
        return None, None

    _, hca, gid_index = min(candidates)
    return hca, gid_index


def _ensure_nccl_ib_env_from_socket_ifname() -> None:
    if os.environ.get("NCCL_IB_DISABLE", "").strip() == "1":
        return

    socket_ifname = os.environ.get("NCCL_SOCKET_IFNAME", "").strip()
    if not socket_ifname:
        return

    hca_is_explicit = bool(os.environ.get("NCCL_IB_HCA", "").strip())
    gid_is_explicit = bool(os.environ.get("NCCL_IB_GID_INDEX", "").strip())
    if hca_is_explicit and gid_is_explicit:
        return

    hca, gid_index = _detect_nccl_ib_hca_and_gid_index(socket_ifname)
    if not hca and not gid_index:
        logger.warning("无法根据 NCCL_SOCKET_IFNAME=%s 推断 NCCL_IB_HCA / NCCL_IB_GID_INDEX", socket_ifname)
        return

    if not hca_is_explicit and hca:
        os.environ["NCCL_IB_HCA"] = hca
        os.environ["DF_AUTO_NCCL_IB_HCA"] = "1"
    if not gid_is_explicit and gid_index:
        os.environ["NCCL_IB_GID_INDEX"] = gid_index
        os.environ["DF_AUTO_NCCL_IB_GID_INDEX"] = "1"

    logger.info(
        "自动设置 NCCL RDMA 配置: NCCL_SOCKET_IFNAME=%s, NCCL_IB_HCA=%s, NCCL_IB_GID_INDEX=%s",
        socket_ifname,
        os.environ.get("NCCL_IB_HCA", ""),
        os.environ.get("NCCL_IB_GID_INDEX", ""),
    )


def _kill_process_tree(pid: int) -> None:
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            capture_output=True,
            timeout=10,
        )
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        time.sleep(2)
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
    except ProcessLookupError:
        pass


def _stream_subprocess(
    cmd: list[str],
    *,
    cwd: str,
    env: dict[str, str],
    log_path: Path,
    timeout_sec: int,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as logf:
        logf.write(f"CMD: {' '.join(cmd)}\n")
        logf.write(f"CWD: {cwd}\n")
        logf.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            env=env,
            start_new_session=True,
        )
        try:
            return proc.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired as exc:
            _kill_process_tree(proc.pid)
            raise TimeoutError(
                f"训练子进程超时（>{timeout_sec}s），日志：{log_path}"
            ) from exc


def _forward_remote_env() -> dict[str, str]:
    prefixes = (
        "SFT_",
        "EVAL_",
        "HF_",
        "MATH_",
        "GLOO_",
        "NCCL_",
        "CUDA_",
        "TORCH_",
        "OMP_",
        "MKL_",
    )
    exact = {
        "PATH",
        "PYTHONPATH",
        "LD_LIBRARY_PATH",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "TRANSFORMERS_CACHE",
        "HF_HOME",
        "HF_DATASETS_CACHE",
        "XDG_CACHE_HOME",
        "CUDA_VISIBLE_DEVICES",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "VIRTUAL_ENV",
        "NO_PROXY",
        "no_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
        "MASTER_ADDR",
        "MASTER_PORT",
        "NNODES",
        "SFT_NUM_GPUS",
        "DF_TRAIN_RUN_ID",
    }
    env: dict[str, str] = {}
    auto_ib_hca = os.environ.get("DF_AUTO_NCCL_IB_HCA") == "1"
    auto_ib_gid = os.environ.get("DF_AUTO_NCCL_IB_GID_INDEX") == "1"
    for key, value in os.environ.items():
        if not value:
            continue
        if auto_ib_hca and key == "NCCL_IB_HCA":
            continue
        if auto_ib_gid and key == "NCCL_IB_GID_INDEX":
            continue
        if key in exact or key.startswith(prefixes):
            env[key] = value
    return env


def _build_remote_worker_ssh_cmd(dataset_path: Path, output_dir: Path, mode: str) -> list[str]:
    node1_host = _node1_host()
    if not node1_host:
        raise RuntimeError("NNODES>1 但未设置 NODE1_HOST，无法从 node0 拉起 node1 worker")

    env = _forward_remote_env()
    env.update({
        "NNODES": str(_nnodes()),
        "NODE_RANK": "1",
        "MASTER_ADDR": os.environ.get("MASTER_ADDR", ""),
        "MASTER_PORT": os.environ.get("MASTER_PORT", "29500"),
        "SFT_NUM_GPUS": os.environ.get("SFT_NUM_GPUS", str(NUM_GPUS)),
        "FORCE_TORCHRUN": "1",
        "DF_TRAIN_RUN_ID": TRAIN_RUN_ID,
        "NODE1_HOST": node1_host,
        "REMOTE_CWD": _remote_cwd(),
        "SSH_OPTS": _ssh_opts(),
        "SFT_REMOTE_PYTHON": _remote_python(),
    })
    env_prefix = " ".join(
        f"{key}={shlex.quote(value)}" for key, value in sorted(env.items())
    )

    script_path = Path("scripts") / Path(__file__).name
    remote_inner = " ".join(
        shlex.quote(token)
        for token in [
            _remote_python(),
            str(script_path),
            "--dataset",
            str(dataset_path.resolve()),
            "--output",
            str(output_dir.resolve()),
            "--mode",
            mode,
        ]
    )
    remote_cmd = f"cd {shlex.quote(_remote_cwd())} && {env_prefix} {remote_inner}"
    ssh_cmd = ["ssh"]
    if _ssh_opts():
        ssh_cmd += shlex.split(_ssh_opts())
    ssh_cmd += [node1_host, remote_cmd]
    return ssh_cmd


def _launch_remote_worker(dataset_path: Path, output_dir: Path, mode: str) -> subprocess.Popen:
    ssh_cmd = _build_remote_worker_ssh_cmd(dataset_path, output_dir, mode)
    remote_log = output_dir / "train_node1.log"
    logger.info("SSH 拉起 node1 worker：%s", " ".join(ssh_cmd))
    logf = open(remote_log, "w", encoding="utf-8")  # noqa: SIM115
    proc = subprocess.Popen(ssh_cmd, stdout=logf, stderr=subprocess.STDOUT, text=True)
    proc._logfile = logf  # type: ignore[attr-defined]
    return proc


def _close_logfile(proc: subprocess.Popen) -> None:
    lf = getattr(proc, "_logfile", None)
    if lf is not None:
        try:
            lf.close()
        except OSError:
            pass


def _stop_remote_worker(proc: subprocess.Popen) -> None:
    try:
        _kill_process_tree(proc.pid)
    except Exception:  # noqa: BLE001
        pass
    finally:
        _close_logfile(proc)


def _wait_remote_worker(proc: subprocess.Popen, timeout_sec: int) -> None:
    try:
        rc = proc.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired as exc:
        _stop_remote_worker(proc)
        raise TimeoutError(f"node1 worker 未在 {timeout_sec}s 内收尾") from exc
    finally:
        _close_logfile(proc)
    if rc != 0:
        raise RuntimeError(f"node1 worker 非零退出（{rc}），详见 train_node1.log")


def _load_results_json(results_path: Path) -> dict:
    if not results_path.exists():
        raise RuntimeError(f"评测结果不存在：{results_path}")
    try:
        return json.loads(results_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"results.json 解析失败：{results_path}") from exc


def _run_eval_with_standalone_script(model_dir: Path, output_dir: Path, mode: str, eval_python: str) -> dict:
    """使用独立的 vLLM 评测脚本运行评测（避免 CUDA fork 问题）。

    DP>1 时由独立脚本内部自动启动 vllm serve + OpenAI API 模式推理。
    DP=1 时直接用 vLLM LLM 类。
    """
    script_path = Path(__file__).resolve().parent / "vllm_eval_standalone.py"
    results_path = output_dir / "results.json"
    log_path = output_dir / "eval.log"

    cmd = [
        eval_python, str(script_path),
        "--model-dir", str(model_dir.resolve()),
        "--output-dir", str(output_dir.resolve()),
        "--mode", mode,
        "--tp-size", str(TP_SIZE),
        "--dp-size", str(DP_SIZE),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--max-new-tokens", str(MAX_NEW_TOKENS),
    ]

    env = dict(os.environ)
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    # 设置评测使用的 GPU
    eval_gpus = os.environ.get("EVAL_CUDA_VISIBLE_DEVICES", "").strip()
    if eval_gpus:
        env["CUDA_VISIBLE_DEVICES"] = eval_gpus
        logger.info("评测使用 GPU: %s", eval_gpus)
    elif "CUDA_VISIBLE_DEVICES" not in env:
        total_gpus = TP_SIZE * DP_SIZE
        default_gpus = ",".join(str(i) for i in range(total_gpus))
        env["CUDA_VISIBLE_DEVICES"] = default_gpus
        logger.info("评测使用默认 GPU: %s", default_gpus)

    # 清理不需要的环境变量
    for key in (
        "NODE1_HOST",
        "FORCE_TORCHRUN",
        "SFT_REMOTE_PYTHON",
        "SFT_EVAL_PYTHON",
        "DF_EVAL_EXTERNAL_LAUNCHER",
    ):
        env.pop(key, None)

    logger.info("使用独立 vLLM 评测脚本：%s", script_path)
    logger.info("启动评测子进程 (TP=%d, DP=%d)：%s", TP_SIZE, DP_SIZE, " ".join(shlex.quote(part) for part in cmd))

    with log_path.open("w", encoding="utf-8", buffering=1) as logf:
        logf.write(f"CMD: {' '.join(cmd)}\n")
        logf.write(f"CWD: {script_path.parent.parent}\n")
        logf.write(f"MODEL_DIR: {model_dir.resolve()}\n")
        logf.write(f"CUDA_VISIBLE_DEVICES: {env.get('CUDA_VISIBLE_DEVICES', 'not set')}\n")
        logf.write(f"VLLM_WORKER_MULTIPROC_METHOD: spawn\n")
        logf.write(f"TP_SIZE: {TP_SIZE}, DP_SIZE: {DP_SIZE}\n")
        logf.flush()
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=logf,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(script_path.parent.parent),
                env=env,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"eval 环境 python 不可执行：{eval_python}") from exc
        rc = proc.wait()

    if rc != 0:
        raise RuntimeError(f"评测子进程失败 (exit={rc})，日志：{log_path}")
    return _load_results_json(results_path)


def _run_eval_stage(model_dir: Path, output_dir: Path, mode: str) -> dict:
    """执行评测阶段。

    SFT 的 save_model 直接把 HF bf16 safetensors 落在 model_dir，评测直接使用即可。
    """
    eval_model_dir = model_dir

    eval_python = _eval_python()
    if not eval_python:
        raise RuntimeError(
            "未配置评测环境 python（SFT_EVAL_PYTHON 环境变量为空）。"
            "评测必须在独立的 df-eval 环境中运行。"
        )
    return _run_eval_with_standalone_script(eval_model_dir, output_dir, mode, eval_python)


# ---------------------------------------------------------------------------
# 1. 数据格式转换 (DataFlow JSONL → LlamaFactory ShareGPT)
# ---------------------------------------------------------------------------
def convert_dataset(src_path: Path, dst_dir: Path) -> Path:
    """将 DataFlow JSONL 转为 LlamaFactory sharegpt 格式的 JSON 文件。

    ShareGPT 格式支持多轮对话，每条样本含 conversations: [{from, value}, ...]
    过滤超过 CUTOFF_LEN 的样本（使用实际 tokenizer 精确计算）
    """
    from transformers import AutoTokenizer

    rows = []
    filtered_count = 0
    total_count = 0

    # 初始化 tokenizer（与训练模型一致）
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        logger.info("已加载 tokenizer: %s", BASE_MODEL)
    except Exception as e:
        logger.warning("无法加载 tokenizer (%s)，回退到粗略估算模式", e)
        tokenizer = None

    with src_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            total_count += 1
            # 提取 instruction/question 和 output/answer
            instruction = obj.get("instruction", obj.get("question", obj.get("input", "")))
            output = obj.get("generated_cot", obj.get("output", obj.get("answer", "")))
            if instruction and output:
                system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."

                # 计算 token 数
                if tokenizer is not None:
                    # 精确计算：使用实际 tokenizer
                    try:
                        # 模拟实际训练格式
                        conversation = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": instruction},
                            {"role": "assistant", "content": output},
                        ]
                        formatted = tokenizer.apply_chat_template(
                            conversation, tokenize=False, add_generation_prompt=False
                        )
                        token_count = len(tokenizer.encode(formatted, add_special_tokens=True))
                    except Exception as e:
                        logger.warning("Token 计算失败，跳过样本: %s", e)
                        filtered_count += 1
                        continue
                else:
                    # 粗略估算（回退模式）
                    token_count = int((len(system_prompt) + len(instruction) + len(output)) * 0.5)

                # 过滤超长样本
                if token_count > CUTOFF_LEN:
                    filtered_count += 1
                    continue

                # ShareGPT 格式：system prompt + user + assistant
                conversation = [
                    {"from": "system", "value": system_prompt},
                    {"from": "human", "value": instruction},
                    {"from": "gpt", "value": output},
                ]
                rows.append({"conversations": conversation})

    if filtered_count > 0:
        logger.warning(
            "过滤超长样本：%d/%d (%.1f%%) 超过 CUTOFF_LEN=%d",
            filtered_count, total_count, 100.0 * filtered_count / total_count, CUTOFF_LEN
        )

    if not rows:
        raise ValueError(f"数据集为空或字段不匹配：{src_path}")

    dst_dir.mkdir(parents=True, exist_ok=True)
    train_file = dst_dir / "train.json"
    train_file.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    # dataset_info.json 供 LlamaFactory 识别
    info = {
        "train": {
            "file_name": "train.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations"},
        }
    }
    (dst_dir / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("格式转换完成：%d 条训练样本 → %s (sharegpt)", len(rows), train_file)
    return train_file


# ---------------------------------------------------------------------------
# 2. LlamaFactory Full SFT (LIMO 配置: Qwen2.5-32B + ZeRO-3)
# ---------------------------------------------------------------------------
def run_sft(data_dir: Path, output_dir: Path) -> Path:
    """调用 LlamaFactory CLI 执行全量 SFT 训练（LIMO 论文配置）。返回模型输出目录。"""
    rank = _node_rank()
    model_dir = output_dir / "sft_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    sft_config = {
        "model_name_or_path": BASE_MODEL,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "full",
        "dataset_dir": str(data_dir),
        "dataset": "train",
        "template": "qwen",
        "cutoff_len": CUTOFF_LEN,
        "output_dir": str(model_dir),
        "num_train_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "per_device_train_batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "lr_scheduler_type": "cosine",
        "warmup_steps": WARMUP_STEPS,
        "bf16": True,
        "flash_attn": "fa2",
        "optim": "adamw_torch",
        "max_grad_norm": 1.0,
        "logging_steps": 10,
        "save_strategy": "no",
        "report_to": "none",
        "overwrite_output_dir": True,
        "dataloader_pin_memory": False,
    }

    # DeepSpeed ZeRO-3 配置 (LIMO 论文)
    ds_config = {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "zero_optimization": {
            "stage": 3,
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_prefetch_bucket_size": 5e8,
            "stage3_param_persistence_threshold": 1e4,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
            # 必须开启：ZeRO-3 下 save_model 才会 gather 分片、落出可加载的 HF bf16 safetensors；
            # 否则只剩 DeepSpeed 原始分片(global_step*)，vLLM 无法加载权重。
            "stage3_gather_16bit_weights_on_model_save": True,
        },
        "activation_checkpointing": {
            "partition_activations": False,
            "cpu_checkpointing": False,
        },
        "bf16": {"enabled": True},
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "betas": "auto",
                "eps": "auto",
                "weight_decay": "auto",
            },
        },
        # 刻意不在 ds_config 里放 "scheduler"：HF Trainer 的 DeepSpeed 集成规则是
        # 「ds_config 有 scheduler → 用 DeepSpeed 调度器、忽略 lr_scheduler_type」
        # （transformers/integrations/deepspeed.py: `if "scheduler" in config`）。
        # 省掉它，才会走 HF 分支用下面 sft_config 的 lr_scheduler_type=cosine + warmup_steps
        # （即 LIMO 的余弦调度）；warmup_max_lr/steps 等 auto 值也由 HF 学习率/warmup_steps 提供。
    }

    # 写临时配置文件
    config_dir = output_dir / f"sft_configs_rank{rank}"
    config_dir.mkdir(parents=True, exist_ok=True)

    ds_config_path = config_dir / "ds_z3.json"
    ds_config_path.write_text(json.dumps(ds_config, indent=2), encoding="utf-8")
    sft_config["deepspeed"] = str(ds_config_path)

    sft_yaml_path = config_dir / "sft.yaml"
    import yaml
    sft_yaml_path.write_text(yaml.dump(sft_config, default_flow_style=False), encoding="utf-8")

    # 执行训练：LlamaFactory 内部调用 torchrun 管理分布式。
    # 双机配置通过环境变量传递（NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT）
    cmd = [
        sys.executable, "-u", "-m", "llamafactory.cli", "train",
        str(sft_yaml_path),
    ]
    env = os.environ.copy()
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env["FORCE_TORCHRUN"] = "1"
    # 双机配置：需在启动脚本外设置 NNODES=2, NODE_RANK=0/1, MASTER_ADDR, MASTER_PORT
    env.setdefault("NNODES", "1")  # 单机默认，双机需外部设置为 2
    env["NPROC_PER_NODE"] = str(NUM_GPUS // int(env.get("NNODES", "1")))  # 每机GPU数
    env.setdefault("MASTER_PORT", "29500")
    # HF datasets 缓存按 node_rank 隔离：多节点共享 vepfs 时若指向同一 datasets 缓存目录，
    # 首次生成会竞态——两节点各写各的 *.incomplete/，谁先原子 rename 成最终名，另一节点手里
    # 的 .incomplete 路径就"文件消失"(OSError: Cannot find data file)。给每个节点独立的绝对
    # 缓存目录即可根治（数据量小，重复预处理开销可忽略）；硬设覆盖外部转发进来的相对/共享路径。
    env["HF_DATASETS_CACHE"] = str((output_dir / f"hf_datasets_cache_node{rank}").resolve())

    logger.info("开始 Full SFT 训练 (LIMO 配置): %s", " ".join(cmd))
    logger.info("分布式配置: NNODES=%s, NPROC_PER_NODE=%s, MASTER_ADDR=%s, MASTER_PORT=%s",
                env.get("NNODES"), env["NPROC_PER_NODE"],
                env.get("MASTER_ADDR", "localhost"), env.get("MASTER_PORT"))

    log_path = output_dir / f"sft_train_rank{rank}.log"
    returncode = _stream_subprocess(
        cmd,
        cwd=str(output_dir),
        env=env,
        log_path=log_path,
        timeout_sec=SFT_TRAIN_TIMEOUT_SEC,
    )

    if returncode != 0:
        raise RuntimeError(f"SFT 训练失败 (exit={returncode})，日志：{log_path}")

    logger.info("Full SFT 训练完成，模型保存于：%s", model_dir)

    # 只在 rank0 执行 DeepSpeed checkpoint 转换和清理
    if rank == 0:
        _convert_and_cleanup_checkpoint(model_dir, output_dir)

    return model_dir


def _sync_tokenizer_from_base(model_dir: Path) -> None:
    """用 base 模型的 tokenizer 文件覆盖 model_dir。

    SFT 不会修改 tokenizer，但 df-train 的 transformers 5.x 写出的 tokenizer 文件与
    df-eval 的 transformers 4.x 存在格式漂移（additional_special_tokens 丢失、
    chat_template 从内联改为独立 .jinja、tokenizer.json 重新序列化等）。直接拷贝 base
    的 tokenizer 文件最干净，可彻底消除格式差异并保证 4.x 正确加载。"""
    base = Path(BASE_MODEL)
    if not base.exists():
        logger.warning("BASE_MODEL 不存在：%s，跳过 tokenizer 同步", base)
        return
    names = ["tokenizer_config.json", "tokenizer.json", "vocab.json",
             "merges.txt", "special_tokens_map.json", "added_tokens.json"]
    copied = [n for n in names if (base / n).exists()
              and (shutil.copy2(base / n, model_dir / n) or True)]
    # base 把 chat_template 内联在 tokenizer_config.json；删除训练残留的独立 .jinja 以免二义
    jinja = model_dir / "chat_template.jinja"
    if jinja.exists():
        jinja.unlink()
    logger.info("已用 base tokenizer 覆盖 %s：%s", model_dir, copied)


def _convert_and_cleanup_checkpoint(model_dir: Path, output_dir: Path) -> None:
    """确认 SFT 产出的 HF bf16 权重可用，并清理 DeepSpeed 大分片。

    ds_config 已开启 stage3_gather_16bit_weights_on_model_save，因此 LlamaFactory 的
    save_model 会直接在 model_dir 落出可加载的 HF bf16 safetensors，无需任何后处理转换。
    本函数只做两件事：
    1. 用 base 模型的 tokenizer 文件覆盖 model_dir（消除 5.x/4.x 格式漂移，保证 df-eval 正确加载）；
    2. 若训练残留了 DeepSpeed 原始分片（latest + global_step*/，~300GB），删除以省磁盘。

    关键：tokenizer 同步必须与「是否有 DeepSpeed checkpoint 需要清理」解耦。开了
    stage3_gather_16bit_weights_on_model_save + save_strategy=no 时，save_model 直接落 HF
    safetensors、**不产生 latest 文件**（那是 save_checkpoint 才有的）；若把同步挂在 latest
    存在的前提下，正常路径会整个跳过同步，导致 df-eval(transformers 4.x) 加载 5.x tokenizer
    时崩溃（extra_special_tokens 为 list → 'list' object has no attribute 'keys'）。
    """
    logger.info("确认 HF bf16 权重、同步 tokenizer 并清理 DeepSpeed 分片...")

    # 1) 先确认 save_model 已产出可加载的 HF safetensors（正常路径无需 latest/global_step*）。
    #    缺失 = 训练保存异常：直接抛错让训练这步当场失败、节点按「训练失败」处理，
    #    而不是任由后续 vLLM 加载空目录 → serve 超时 → 伪装成一次「评测 0 分」（会与
    #    数据质量差的 0 分混淆）。抛错发生在清理之前，任何 DeepSpeed 分片都会原样保留供人工抢救。
    if not any(model_dir.glob("*.safetensors")):
        raise RuntimeError(
            f"save_model 未产出 HF safetensors，训练保存异常（应由 save_model + "
            f"stage3_gather_16bit_weights_on_model_save 产出）：{model_dir}；"
            f"已保留任何 DeepSpeed 分片（global_step*/）供人工抢救。")

    # 2) 用 base tokenizer 覆盖，消除 transformers 5.x/4.x 格式漂移（与有无 DeepSpeed 分片无关）
    logger.info("确认 HF bf16 权重位于 model_dir：%s", model_dir)
    _sync_tokenizer_from_base(model_dir)

    # 3) 清理 DeepSpeed 原始分片（若存在）：latest 指向的目录 + 任何 global_step*/
    logger.info("清理 DeepSpeed 分片（若有）...")
    checkpoint_dirs = {p for p in model_dir.glob("global_step*") if p.is_dir()}
    latest_file = model_dir / "latest"
    if latest_file.exists():
        tag = latest_file.read_text(encoding="utf-8").strip()
        if tag and (model_dir / tag).is_dir():
            checkpoint_dirs.add(model_dir / tag)
    for ckpt in checkpoint_dirs:
        size = sum(f.stat().st_size for f in ckpt.rglob("*") if f.is_file())
        logger.info("删除 DeepSpeed 分片目录：%s (大小：%.2f GB)", ckpt, size / (1024**3))
        shutil.rmtree(ckpt)
    if latest_file.exists():
        latest_file.unlink()

    # 显示清理后的磁盘占用
    total_size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
    logger.info("清理完成，模型目录大小：%.2f GB", total_size / (1024**3))


# ---------------------------------------------------------------------------
# 主入口
# 主入口
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="LlamaFactory Math SFT + Eval")
    parser.add_argument("--dataset", help="输入数据集 JSONL 路径（训练时必需）")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument(
        "--model-dir",
        default="",
        help="已训练模型目录；提供后跳过数据转换和 SFT，仅执行评测",
    )
    parser.add_argument(
        "--mode", default="dev", choices=["dev", "test"],
        help="dev=MATH train only（MCTS 迭代）；test=full math/science benchmark suite（终评）",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset) if args.dataset else None
    model_dir = Path(args.model_dir) if args.model_dir else None
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = args.mode
    rank = _node_rank()
    remote_proc: subprocess.Popen | None = None

    _ensure_nccl_ib_env_from_socket_ifname()

    # 标记失败发生在哪个阶段：训练/准备阶段的致命错误应以非零退出（上层判「训练失败」），
    # 只有真正进入评测后才允许回写 score=0.0。
    eval_started = False

    try:
        if model_dir is not None:
            # --model-dir 模式：本质就是纯评测已有模型
            if not model_dir.exists():
                raise RuntimeError(f"评测模型目录不存在：{model_dir}")

            eval_python = _eval_python() or sys.executable
            eval_started = True
            results = _run_eval_with_standalone_script(model_dir, output_dir, mode, eval_python)
        else:
            if dataset_path is None:
                raise RuntimeError("未提供 --dataset；训练模式下必须指定输入数据集")
            if _nnodes() > 1 and rank == 0 and not _node1_host():
                raise RuntimeError("NNODES>1 但未设置 NODE1_HOST")

            # 1. 数据转换完全由 node0 / rank0 负责；node1 worker 不再介入。
            data_dir = output_dir / "sft_data"
            if rank == 0:
                convert_dataset(dataset_path, data_dir)

            # 2. 双机模式：由 node0 脚本内部先 SSH 拉起 node1 worker，再本地进入 rank0 训练。
            if _should_launch_remote_worker():
                remote_proc = _launch_remote_worker(dataset_path, output_dir, mode)

            # 3. Full SFT：torchrun 在两机间协调的分布式作业。
            model_dir = run_sft(data_dir, output_dir)
            if remote_proc is not None:
                _wait_remote_worker(remote_proc, timeout_sec=max(60, SFT_TRAIN_TIMEOUT_SEC // 100))
                remote_proc = None

            # 4. 评测：只在 rank0 单机执行；node1 worker 训练完即退出。
            if rank != 0:
                logger.info("rank=%d 训练完成，非 rank0 不参与评测，退出。", rank)
                return
            eval_started = True
            results = _run_eval_stage(model_dir, output_dir, mode=mode)

    except Exception as exc:
        logger.error("流程失败：%s", exc, exc_info=True)
        if remote_proc is not None:
            _stop_remote_worker(remote_proc)
        if rank != 0:
            return  # 非 rank0 失败不写 results.json（由 rank0 统一裁决）
        # 走到这里=训练或评测「没跑成」(基础设施/致命错误)：一律非零退出，让上层 TrainerWrapper
        # 命中 rc!=0 → TrainResult(failed=True) → 节点按「系统级失败」处理。
        # 关键：真·0 分【绝不】经过此异常路径——评测正常跑完(哪怕全错)会把 score=0.0 写进
        # results.json 并正常返回(exit 0)。故此处不再回写 score=0.0 伪装成「评测 0 分」，
        # 从而把「评测基础设施失败(如 vllm serve 起不来)」与「真·0 分」彻底区分开。
        stage = "评测(基础设施/致命错误)" if eval_started else "训练/准备"
        logger.error("%s阶段失败，非零退出（判定为系统级失败 failed，非「评测 0 分」）。", stage)
        sys.exit(1)

    # 4. 输出（仅 rank0 到达此处）
    results_path = output_dir / "results.json"
    results_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("结果已写入：%s (score=%.4f)", results_path, results["score"])


if __name__ == "__main__":
    main()
