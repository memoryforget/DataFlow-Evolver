"""LlamaFactory Full SFT + Math Benchmark 评测脚本（单机多卡）。

契约：
    python scripts/llamafactory_math_sft_eval.py --dataset <ds.jsonl> --output <out_dir> \\
        [--mode dev|test] [--dev-set <dev.jsonl>]

模式：
    dev（默认）：在框架经 --dev-set 下传的题集上评测，驱动 MCTS 奖励信号。题集内容与条数由
          框架的 trainer.dev_set 决定（core/dev_set.py 物化 + manifest），本脚本不抽样、
          不回退默认题库；缺 --dev-set 直接报错，不会闷头评全量。
    test：GSM8K + AIME2024 + MATH500 + AMC23 + OlympiadBench + CHMath +
          Gaokao + Kaoyan + GradeSchool + MinervaMath + GPQA，仅终评用。

流程：
    1. 数据格式转换（DataFlow JSONL → LlamaFactory sharegpt 格式）
    2. LlamaFactory Full SFT（DeepSpeed ZeRO-3 + FlashAttention-2，单机 SFT_NUM_GPUS 卡）
    3. vLLM 推理（直接加载 fine-tuned 模型）对选定 bench 评测
    4. bench 取平均 accuracy → results.json

训练超参沿用 LIMO 论文配置（2502.03387）：序列长度 16384、学习率 5e-6、15 epochs、
无 warmup（保留 5 步防初期震荡）、ZeRO-3。全局批次 = SFT_NUM_GPUS × BATCH_SIZE × GRAD_ACCUM。
基座与卡数经 SFT_BASE_MODEL / SFT_NUM_GPUS 覆盖；换基座时记得同步 CUTOFF_LEN 与 EVAL_* 的
上下文长度，以及 EVAL_TP_SIZE × EVAL_DP_SIZE（须整除可用卡数）。

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
NUM_GPUS = int(os.environ.get("SFT_NUM_GPUS", "8"))  # 训练：本机参与训练的卡数（torchrun）
CUTOFF_LEN = int(os.environ.get("SFT_CUTOFF_LEN", "16384"))  # LIMO 序列长度
WARMUP_STEPS = int(os.environ.get("SFT_WARMUP_STEPS", "5"))  # 论文无 warmup，保留5步防初期震荡
# 评测：使用独立 vLLM 脚本，支持 TP（tensor parallel）+ DP（data parallel）。
# TP × DP 必须整除本机可用卡数；换基座后按权重大小调 TP（单卡放得下就 TP=1、把卡都给 DP 更快）。
TP_SIZE = int(os.environ.get("EVAL_TP_SIZE", "4"))
DP_SIZE = int(os.environ.get("EVAL_DP_SIZE", "2"))  # 默认 4×2，占满 8 卡
MAX_MODEL_LEN = int(os.environ.get("EVAL_MAX_MODEL_LEN", "32768"))  # 需 ≤ 基座上下文窗口
MAX_NEW_TOKENS = int(os.environ.get("EVAL_MAX_NEW_TOKENS", "32768"))  # 评测生成长度上限，与 LIMO 一致(实际受 MAX_MODEL_LEN-prompt 挤占)
HF_CACHE_DIR = os.environ.get("HF_CACHE_DIR", "./hf_cache")  # HuggingFace 缓存目录（vLLM tokenizer）
MAX_BAD_CASES_PER_BENCH = 30
SFT_TRAIN_TIMEOUT_SEC = int(os.environ.get("SFT_TRAIN_TIMEOUT_SEC", "21600"))
TRAIN_RUN_ID = os.environ.get("DF_TRAIN_RUN_ID", "manual")


def _eval_python() -> str:
    return os.environ.get("SFT_EVAL_PYTHON", "").strip()


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



def _load_results_json(results_path: Path) -> dict:
    if not results_path.exists():
        raise RuntimeError(f"评测结果不存在：{results_path}")
    try:
        return json.loads(results_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"results.json 解析失败：{results_path}") from exc


def _run_eval_with_standalone_script(
    model_dir: Path, output_dir: Path, mode: str, eval_python: str, dev_set: str = ""
) -> dict:
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
    if dev_set:
        cmd += ["--dev-set", str(Path(dev_set).resolve())]

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

    # 清掉训练阶段的分布式痕迹：评测是独立的单进程 vLLM 作业，继承 FORCE_TORCHRUN 之类
    # 会让它误以为自己在 torchrun 里。SFT_EVAL_PYTHON 留着则会递归再拉一层评测。
    for key in ("FORCE_TORCHRUN", "SFT_EVAL_PYTHON", "DF_EVAL_EXTERNAL_LAUNCHER"):
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


def _run_eval_stage(model_dir: Path, output_dir: Path, mode: str, dev_set: str = "") -> dict:
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
    return _run_eval_with_standalone_script(
        eval_model_dir, output_dir, mode, eval_python, dev_set
    )


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
# 2. LlamaFactory Full SFT（LIMO 超参 + ZeRO-3，单机多卡）
# ---------------------------------------------------------------------------
def run_sft(data_dir: Path, output_dir: Path) -> Path:
    """调用 LlamaFactory CLI 执行全量 SFT 训练（LIMO 论文配置）。返回模型输出目录。"""
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

    # 执行训练：LlamaFactory 内部调用 torchrun 管理单机多卡。
    cmd = [
        sys.executable, "-u", "-m", "llamafactory.cli", "train",
        str(sft_yaml_path),
    ]
    env = os.environ.copy()
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env["FORCE_TORCHRUN"] = "1"
    env["NPROC_PER_NODE"] = str(NUM_GPUS)
    env.setdefault("MASTER_PORT", "29500")
    # datasets 缓存放进本次输出目录：默认的共享缓存会让并发/连续的多次训练互相踩
    # （*.incomplete 目录被另一次运行原子 rename 掉 → OSError: Cannot find data file）。
    # 数据量小，重复预处理的开销可忽略。
    env["HF_DATASETS_CACHE"] = str((output_dir / "hf_datasets_cache").resolve())

    logger.info("开始 Full SFT 训练 (LIMO 配置): %s", " ".join(cmd))
    logger.info("单机 %s 卡，MASTER_PORT=%s", env["NPROC_PER_NODE"], env.get("MASTER_PORT"))

    log_path = output_dir / "sft_train.log"
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
    parser.add_argument(
        "--dev-set", default="",
        help="框架物化好的 dev 集 jsonl（core/dev_set.py）。--mode dev 必需：评哪些题由框架"
             "决定，本脚本不抽样也不猜默认题库。--mode test 忽略此项（终评走固定 benchmark）。",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset) if args.dataset else None
    model_dir = Path(args.model_dir) if args.model_dir else None
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = args.mode

    # 标记失败发生在哪个阶段：训练/准备阶段的致命错误应以非零退出（上层判「训练失败」），
    # 只有真正进入评测后才允许回写 score=0.0。
    eval_started = False

    try:
        # 缺 dev 题集要在动 GPU 之前就拦下：这条链路是「先 SFT 再评测」，等到评测阶段才发现
        # 题集没给，一整轮训练已经烧掉了。
        if mode == "dev" and not args.dev_set:
            raise RuntimeError(
                "--mode dev 需要 --dev-set：dev 题集由框架按 trainer.dev_set 物化后下传"
                "（见 core/dev_set.py）。trainer.dev_set.source 留空则不会物化，请配上它；"
                "手动跑评测时自行传一份题集文件。"
            )

        if model_dir is not None:
            # --model-dir 模式：本质就是纯评测已有模型
            if not model_dir.exists():
                raise RuntimeError(f"评测模型目录不存在：{model_dir}")

            eval_python = _eval_python() or sys.executable
            eval_started = True
            results = _run_eval_with_standalone_script(
                model_dir, output_dir, mode, eval_python, args.dev_set
            )
        else:
            if dataset_path is None:
                raise RuntimeError("未提供 --dataset；训练模式下必须指定输入数据集")

            # 1. 格式转换：DataFlow JSONL → LlamaFactory sharegpt
            data_dir = output_dir / "sft_data"
            convert_dataset(dataset_path, data_dir)

            # 2. Full SFT：单机多卡，torchrun 由 LlamaFactory 内部拉起
            model_dir = run_sft(data_dir, output_dir)

            # 3. 评测
            eval_started = True
            results = _run_eval_stage(model_dir, output_dir, mode=mode, dev_set=args.dev_set)

    except Exception as exc:
        logger.error("流程失败：%s", exc, exc_info=True)
        # 走到这里=训练或评测「没跑成」(基础设施/致命错误)：一律非零退出，让上层 TrainerWrapper
        # 命中 rc!=0 → TrainResult(failed=True) → 节点按「系统级失败」处理。
        # 关键：真·0 分【绝不】经过此异常路径——评测正常跑完(哪怕全错)会把 score=0.0 写进
        # results.json 并正常返回(exit 0)。故此处不再回写 score=0.0 伪装成「评测 0 分」，
        # 从而把「评测基础设施失败(如 vllm serve 起不来)」与「真·0 分」彻底区分开。
        stage = "评测(基础设施/致命错误)" if eval_started else "训练/准备"
        logger.error("%s阶段失败，非零退出（判定为系统级失败 failed，非「评测 0 分」）。", stage)
        sys.exit(1)

    # 4. 输出
    results_path = output_dir / "results.json"
    results_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("结果已写入：%s (score=%.4f)", results_path, results["score"])


if __name__ == "__main__":
    main()
