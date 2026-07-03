"""LlamaFactory Full SFT + Math Benchmark 评测脚本（LIMO 配置：Qwen2.5-32B 双机16卡）。

契约：
    python scripts/llamafactory_math_sft_eval.py --dataset <ds.jsonl> --output <out_dir> [--mode dev|test]

模式：
    dev（默认）：MATH train 全量 (7500 条) 评测，驱动 MCTS 奖励信号。
    test：GSM8K test + MATH-500 + AIME 2024 完整套件，仅终评用。

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

    主节点 (Node 0):
        export NNODES=2
        export NODE_RANK=0
        export MASTER_ADDR=<主节点IP>
        export MASTER_PORT=29500
        python scripts/llamafactory_math_sft_eval.py --dataset data.jsonl --output output/

    从节点 (Node 1):
        export NNODES=2
        export NODE_RANK=1
        export MASTER_ADDR=<主节点IP>
        export MASTER_PORT=29500
        python scripts/llamafactory_math_sft_eval.py --dataset data.jsonl --output output/

    注意：
    1. 两个节点必须使用相同的 --dataset 和 --output 路径（共享文件系统或手动同步）
    2. 显存占用估算：ZeRO-3 将 32B 模型状态均摊至 ~32GB/卡，激活值约 20-25GB，总占用 55-65GB
    3. 全局批次大小 = NNODES × (NUM_GPUS/NNODES) × BATCH_SIZE × GRAD_ACCUM = 2 × 8 × 1 × 4 = 64

输出：<out_dir>/results.json
    {"score": float, "metric_name": "math_avg_accuracy", "bad_cases": [...]}
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 配置常量（可通过环境变量覆盖）— LIMO 论文配置
# ---------------------------------------------------------------------------
BASE_MODEL = os.environ.get("SFT_BASE_MODEL", "Qwen/Qwen2.5-32B-Instruct")
NUM_EPOCHS = float(os.environ.get("SFT_NUM_EPOCHS", "15"))
LEARNING_RATE = float(os.environ.get("SFT_LEARNING_RATE", "5.0e-6"))
BATCH_SIZE = int(os.environ.get("SFT_BATCH_SIZE", "1"))
GRAD_ACCUM = int(os.environ.get("SFT_GRAD_ACCUM", "4"))
NUM_GPUS = int(os.environ.get("SFT_NUM_GPUS", "16"))  # 双机16卡
CUTOFF_LEN = int(os.environ.get("SFT_CUTOFF_LEN", "16384"))  # LIMO 序列长度
WARMUP_STEPS = int(os.environ.get("SFT_WARMUP_STEPS", "5"))  # 论文无 warmup，保留5步防初期震荡
TP_SIZE = int(os.environ.get("EVAL_TP_SIZE", "2"))  # 评测时32B需要更大TP
DP_SIZE = int(os.environ.get("EVAL_DP_SIZE", "8"))
MAX_MODEL_LEN = int(os.environ.get("EVAL_MAX_MODEL_LEN", "131072"))  # Qwen2.5-32B 上下文窗口
MAX_NEW_TOKENS = int(os.environ.get("EVAL_MAX_NEW_TOKENS", "32768"))  # LIMO 论文：硬截断于32768
HF_CACHE_DIR = os.environ.get("HF_CACHE_DIR", None)
MAX_BAD_CASES_PER_BENCH = 10


# ---------------------------------------------------------------------------
# 1. 数据格式转换 (DataFlow JSONL → LlamaFactory ShareGPT)
# ---------------------------------------------------------------------------
def convert_dataset(src_path: Path, dst_dir: Path) -> Path:
    """将 DataFlow JSONL 转为 LlamaFactory sharegpt 格式的 JSON 文件。

    ShareGPT 格式支持多轮对话，每条样本含 conversations: [{from, value}, ...]
    """
    rows = []
    with src_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            # 提取 instruction/question 和 output/answer
            instruction = obj.get("instruction", obj.get("question", obj.get("input", "")))
            output = obj.get("generated_cot", obj.get("output", obj.get("answer", "")))
            if instruction and output:
                # ShareGPT 格式：system prompt + user + assistant
                conversation = [
                    {"from": "system", "value": "Please reason step by step, and put your final answer within \\boxed{}"},
                    {"from": "human", "value": instruction},
                    {"from": "gpt", "value": output},
                ]
                rows.append({"conversations": conversation})

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
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": "auto",
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto",
                "total_num_steps": "auto",
            },
        },
    }

    # 写临时配置文件
    config_dir = output_dir / "sft_configs"
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
        sys.executable, "-m", "llamafactory.cli", "train",
        str(sft_yaml_path),
    ]
    env = os.environ.copy()
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env["FORCE_TORCHRUN"] = "1"
    # 双机配置：需在启动脚本外设置 NNODES=2, NODE_RANK=0/1, MASTER_ADDR, MASTER_PORT
    env.setdefault("NNODES", "1")  # 单机默认，双机需外部设置为 2
    env["NPROC_PER_NODE"] = str(NUM_GPUS // int(env.get("NNODES", "1")))  # 每机GPU数
    env.setdefault("MASTER_PORT", "29500")

    logger.info("开始 Full SFT 训练 (LIMO 配置): %s", " ".join(cmd))
    logger.info("分布式配置: NNODES=%s, NPROC_PER_NODE=%s, MASTER_ADDR=%s, MASTER_PORT=%s",
                env.get("NNODES"), env["NPROC_PER_NODE"],
                env.get("MASTER_ADDR", "localhost"), env.get("MASTER_PORT"))

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=21600, cwd=str(output_dir), env=env
    )
    log_path = output_dir / "sft_train.log"
    log_path.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")

    if result.returncode != 0:
        raise RuntimeError(f"SFT 训练失败 (exit={result.returncode})，日志：{log_path}")

    logger.info("Full SFT 训练完成，模型保存于：%s", model_dir)
    return model_dir


# ---------------------------------------------------------------------------
# 3. Math Benchmark 评测
# ---------------------------------------------------------------------------
EVAL_SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}"


def run_eval(model_dir: Path, output_dir: Path, mode: str = "dev") -> dict:
    """用 vLLM 加载 fine-tuned 模型评测。

    mode='dev': MATH train 全量 (7500 条公开训练题，非 benchmark test set) 驱动 MCTS 奖励。
    mode='test': 完整测试套件，区分小基准（多样本 pass@1）和大基准（单样本贪心）。
    """
    from vllm import LLM, SamplingParams
    import math

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from math_benchmarks import (
        load_gsm8k, gsm8k_prompt, gsm8k_score,
        load_math500, math500_prompt, math500_score,
        load_math_train, math_train_prompt, math_train_score,
        load_aime2024, aime_prompt, aime_score,
        load_amc23, amc23_prompt, amc23_score,
        load_chmath, chmath_prompt, chmath_score,
        load_olympiadbench, olympiadbench_prompt, olympiadbench_score,
        load_gaokao, gaokao_prompt, gaokao_score,
        load_kaoyan, kaoyan_prompt, kaoyan_score,
        load_gradeschool, gradeschool_prompt, gradeschool_score,
        load_minervamath, minervamath_prompt, minervamath_score,
        load_gpqa, gpqa_prompt, gpqa_score,
    )

    logger.info("加载 vLLM 模型：%s (TP=%d, DP=%d)", model_dir, TP_SIZE, DP_SIZE)
    llm = LLM(
        model=str(model_dir),
        tensor_parallel_size=TP_SIZE,
        data_parallel_size=DP_SIZE,
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=True,
        enable_prefix_caching=True,
    )

    tokenizer = llm.get_tokenizer()

    if mode == "test":
        # 小基准（<50题）：生成 4 个样本，temperature=0.6，计算无偏 pass@1
        small_benchmarks = [
            ("AIME24", load_aime2024, aime_prompt, aime_score),
            ("AMC23", load_amc23, amc23_prompt, amc23_score),
            ("CHMath", load_chmath, chmath_prompt, chmath_score),
        ]
        # 大基准（≥50题）：单次贪心，temperature=0
        large_benchmarks = [
            ("MATH500", load_math500, math500_prompt, math500_score),
            ("OlympiadBench", load_olympiadbench, olympiadbench_prompt, olympiadbench_score),
            ("Gaokao", load_gaokao, gaokao_prompt, gaokao_score),
            ("Kaoyan", load_kaoyan, kaoyan_prompt, kaoyan_score),
            ("GradeSchool", load_gradeschool, gradeschool_prompt, gradeschool_score),
            ("MinervaMath", load_minervamath, minervamath_prompt, minervamath_score),
            ("GPQA", load_gpqa, gpqa_prompt, gpqa_score),
        ]
        metric_name = "math_test_avg_accuracy"
    else:
        # dev mode: MATH train 全量，单样本贪心
        math_train_path = os.environ.get("MATH_TRAIN_PATH", None)
        small_benchmarks = []
        large_benchmarks = [
            ("MATH-train", lambda **kw: load_math_train(data_path=math_train_path, **kw), math_train_prompt, math_train_score),
        ]
        metric_name = "math_train_accuracy"

    scores = []
    all_bad_cases: list[dict] = []
    detail = {}

    # 处理小基准：多样本 pass@1
    for bench_name, load_fn, prompt_fn, score_fn in small_benchmarks:
        logger.info("评测 %s (多样本 pass@1) ...", bench_name)
        examples = load_fn(cache_dir=HF_CACHE_DIR)
        if not examples:
            logger.warning("%s 数据集为空，跳过", bench_name)
            continue

        # 构造 prompts
        prompts = []
        for ex in examples:
            user_msg = prompt_fn(ex["question"])
            conversation = [
                {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
            prompt_text = tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt_text)

        # 生成 4 个样本（temperature=0.6）
        params_multi = SamplingParams(
            n=4,  # 每个 prompt 生成 4 个样本
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            presence_penalty=1.5,
            max_tokens=MAX_NEW_TOKENS,
            stop_token_ids=[151643, 151645],
        )
        outputs = llm.generate(prompts, params_multi)

        # 计算无偏 pass@1
        correct_count = 0
        bad_cases = []
        for out, ex in zip(outputs, examples):
            # 获取该问题的 4 个样本
            samples = [o.text for o in out.outputs]
            # 逐个判断是否正确
            correctness = []
            for sample in samples:
                acc, _ = score_fn([sample], [ex])
                correctness.append(acc > 0.5)  # acc=1.0 表示正确

            # 无偏 pass@1 估计：1 - C(n-c, k) / C(n, k)
            # 其中 n=4（总样本数），c=sum(correctness)（正确样本数），k=1
            c = sum(correctness)
            n = len(correctness)
            k = 1
            if c == 0:
                pass_at_1 = 0.0
                # 记录 bad case（取第一个样本）
                bad_cases.append({
                    "input": ex["question"],
                    "prediction": samples[0],
                    "ground_truth": ex["gold_answer"],
                    "error_type": "wrong_answer",
                })
            else:
                # pass@1 = 1 - C(n-c, k) / C(n, k)
                pass_at_1 = 1.0 - (math.comb(n - c, k) / math.comb(n, k))

            correct_count += pass_at_1

        accuracy = correct_count / len(examples) if examples else 0.0
        scores.append(accuracy)
        for bc in bad_cases[:MAX_BAD_CASES_PER_BENCH]:
            bc["error_type"] = f"{bench_name}:{bc.get('error_type', 'wrong_answer')}"
            all_bad_cases.append(bc)
        detail[bench_name] = {"accuracy": accuracy, "total": len(examples), "method": "pass@1"}
        logger.info("%s pass@1: %.4f (%d 条)", bench_name, accuracy, len(examples))

    # 处理大基准：单样本贪心
    params_single = SamplingParams(
        temperature=0,
        max_tokens=MAX_NEW_TOKENS,
        stop_token_ids=[151643, 151645],
    )

    for bench_name, load_fn, prompt_fn, score_fn in large_benchmarks:
        logger.info("评测 %s (单样本贪心) ...", bench_name)
        examples = load_fn(cache_dir=HF_CACHE_DIR)
        if not examples:
            logger.warning("%s 数据集为空，跳过", bench_name)
            continue

        # 构造 prompts
        prompts = []
        for ex in examples:
            user_msg = prompt_fn(ex["question"])
            conversation = [
                {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
            prompt_text = tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt_text)

        # 单次生成（temperature=0）
        outputs = llm.generate(prompts, params_single)
        predictions = [out.outputs[0].text for out in outputs]

        accuracy, bad_cases = score_fn(predictions, examples)
        scores.append(accuracy)
        for bc in bad_cases[:MAX_BAD_CASES_PER_BENCH]:
            bc["error_type"] = f"{bench_name}:{bc.get('error_type', 'wrong_answer')}"
            all_bad_cases.append(bc)
        detail[bench_name] = {"accuracy": accuracy, "total": len(examples), "method": "greedy"}
        logger.info("%s accuracy: %.4f (%d 条)", bench_name, accuracy, len(examples))

    avg_score = sum(scores) / len(scores) if scores else 0.0
    logger.info("平均分 (%s): %.4f", metric_name, avg_score)

    # 保存各 bench 详情
    (output_dir / "eval_detail.json").write_text(
        json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {
        "score": avg_score,
        "metric_name": metric_name,
        "bad_cases": all_bad_cases,
    }


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="LlamaFactory Math SFT + Eval")
    parser.add_argument("--dataset", required=True, help="输入数据集 JSONL 路径")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument(
        "--mode", default="dev", choices=["dev", "test"],
        help="dev=MATH train only（MCTS 迭代）；test=GSM8K test + MATH-500 + AIME（终评）",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = args.mode

    try:
        # 1. 格式转换
        data_dir = output_dir / "sft_data"
        convert_dataset(dataset_path, data_dir)

        # 2. Full SFT
        model_dir = run_sft(data_dir, output_dir)

        # 3. 评测
        results = run_eval(model_dir, output_dir, mode=mode)

    except Exception as exc:
        logger.error("流程失败：%s", exc, exc_info=True)
        results = {
            "score": 0.0,
            "metric_name": "math_train_accuracy" if mode == "dev" else "math_test_avg_accuracy",
            "bad_cases": [],
            "_error": str(exc),
        }

    # 4. 输出
    results_path = output_dir / "results.json"
    results_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("结果已写入：%s (score=%.4f)", results_path, results["score"])


if __name__ == "__main__":
    main()
