"""LlamaFactory LoRA SFT + Math Benchmark 评测脚本。

契约：
    python scripts/llamafactory_math_sft_eval.py --dataset <ds.jsonl> --output <out_dir>

流程：
    1. 数据格式转换（DataFlow JSONL → LlamaFactory alpaca 格式）
    2. LlamaFactory LoRA SFT（Qwen3-8B, DeepSpeed ZeRO-2, 16 GPU）
    3. vLLM 推理（加载 base+adapter）对 GSM8K / MATH-500 / AIME 2024 评测
    4. 三个 bench 取平均 accuracy → results.json

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
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 配置常量（可通过环境变量覆盖）
# ---------------------------------------------------------------------------
BASE_MODEL = os.environ.get("SFT_BASE_MODEL", "Qwen/Qwen3-8B")
LORA_RANK = int(os.environ.get("SFT_LORA_RANK", "16"))
LORA_ALPHA = int(os.environ.get("SFT_LORA_ALPHA", "32"))
NUM_EPOCHS = float(os.environ.get("SFT_NUM_EPOCHS", "3"))
LEARNING_RATE = float(os.environ.get("SFT_LEARNING_RATE", "2e-4"))
BATCH_SIZE = int(os.environ.get("SFT_BATCH_SIZE", "4"))
GRAD_ACCUM = int(os.environ.get("SFT_GRAD_ACCUM", "4"))
NUM_GPUS = int(os.environ.get("SFT_NUM_GPUS", "16"))
TP_SIZE = int(os.environ.get("EVAL_TP_SIZE", "16"))
MAX_MODEL_LEN = int(os.environ.get("EVAL_MAX_MODEL_LEN", "4096"))
MAX_NEW_TOKENS = int(os.environ.get("EVAL_MAX_NEW_TOKENS", "2048"))
HF_CACHE_DIR = os.environ.get("HF_CACHE_DIR", None)
MAX_BAD_CASES_PER_BENCH = 10


# ---------------------------------------------------------------------------
# 1. 数据格式转换
# ---------------------------------------------------------------------------
def convert_dataset(src_path: Path, dst_dir: Path) -> Path:
    """将 DataFlow JSONL 转为 LlamaFactory alpaca 格式的 JSON 文件。"""
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
            instruction = obj.get("instruction", obj.get("question", obj.get("input", "")))
            output = obj.get("generated_cot", obj.get("output", obj.get("answer", "")))
            if instruction and output:
                rows.append({"instruction": instruction, "input": "", "output": output})

    if not rows:
        raise ValueError(f"数据集为空或字段不匹配：{src_path}")

    dst_dir.mkdir(parents=True, exist_ok=True)
    train_file = dst_dir / "train.json"
    train_file.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    # dataset_info.json 供 LlamaFactory 识别
    info = {"train": {"file_name": "train.json"}}
    (dst_dir / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("格式转换完成：%d 条训练样本 → %s", len(rows), train_file)
    return train_file


# ---------------------------------------------------------------------------
# 2. LlamaFactory LoRA SFT
# ---------------------------------------------------------------------------
def run_sft(data_dir: Path, output_dir: Path) -> Path:
    """调用 LlamaFactory CLI 执行 LoRA SFT 训练。返回 adapter 目录。"""
    adapter_dir = output_dir / "lora_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    sft_config = {
        "model_name_or_path": BASE_MODEL,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_target": "all",
        "dataset_dir": str(data_dir),
        "dataset": "train",
        "template": "qwen3",
        "output_dir": str(adapter_dir),
        "num_train_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "per_device_train_batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "bf16": True,
        "logging_steps": 10,
        "save_strategy": "no",
        "report_to": "none",
        "overwrite_output_dir": True,
    }

    # DeepSpeed ZeRO-2 配置
    ds_config = {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {"device": "none"},
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "overlap_comm": True,
        },
    }

    # 写临时配置文件
    config_dir = output_dir / "sft_configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    ds_config_path = config_dir / "ds_z2.json"
    ds_config_path.write_text(json.dumps(ds_config, indent=2), encoding="utf-8")
    sft_config["deepspeed"] = str(ds_config_path)

    sft_yaml_path = config_dir / "sft.yaml"
    # LlamaFactory 用 yaml 或直接 CLI args；这里写 yaml
    import yaml
    sft_yaml_path.write_text(yaml.dump(sft_config, default_flow_style=False), encoding="utf-8")

    # 执行训练
    cmd = [
        "torchrun",
        f"--nproc_per_node={NUM_GPUS}",
        "--master_port=29500",
        "-m", "llamafactory.cli", "train",
        str(sft_yaml_path),
    ]
    logger.info("开始 SFT 训练：%s", " ".join(cmd))
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=3600, cwd=str(output_dir)
    )
    log_path = output_dir / "sft_train.log"
    log_path.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")

    if result.returncode != 0:
        raise RuntimeError(f"SFT 训练失败 (exit={result.returncode})，日志：{log_path}")

    logger.info("SFT 训练完成，adapter 保存于：%s", adapter_dir)
    return adapter_dir


# ---------------------------------------------------------------------------
# 3. Math Benchmark 评测
# ---------------------------------------------------------------------------
def run_eval(adapter_dir: Path, output_dir: Path) -> dict:
    """用 vLLM 加载 base+adapter 对三个 bench 评测，返回 results dict。"""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from math_benchmarks import (
        load_gsm8k, gsm8k_prompt, gsm8k_score,
        load_math500, math500_prompt, math500_score,
        load_aime2024, aime_prompt, aime_score,
    )

    logger.info("加载 vLLM 模型：%s (TP=%d)", BASE_MODEL, TP_SIZE)
    llm = LLM(
        model=BASE_MODEL,
        enable_lora=True,
        max_lora_rank=LORA_RANK,
        tensor_parallel_size=TP_SIZE,
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=True,
    )
    params = SamplingParams(temperature=0, max_tokens=MAX_NEW_TOKENS)
    lora_req = LoRARequest("sft_adapter", 1, str(adapter_dir))

    benchmarks = [
        ("GSM8K", load_gsm8k, gsm8k_prompt, gsm8k_score),
        ("MATH-500", load_math500, math500_prompt, math500_score),
        ("AIME-2024", load_aime2024, aime_prompt, aime_score),
    ]

    scores = []
    all_bad_cases: list[dict] = []
    detail = {}

    for bench_name, load_fn, prompt_fn, score_fn in benchmarks:
        logger.info("评测 %s ...", bench_name)
        examples = load_fn(cache_dir=HF_CACHE_DIR)
        prompts = [prompt_fn(ex["question"]) for ex in examples]

        outputs = llm.generate(prompts, params, lora_request=lora_req)
        predictions = [out.outputs[0].text for out in outputs]

        accuracy, bad_cases = score_fn(predictions, examples)
        scores.append(accuracy)
        # 每个 bench 最多取 MAX_BAD_CASES_PER_BENCH 条 bad cases
        for bc in bad_cases[:MAX_BAD_CASES_PER_BENCH]:
            bc["error_type"] = f"{bench_name}:{bc.get('error_type', 'wrong_answer')}"
            all_bad_cases.append(bc)
        detail[bench_name] = {"accuracy": accuracy, "total": len(examples)}
        logger.info("%s accuracy: %.4f (%d 条)", bench_name, accuracy, len(examples))

    avg_score = sum(scores) / len(scores) if scores else 0.0
    logger.info("平均分 (math_avg_accuracy): %.4f", avg_score)

    # 保存各 bench 详情
    (output_dir / "eval_detail.json").write_text(
        json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {
        "score": avg_score,
        "metric_name": "math_avg_accuracy",
        "bad_cases": all_bad_cases,
    }


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="LlamaFactory Math SFT + Eval")
    parser.add_argument("--dataset", required=True, help="输入数据集 JSONL 路径")
    parser.add_argument("--output", required=True, help="输出目录")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. 格式转换
        data_dir = output_dir / "sft_data"
        convert_dataset(dataset_path, data_dir)

        # 2. LoRA SFT
        adapter_dir = run_sft(data_dir, output_dir)

        # 3. 评测
        results = run_eval(adapter_dir, output_dir)

    except Exception as exc:
        logger.error("流程失败：%s", exc, exc_info=True)
        results = {
            "score": 0.0,
            "metric_name": "math_avg_accuracy",
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
