"""LlamaFactory Full SFT + Math Benchmark 评测脚本。

契约：
    python scripts/llamafactory_math_sft_eval.py --dataset <ds.jsonl> --output <out_dir> [--mode dev|test]

模式：
    dev（默认）：MATH train 全量 (7500 条) 评测，驱动 MCTS 奖励信号。
    test：GSM8K test + MATH-500 + AIME 2024 完整套件，仅终评用。

流程：
    1. 数据格式转换（DataFlow JSONL → LlamaFactory alpaca 格式）
    2. LlamaFactory Full SFT（Qwen3-8B, DeepSpeed ZeRO-2, 16 GPU）
    3. vLLM 推理（直接加载 fine-tuned 模型）对选定 bench 评测
    4. bench 取平均 accuracy → results.json

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
BASE_MODEL = os.environ.get("SFT_BASE_MODEL", "Qwen/Qwen3.5-9B")
NUM_EPOCHS = float(os.environ.get("SFT_NUM_EPOCHS", "3"))
LEARNING_RATE = float(os.environ.get("SFT_LEARNING_RATE", "2e-5"))
BATCH_SIZE = int(os.environ.get("SFT_BATCH_SIZE", "4"))
GRAD_ACCUM = int(os.environ.get("SFT_GRAD_ACCUM", "4"))
NUM_GPUS = int(os.environ.get("SFT_NUM_GPUS", "8"))
TP_SIZE = int(os.environ.get("EVAL_TP_SIZE", "1"))
DP_SIZE = int(os.environ.get("EVAL_DP_SIZE", "8"))
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
# 2. LlamaFactory Full SFT
# ---------------------------------------------------------------------------
def run_sft(data_dir: Path, output_dir: Path) -> Path:
    """调用 LlamaFactory CLI 执行全量 SFT 训练。返回模型输出目录。"""
    model_dir = output_dir / "sft_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    sft_config = {
        "model_name_or_path": BASE_MODEL,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "full",
        "dataset_dir": str(data_dir),
        "dataset": "train",
        "template": "qwen3",
        "output_dir": str(model_dir),
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
    import yaml
    sft_yaml_path.write_text(yaml.dump(sft_config, default_flow_style=False), encoding="utf-8")

    # 执行训练：让 LlamaFactory 自己管理分布式启动（内部会调 torchrun）。
    # 把当前解释器所在 bin 目录放到 PATH 最前面，确保 LlamaFactory 内部
    # subprocess 调用的 torchrun 解析到同一环境。
    cmd = [
        sys.executable, "-m", "llamafactory.cli", "train",
        str(sft_yaml_path),
    ]
    env = os.environ.copy()
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env["FORCE_TORCHRUN"] = "1"
    env["NNODES"] = "1"
    env["NPROC_PER_NODE"] = str(NUM_GPUS)
    env["MASTER_PORT"] = "29500"
    logger.info("开始 Full SFT 训练：%s", " ".join(cmd))
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=3600, cwd=str(output_dir), env=env
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
    mode='test': GSM8K test + MATH-500 + AIME 2024 完整套件，仅终评。
    """
    from vllm import LLM, SamplingParams

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from math_benchmarks import (
        load_gsm8k, gsm8k_prompt, gsm8k_score,
        load_math500, math500_prompt, math500_score,
        load_math_train, math_train_prompt, math_train_score,
        load_aime2024, aime_prompt, aime_score,
    )

    logger.info("加载 vLLM 模型：%s (TP=%d, DP=%d)", model_dir, TP_SIZE, DP_SIZE)
    llm = LLM(
        model=str(model_dir),
        tensor_parallel_size=TP_SIZE,
        data_parallel_size=DP_SIZE,
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=True,
    )
    params = SamplingParams(temperature=0, max_tokens=MAX_NEW_TOKENS)

    if mode == "test":
        benchmarks = [
            ("GSM8K-test", lambda **kw: load_gsm8k(split="test", **kw), gsm8k_prompt, gsm8k_score),
            ("MATH-500", load_math500, math500_prompt, math500_score),
            ("AIME-2024", load_aime2024, aime_prompt, aime_score),
        ]
        metric_name = "math_test_avg_accuracy"
    else:
        math_train_path = os.environ.get("MATH_TRAIN_PATH", None)
        benchmarks = [
            ("MATH-train", lambda **kw: load_math_train(data_path=math_train_path, **kw), math_train_prompt, math_train_score),
        ]
        metric_name = "math_train_accuracy"

    scores = []
    all_bad_cases: list[dict] = []
    detail = {}

    tokenizer = llm.get_tokenizer()

    for bench_name, load_fn, prompt_fn, score_fn in benchmarks:
        logger.info("评测 %s ...", bench_name)
        examples = load_fn(cache_dir=HF_CACHE_DIR)

        # 使用 chat template 构造含 system prompt 的完整输入
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

        outputs = llm.generate(prompts, params)
        predictions = [out.outputs[0].text for out in outputs]

        accuracy, bad_cases = score_fn(predictions, examples)
        scores.append(accuracy)
        for bc in bad_cases[:MAX_BAD_CASES_PER_BENCH]:
            bc["error_type"] = f"{bench_name}:{bc.get('error_type', 'wrong_answer')}"
            all_bad_cases.append(bc)
        detail[bench_name] = {"accuracy": accuracy, "total": len(examples)}
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
