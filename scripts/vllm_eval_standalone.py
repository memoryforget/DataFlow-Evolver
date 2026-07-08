"""独立的 vLLM 评测脚本。

用法：
    # TP=4, DP=1（直接 LLM 类，单进程）
    python scripts/vllm_eval_standalone.py \
        --model-dir <model_path> \
        --output-dir <output_path> \
        --mode dev \
        --tp-size 4

    # TP=4, DP=2（自动启动 vllm serve，通过 OpenAI API 推理）
    python scripts/vllm_eval_standalone.py \
        --model-dir <model_path> \
        --output-dir <output_path> \
        --mode dev \
        --tp-size 4 \
        --dp-size 2
"""
import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SERVE_PORT = int(os.environ.get("VLLM_SERVE_PORT", "18902"))
SERVE_HOST = "127.0.0.1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--mode", type=str, default="dev", choices=["dev", "test"])
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--dp-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    return parser.parse_args()


def _wait_for_server(base_url: str, timeout: int = 300) -> bool:
    """等待 vllm serve 就绪。"""
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request(f"{base_url}/v1/models")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def _generate_via_api(base_url: str, model: str, messages_list: list[list[dict]],
                      max_tokens: int, concurrency: int = 64) -> list[str]:
    """通过 OpenAI Chat Completions API 并发推理。"""
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    results = [""] * len(messages_list)

    def _call(idx: int) -> tuple[int, str]:
        payload = json.dumps({
            "model": model,
            "messages": messages_list[idx],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }).encode()
        req = urllib.request.Request(
            f"{base_url}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read())
        return idx, data["choices"][0]["message"]["content"]

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for idx, text in pool.map(lambda i: _call(i), range(len(messages_list))):
            results[idx] = text
    return results


def _load_benchmarks(mode: str):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from math_benchmarks import (
        load_gsm8k, gsm8k_prompt,
        load_math500, math500_prompt,
        load_math_train, math_train_prompt,
        load_aime2024, aime_prompt,
        load_amc23, amc23_prompt,
        load_chmath, chmath_prompt,
        load_olympiadbench, olympiadbench_prompt,
        load_gaokao, gaokao_prompt,
        load_kaoyan, kaoyan_prompt,
        load_gradeschool, gradeschool_prompt,
        load_minervamath, minervamath_prompt,
        load_gpqa, gpqa_prompt,
    )
    if mode == "test":
        return [
            ("gsm8k", load_gsm8k, gsm8k_prompt),
            ("math500", load_math500, math500_prompt),
            ("aime2024", load_aime2024, aime_prompt),
            ("amc23", load_amc23, amc23_prompt),
            ("olympiadbench", load_olympiadbench, olympiadbench_prompt),
            ("chmath", load_chmath, chmath_prompt),
            ("gaokao", load_gaokao, gaokao_prompt),
            ("kaoyan", load_kaoyan, kaoyan_prompt),
            ("gradeschool", load_gradeschool, gradeschool_prompt),
            ("minervamath", load_minervamath, minervamath_prompt),
            ("gpqa", load_gpqa, gpqa_prompt),
        ]
    return [("math_train", load_math_train, math_train_prompt)]


def _build_messages(examples: list, prompt_fn) -> list[list[dict]]:
    """构建 chat messages 列表，供 /v1/chat/completions 使用。"""
    messages_list = []
    for ex in examples:
        sys_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
        user_prompt = prompt_fn(ex["question"])
        choices = ex.get("choices")
        if choices:
            option_lines = "\n".join(f"{label}. {text}" for label, text in choices)
            user_prompt = f"{user_prompt}\n\nOptions:\n{option_lines}"
        messages_list.append([
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ])
    return messages_list


def _build_prompts(examples: list, prompt_fn, tokenizer) -> list[str]:
    """构建带 chat template 的 prompt 列表（DP=1 offline 模式使用）。"""
    messages_list = _build_messages(examples, prompt_fn)
    return [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in messages_list
    ]


def run_eval_offline(model_dir: Path, mode: str, tp_size: int,
                     max_model_len: int, max_new_tokens: int) -> dict:
    """DP=1: 直接用 vLLM LLM 类推理。"""
    from vllm import LLM, SamplingParams
    from math_benchmarks.grader import check_is_correct
    from math_benchmarks.answer_extract import extract_answer

    llm = LLM(
        model=str(model_dir),
        tensor_parallel_size=tp_size,
        max_model_len=max_model_len,
        trust_remote_code=True,
        enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    benchmarks = _load_benchmarks(mode)

    bench_scores = {}
    all_bad_cases = []
    MAX_BAD_CASES_PER_BENCH = 30
    for bench_name, load_fn, prompt_fn in benchmarks:
        try:
            examples = load_fn()
            logger.info("评测 %s: %d 条样本", bench_name, len(examples))
            prompts = _build_prompts(examples, prompt_fn, tokenizer)
            outputs = llm.generate(prompts, sampling_params)
            correct = 0
            for i, ex in enumerate(examples):
                pred_text = outputs[i].outputs[0].text
                pred_answer = extract_answer(pred_text, bench_name)
                if check_is_correct(pred_answer, ex["gold_answer"]):
                    correct += 1
                elif len(all_bad_cases) < MAX_BAD_CASES_PER_BENCH * len(benchmarks):
                    all_bad_cases.append({
                        "input": ex["question"],
                        "prediction": pred_text[:500],
                        "extracted_answer": pred_answer,
                        "ground_truth": ex["gold_answer"],
                        "error_type": f"{bench_name}:wrong_answer",
                    })
            accuracy = correct / len(examples)
            bench_scores[bench_name] = accuracy
            logger.info("%s accuracy: %.4f (%d/%d)", bench_name, accuracy, correct, len(examples))
        except Exception as e:
            logger.error("评测 %s 失败: %s", bench_name, e, exc_info=True)
            bench_scores[bench_name] = 0.0

    if not bench_scores:
        raise RuntimeError("所有 benchmark 评测失败")
    avg_score = sum(bench_scores.values()) / len(bench_scores)
    logger.info("平均得分: %.4f", avg_score)
    return {"score": avg_score, "metric_name": f"{mode}_accuracy",
            "bench_scores": bench_scores, "bad_cases": all_bad_cases}


def _cleanup_vllm_workers():
    """清理可能残留的 vLLM GPU worker 进程（EngineCore_DP*, Worker_TP* 等）。"""
    import glob
    my_uid = os.getuid()
    for proc_dir in glob.glob("/proc/[0-9]*"):
        try:
            if os.stat(proc_dir).st_uid != my_uid:
                continue
            cmdline_path = os.path.join(proc_dir, "cmdline")
            with open(cmdline_path, "rb") as f:
                cmdline = f.read().decode("utf-8", errors="replace")
            if any(tag in cmdline for tag in ("VLLM::Worker", "VLLM::EngineCore", "VLLM::DPCoordinator")):
                pid = int(os.path.basename(proc_dir))
                os.kill(pid, signal.SIGKILL)
                logger.info("清理残留 vLLM worker: PID %d", pid)
        except (OSError, ValueError, ProcessLookupError):
            continue


def run_eval_serve(model_dir: Path, mode: str, tp_size: int, dp_size: int,
                   max_model_len: int, max_new_tokens: int) -> dict:
    """DP>1: 启动 vllm serve 子进程，通过 OpenAI Chat API 推理。"""
    from math_benchmarks.grader import check_is_correct
    from math_benchmarks.answer_extract import extract_answer

    base_url = f"http://{SERVE_HOST}:{SERVE_PORT}"
    model_str = str(model_dir)

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_str,
        "--host", SERVE_HOST,
        "--port", str(SERVE_PORT),
        "--tensor-parallel-size", str(tp_size),
        "--data-parallel-size", str(dp_size),
        "--max-model-len", str(max_model_len),
        "--trust-remote-code",
        "--dtype", "bfloat16",
        "--enforce-eager",
        "--gpu-memory-utilization", "0.9",
    ]
    logger.info("启动 vllm serve: TP=%d DP=%d, port=%d", tp_size, dp_size, SERVE_PORT)
    serve_proc = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )

    try:
        if not _wait_for_server(base_url, timeout=300):
            raise RuntimeError("vllm serve 启动超时（300s）")
        logger.info("vllm serve 就绪")

        benchmarks = _load_benchmarks(mode)

        bench_scores = {}
        all_bad_cases = []
        MAX_BAD_CASES_PER_BENCH = 30
        for bench_name, load_fn, prompt_fn in benchmarks:
            try:
                examples = load_fn()
                logger.info("评测 %s: %d 条样本", bench_name, len(examples))
                messages_list = _build_messages(examples, prompt_fn)
                pred_texts = _generate_via_api(base_url, model_str, messages_list, max_new_tokens)
                correct = 0
                for i, ex in enumerate(examples):
                    pred_answer = extract_answer(pred_texts[i], bench_name)
                    if check_is_correct(pred_answer, ex["gold_answer"]):
                        correct += 1
                    elif len(all_bad_cases) < MAX_BAD_CASES_PER_BENCH * len(benchmarks):
                        all_bad_cases.append({
                            "input": ex["question"],
                            "prediction": pred_texts[i][:500],
                            "extracted_answer": pred_answer,
                            "ground_truth": ex["gold_answer"],
                            "error_type": f"{bench_name}:wrong_answer",
                        })
                accuracy = correct / len(examples)
                bench_scores[bench_name] = accuracy
                logger.info("%s accuracy: %.4f (%d/%d)",
                            bench_name, accuracy, correct, len(examples))
            except Exception as e:
                logger.error("评测 %s 失败: %s", bench_name, e, exc_info=True)
                bench_scores[bench_name] = 0.0

        if not bench_scores:
            raise RuntimeError("所有 benchmark 评测失败")
        avg_score = sum(bench_scores.values()) / len(bench_scores)
        logger.info("平均得分: %.4f", avg_score)
        return {"score": avg_score, "metric_name": f"{mode}_accuracy",
                "bench_scores": bench_scores, "bad_cases": all_bad_cases}
    finally:
        logger.info("正在停止 vllm serve...")
        try:
            os.killpg(os.getpgid(serve_proc.pid), signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass
        try:
            serve_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(serve_proc.pid), signal.SIGKILL)
            serve_proc.wait(timeout=5)
        # 清理可能残留的 vLLM worker 进程（它们可能不在同一进程组）
        _cleanup_vllm_workers()
        logger.info("vllm serve 已停止")


def main():
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.dp_size > 1:
            results = run_eval_serve(
                model_dir=model_dir, mode=args.mode,
                tp_size=args.tp_size, dp_size=args.dp_size,
                max_model_len=args.max_model_len,
                max_new_tokens=args.max_new_tokens,
            )
        else:
            results = run_eval_offline(
                model_dir=model_dir, mode=args.mode,
                tp_size=args.tp_size,
                max_model_len=args.max_model_len,
                max_new_tokens=args.max_new_tokens,
            )

        results_path = output_dir / "results.json"
        results_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("结果已写入：%s (score=%.4f)", results_path, results["score"])

    except Exception as e:
        logger.error("评测失败：%s", e, exc_info=True)
        results = {
            "score": 0.0,
            "metric_name": f"{args.mode}_accuracy",
            "bad_cases": [],
            "_error": str(e),
        }
        results_path = output_dir / "results.json"
        results_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

