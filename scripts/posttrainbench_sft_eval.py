"""PostTrainBench Tier-2 桥接：单卡全量 SFT + 官方 evaluate.py 评测。

遵守框架训练桥接契约（engine/trainer_wrapper.py）：
    <train_python> scripts/posttrainbench_sft_eval.py --mode dev \\
        --base-model <本地模型路径|${PTB_BASE_MODEL}> \\
        --ptb-task-dir <.../PostTrainBench/src/eval/tasks/<bench>|${PTB_TASK_DIR}> \\
        --dev-limit 150 --benchmark <bench> \\
        --dataset <ds.jsonl>  --output <out_dir>          # 后两个由框架自动追加
    脚本回写：<out_dir>/results.json = {"score": float, "metric_name": str, "bad_cases": []}

模式：
    dev  ：单卡全量 SFT（1 epoch, lr 2e-5, 无 LoRA）→ 标准 HF 目录 <out>/sft_model，
           然后在 df-eval 环境（SFT_EVAL_PYTHON）跑官方 <task>/evaluate.py --limit <dev-limit>，
           只取聚合 metrics 的 accuracy 作 score。
    test ：只评测不训练（须给 --model-dir），跑官方 evaluate.py --limit -1（全量终评）。

合规（防泄露）：
    - 训练数据来自本框架自行合成的 <dataset>，绝不使用任何 benchmark 的 test 数据；
    - 评测只读官方 --json-output-file 的**聚合标量**（accuracy），绝不解析/落盘 per-sample 的
      test 题面或答案；bad_cases 恒为 []。

格式对齐（硬约束）：
    训练数据按目标模型族（qwen/gemma/smollm）选对应 LlamaFactory 模板，与评测端
    templates/*.jinja 的 chat 模板一致；system prompt / 最终答案格式按 benchmark 对齐
    （见 scripts/ptb_benchmarks.py）。
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import shlex
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ptb_benchmarks as ptb  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("posttrainbench_sft_eval")

# --- 训练超参（固定 recipe；可用 env 覆盖，但不纳入 MCTS 搜索空间）---
NUM_EPOCHS = float(os.environ.get("SFT_NUM_EPOCHS", "1"))
LEARNING_RATE = float(os.environ.get("SFT_LEARNING_RATE", "2.0e-5"))
BATCH_SIZE = int(os.environ.get("SFT_BATCH_SIZE", "1"))
GRAD_ACCUM = int(os.environ.get("SFT_GRAD_ACCUM", "8"))
NUM_GPUS = int(os.environ.get("SFT_NUM_GPUS", "1"))            # 单卡 1×H100
CUTOFF_LEN = int(os.environ.get("SFT_CUTOFF_LEN", "4096"))
WARMUP_RATIO = float(os.environ.get("SFT_WARMUP_RATIO", "0.03"))
GRAD_CKPT = os.environ.get("SFT_GRAD_CKPT", "1") == "1"        # 4B 显存吃紧时开梯度检查点
FLASH_ATTN = os.environ.get("SFT_FLASH_ATTN", "sdpa").strip()
# 4B 全量 SFT 单卡吃紧时可开 DeepSpeed ZeRO-2（优化器 CPU offload）：SFT_DEEPSPEED=zero2_offload
DEEPSPEED_MODE = os.environ.get("SFT_DEEPSPEED", "").strip()
SFT_TRAIN_TIMEOUT_SEC = int(os.environ.get("SFT_TRAIN_TIMEOUT_SEC", "5400"))
SFT_TEMPLATE_OVERRIDE = os.environ.get("SFT_TEMPLATE", "").strip()

# --- 评测参数 ---
EVAL_GPU_MEM_UTIL = os.environ.get("EVAL_GPU_MEM_UTIL", "").strip()
EVAL_CUDA_VISIBLE_DEVICES = os.environ.get("EVAL_CUDA_VISIBLE_DEVICES", "0").strip()
# inspect 家族 evaluate.py 的并发（官方默认仅 2，全量评测会极慢）；只影响吞吐，不影响分数。
EVAL_MAX_CONNECTIONS = os.environ.get("EVAL_MAX_CONNECTIONS", "").strip()
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "inspectai")
DEFAULT_HF_ENDPOINT = os.environ.get("PTB_HF_ENDPOINT", "https://hf-mirror.com")


# ---------------------------------------------------------------------------
# 模型族 / 模板（对齐评测端 templates/*.jinja，逻辑与官方 evaluate.py model_type 一致）
# ---------------------------------------------------------------------------
def model_family(base_model: str) -> str:
    low = base_model.lower()
    for fam in ("qwen", "gemma", "smollm", "llama"):
        if fam in low:
            return fam
    cfg = Path(base_model) / "config.json"
    if cfg.exists():
        try:
            arch = json.loads(cfg.read_text(encoding="utf-8"))["architectures"][0].lower()
        except (json.JSONDecodeError, OSError, KeyError, IndexError):
            arch = ""
        for fam in ("qwen", "gemma", "smollm", "llama"):
            if fam in arch:
                return fam
    raise ValueError(f"无法从 {base_model} 判定模型族（qwen/gemma/smollm/llama）")


def llamafactory_template(family: str) -> str:
    """把模型族映射到 LlamaFactory 内置对话模板（与评测端 jinja 对齐）。"""
    return {
        "qwen": "qwen",       # ChatML，对齐 qwen3.jinja
        "gemma": "gemma",     # <start_of_turn>，对齐 gemma3.jinja
        "smollm": "smollm",   # ChatML + think 模式，对齐 smollm.jinja
        "llama": "llama3",
    }.get(family, "qwen")


# ---------------------------------------------------------------------------
# 数据转换：DataFlow 合成 JSONL → LlamaFactory ShareGPT（对齐 benchmark 系统提示/格式）
# ---------------------------------------------------------------------------
_INSTRUCTION_KEYS = ("instruction", "question", "prompt", "input", "query")
_OUTPUT_KEYS = ("generated_cot", "output", "answer", "response", "completion", "solution", "code")


def _first_present(obj: dict, keys: tuple[str, ...]) -> str:
    for k in keys:
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v
        if v not in (None, "", [], {}):
            return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    return ""


def convert_dataset(
    src_path: Path,
    dst_dir: Path,
    benchmark: str,
    base_model: str,
    cutoff_len: int = CUTOFF_LEN,
) -> Path:
    """把合成 JSONL 转成 LlamaFactory sharegpt，并按 benchmark 对齐 system prompt / 工具调用格式。

    - 通用：{system(benchmark), human(instruction), gpt(output)}。
    - 工具调用类（bfcl）：若样本带 tools，则写入 sharegpt 的 tools 字段；assistant 值为工具调用文本。
    - 超长样本用目标模型 tokenizer 过滤（tokenizer 不可用时按字符粗估）。
    返回 train.json 路径，并写出 dataset_info.json。
    """
    bench = ptb.get_benchmark(benchmark)
    system_prompt = bench["sft_system_prompt"]
    is_tool = bool(bench.get("tool_calling"))

    # 借鉴 ANDES：把内容级结构约束（code / tool_call）也写进 SFT system prompt，
    # 训练时强化「答案用对应 fence 包裹」的结构，与合成端的 --task 约束一致（ShareGPT schema 不变）。
    fmt_instruction = ptb.format_instruction_for(benchmark)
    if fmt_instruction:
        system_prompt = f"{system_prompt}\n{fmt_instruction}"
        logger.info(
            "benchmark=%s format_requirement=%s：已在 SFT system prompt 追加结构约束。",
            benchmark, ptb.format_requirement_for(benchmark),
        )

    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001 —— tokenizer 不可用时降级为字符粗估
        logger.warning("加载 tokenizer 失败（%s），改用字符数粗估过滤超长样本。", exc)

    rows: list[dict] = []
    total = kept = filtered = 0
    with src_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            total += 1
            instruction = _first_present(obj, _INSTRUCTION_KEYS)
            output = _first_present(obj, _OUTPUT_KEYS)
            if not instruction or not output:
                continue

            conversation = [
                {"from": "system", "value": system_prompt},
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": output},
            ]
            row: dict = {"conversations": conversation}
            if is_tool and obj.get("tools"):
                tools = obj["tools"]
                row["tools"] = tools if isinstance(tools, str) else json.dumps(tools, ensure_ascii=False)

            if _too_long(tokenizer, system_prompt, instruction, output, cutoff_len):
                filtered += 1
                continue
            rows.append(row)
            kept += 1

    if filtered:
        logger.warning("过滤超长样本 %d/%d（cutoff_len=%d）", filtered, total, cutoff_len)
    if not rows:
        raise ValueError(f"数据集为空或字段不匹配（instruction/output 均需非空）：{src_path}")

    dst_dir.mkdir(parents=True, exist_ok=True)
    train_file = dst_dir / "train.json"
    train_file.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    columns = {"messages": "conversations"}
    if is_tool:
        columns["tools"] = "tools"
    info = {"train": {"file_name": "train.json", "formatting": "sharegpt", "columns": columns}}
    (dst_dir / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("格式转换完成：%d 条（benchmark=%s, tool=%s）→ %s", kept, benchmark, is_tool, train_file)
    return train_file


def _too_long(tokenizer, system_prompt: str, instruction: str, output: str, cutoff_len: int) -> bool:
    if tokenizer is not None:
        try:
            conv = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": output},
            ]
            text = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
            return len(tokenizer.encode(text, add_special_tokens=True)) > cutoff_len
        except Exception:  # noqa: BLE001
            pass
    return int((len(system_prompt) + len(instruction) + len(output)) * 0.4) > cutoff_len


# ---------------------------------------------------------------------------
# 单卡全量 SFT（LlamaFactory）
# ---------------------------------------------------------------------------
def _deepspeed_config(mode: str) -> dict | None:
    """按需生成单卡可用的 DeepSpeed 配置（ZeRO-2 + 优化器 CPU offload，缓解 4B 单卡显存）。"""
    if mode != "zero2_offload":
        return None
    return {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "contiguous_gradients": True,
            "overlap_comm": True,
        },
    }


def build_sft_config(base_model: str, data_dir: Path, model_dir: Path, template: str) -> dict:
    """构造 LlamaFactory 全量 SFT 配置（单卡、1 epoch、lr 2e-5、无 LoRA）。"""
    cfg = {
        "model_name_or_path": base_model,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "full",            # 全量 SFT，不使用 LoRA
        "dataset_dir": str(data_dir.resolve()),
        "dataset": "train",
        "template": template,
        "cutoff_len": CUTOFF_LEN,
        "output_dir": str(model_dir.resolve()),
        "num_train_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "per_device_train_batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": WARMUP_RATIO,
        "bf16": True,
        "flash_attn": FLASH_ATTN,
        "optim": "adamw_torch",
        "gradient_checkpointing": GRAD_CKPT,
        "max_grad_norm": 1.0,
        "logging_steps": 10,
        "save_strategy": "no",
        "report_to": "none",
        "overwrite_output_dir": True,
        "dataloader_pin_memory": False,
    }
    return cfg


def run_sft(base_model: str, data_dir: Path, output_dir: Path, template: str) -> Path:
    """执行单卡全量 SFT，返回标准 HF 模型目录 <output>/sft_model。"""
    import yaml

    model_dir = output_dir / "sft_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    sft_config = build_sft_config(base_model, data_dir, model_dir, template)

    config_dir = output_dir / "sft_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    ds = _deepspeed_config(DEEPSPEED_MODE)
    if ds is not None:
        ds_path = config_dir / "ds.json"
        ds_path.write_text(json.dumps(ds, indent=2), encoding="utf-8")
        sft_config["deepspeed"] = str(ds_path)

    sft_yaml = config_dir / "sft.yaml"
    sft_yaml.write_text(yaml.dump(sft_config, default_flow_style=False), encoding="utf-8")

    cmd = [sys.executable, "-u", "-m", "llamafactory.cli", "train", str(sft_yaml.resolve())]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    apply_env_prefix_fixups(env, sys.executable)
    env.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("SFT_CUDA_VISIBLE_DEVICES", "0"))
    if NUM_GPUS > 1:
        env["FORCE_TORCHRUN"] = "1"
        env["NNODES"] = "1"
        env["NPROC_PER_NODE"] = str(NUM_GPUS)
        env.setdefault("MASTER_PORT", "29500")

    log_path = output_dir / "sft_train.log"
    logger.info("开始单卡全量 SFT（epochs=%s lr=%s template=%s ds=%s）：%s",
                NUM_EPOCHS, LEARNING_RATE, template, DEEPSPEED_MODE or "off", " ".join(cmd))
    rc = _stream_subprocess(cmd, cwd=str(output_dir), env=env, log_path=log_path,
                            timeout_sec=SFT_TRAIN_TIMEOUT_SEC)
    if rc != 0:
        raise RuntimeError(f"SFT 训练失败（exit={rc}），日志：{log_path}")
    if not any(model_dir.glob("*.safetensors")):
        raise RuntimeError(f"SFT 未产出 HF safetensors 权重：{model_dir}")
    _sync_model_metadata_from_base(base_model, model_dir)
    logger.info("SFT 完成，模型目录：%s", model_dir)
    return model_dir


def _sync_model_metadata_from_base(base_model: str, model_dir: Path) -> None:
    """用 base 模型元数据覆盖 SFT 输出，避免 tokenizer/config 漂移影响官方评测。"""
    base = Path(base_model)
    if not base.exists():
        logger.warning("base model 不存在：%s，跳过模型元数据同步", base)
        return
    names = [
        "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt",
        "special_tokens_map.json", "added_tokens.json", "config.json", "generation_config.json",
    ]
    copied = [
        name for name in names
        if (base / name).exists() and (shutil.copy2(base / name, model_dir / name) or True)
    ]
    jinja = model_dir / "chat_template.jinja"
    if jinja.exists():
        jinja.unlink()
    logger.info("已用 base 模型元数据覆盖 %s：%s", model_dir, copied)


def apply_env_prefix_fixups(env: dict, python_exe: str) -> dict:
    """按目标解释器所属 conda env 补齐 `conda activate` 才会设置的变量（原地改 env 并返回）。

    框架用**绝对路径**直接调 df-train / df-eval 的 python，不经过 conda activate，因此
    三类隐式依赖会缺失：
      - LD_LIBRARY_PATH：env 自带的 libstdc++ 比系统新，缺路径时 sqlite3/vllm 会因
        `CXXABI_1.3.15 not found` 导入失败；
      - CUDA_HOME：DeepSpeed 找不到 env 内的 nvcc，会回退到不存在的 /usr/local/cuda；
      - PATH：inspect_ai 以子进程形式起 `vllm` 命令行，env 的 bin 不在 PATH 时报
        `No such file or directory: 'vllm'`。
    """
    prefix = Path(python_exe).resolve().parent.parent
    bin_dir = prefix / "bin"
    if bin_dir.is_dir():
        prev = env.get("PATH", "")
        parts = [p for p in prev.split(os.pathsep) if p]
        if str(bin_dir) not in parts:
            env["PATH"] = os.pathsep.join([str(bin_dir), *parts])
    lib_dir = prefix / "lib"
    if lib_dir.is_dir():
        prev = env.get("LD_LIBRARY_PATH", "")
        parts = [p for p in prev.split(os.pathsep) if p]
        if str(lib_dir) not in parts:
            env["LD_LIBRARY_PATH"] = os.pathsep.join([str(lib_dir), *parts])
    nvcc = prefix / "bin" / "nvcc"
    if nvcc.exists() and not Path(env.get("CUDA_HOME", "/nonexistent"), "bin", "nvcc").exists():
        env["CUDA_HOME"] = str(prefix)
        env["CUDA_PATH"] = str(prefix)
    return env


def _stream_subprocess(cmd: list[str], *, cwd: str, env: dict, log_path: Path, timeout_sec: int) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as logf:
        logf.write(f"CMD: {' '.join(cmd)}\nCWD: {cwd}\n")
        logf.flush()
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=cwd,
                                env=env, start_new_session=True)
        try:
            return proc.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired as exc:
            _kill_tree(proc.pid)
            raise TimeoutError(f"SFT 子进程超时（>{timeout_sec}s），日志：{log_path}") from exc


def _kill_tree(pid: int) -> None:
    import signal
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        time.sleep(2)
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


# ---------------------------------------------------------------------------
# 官方 evaluate.py 评测（在 df-eval 环境；只取聚合 accuracy）
# ---------------------------------------------------------------------------
def run_official_eval(
    benchmark: str,
    ptb_task_dir: Path,
    model_path: Path,
    output_dir: Path,
    limit: int,
) -> dict:
    """在 df-eval（SFT_EVAL_PYTHON）里跑官方 <task>/evaluate.py，返回官方聚合 metrics。

    只读 --json-output-file 的聚合标量，绝不解析 per-sample test 内容。
    """
    bench = ptb.get_benchmark(benchmark)
    eval_python = os.environ.get("SFT_EVAL_PYTHON", "").strip()
    if not eval_python:
        raise RuntimeError("未配置 SFT_EVAL_PYTHON（评测须在 df-eval 环境运行）。")

    evaluate_py = ptb_task_dir / "evaluate.py"
    if not evaluate_py.exists():
        raise RuntimeError(f"官方评测脚本不存在：{evaluate_py}")
    templates_dir = ptb.templates_dir_for(str(ptb_task_dir))
    metrics_file = output_dir / "metrics.json"

    argv = ptb.build_eval_argv(
        benchmark,
        model_path=str(Path(model_path).resolve()),
        templates_dir=str(templates_dir),
        limit=limit,
        json_output_file=str(metrics_file.resolve()),
        gpu_memory_utilization=(float(EVAL_GPU_MEM_UTIL) if EVAL_GPU_MEM_UTIL else None),
        max_connections=(int(EVAL_MAX_CONNECTIONS) if EVAL_MAX_CONNECTIONS else None),
    )
    cmd = [eval_python, str(evaluate_py), *argv]

    env = os.environ.copy()
    apply_env_prefix_fixups(env, eval_python)
    env["VLLM_API_KEY"] = VLLM_API_KEY
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env.setdefault("HF_ENDPOINT", DEFAULT_HF_ENDPOINT)
    if EVAL_CUDA_VISIBLE_DEVICES:
        env["CUDA_VISIBLE_DEVICES"] = EVAL_CUDA_VISIBLE_DEVICES
    if bench.get("needs_judge") and not env.get("OPENAI_API_KEY"):
        logger.warning("benchmark=%s 需要 GPT judge，但 OPENAI_API_KEY 未设置，评测可能失败。", benchmark)

    log_path = output_dir / "eval.log"
    logger.info("启动官方评测：cwd=%s cmd=%s", ptb_task_dir, " ".join(shlex.quote(c) for c in cmd))
    rc = _stream_subprocess(cmd, cwd=str(ptb_task_dir), env=env, log_path=log_path,
                            timeout_sec=int(os.environ.get("PTB_EVAL_TIMEOUT_SEC", str(SFT_TRAIN_TIMEOUT_SEC))))
    if rc != 0:
        raise RuntimeError(f"官方评测失败（exit={rc}），日志：{log_path}")
    if not metrics_file.exists():
        raise RuntimeError(f"评测未产出 metrics 文件：{metrics_file}")
    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    score_key = bench["score_key"]
    if score_key not in metrics:
        raise RuntimeError(f"metrics 缺少聚合分键 {score_key!r}：{sorted(metrics.keys())}")
    return metrics


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def _resolve_base_model(arg_value: str) -> str:
    val = (arg_value or "").strip() or os.environ.get("PTB_BASE_MODEL", "").strip() \
        or os.environ.get("SFT_BASE_MODEL", "").strip()
    if not val:
        raise RuntimeError("未提供 base model（--base-model 或 PTB_BASE_MODEL / SFT_BASE_MODEL）。")
    return val


def _resolve_task_dir(arg_value: str) -> Path:
    val = (arg_value or "").strip() or os.environ.get("PTB_TASK_DIR", "").strip()
    if not val:
        raise RuntimeError("未提供 --ptb-task-dir（或 PTB_TASK_DIR）。")
    return Path(val).resolve()


def _write_results(output_dir: Path, score: float, benchmark: str, metrics: dict | None = None) -> None:
    """按框架契约回写 results.json：保留 score 三件套，附带官方聚合 metrics。"""
    results = {
        "score": float(score),
        "metric_name": benchmark,
        "bad_cases": [],
    }
    if metrics is not None:
        results["metrics"] = metrics
        if "stderr" in metrics:
            results["stderr"] = metrics["stderr"]
    (output_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("结果已写入：%s（%s=%.4f）", output_dir / "results.json", benchmark, score)


def main() -> None:
    parser = argparse.ArgumentParser(description="PostTrainBench 单卡全量 SFT + 官方评测桥接")
    parser.add_argument("--dataset", help="合成训练数据 JSONL（dev 训练必需）")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--model-dir", default="", help="已训练模型目录；提供后只评测不训练")
    parser.add_argument("--mode", default="dev", choices=["dev", "test"],
                        help="dev=SFT+评测(--limit dev-limit)；test=只评测(--limit -1 全量终评，需 --model-dir)")
    parser.add_argument("--base-model", default="", help="本地 base model 路径（或 ${PTB_BASE_MODEL}）")
    parser.add_argument("--ptb-task-dir", default="", help="PostTrainBench 任务目录（或 ${PTB_TASK_DIR}）")
    parser.add_argument("--dev-limit", type=int, default=150, help="dev 评测采样上限")
    parser.add_argument("--benchmark", default="", help="benchmark 名（缺省由 --ptb-task-dir 反查）")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    ptb_task_dir = _resolve_task_dir(args.ptb_task_dir)
    benchmark = ptb.resolve_benchmark_name(
        args.benchmark or os.environ.get("PTB_BENCHMARK", ""), str(ptb_task_dir)
    )
    model_dir = Path(args.model_dir).resolve() if args.model_dir else None
    logger.info("mode=%s benchmark=%s task_dir=%s model_dir=%s",
                args.mode, benchmark, ptb_task_dir, model_dir)

    try:
        if model_dir is not None:
            # 只评测已有模型（终评 / 复评）
            if not model_dir.exists():
                raise RuntimeError(f"评测模型目录不存在：{model_dir}")
            limit = -1 if args.mode == "test" else args.dev_limit
            metrics = run_official_eval(benchmark, ptb_task_dir, model_dir, output_dir, limit)
        elif args.mode == "test":
            raise RuntimeError("--mode test 需要 --model-dir（只评测不训练）。")
        else:
            # dev：单卡全量 SFT + 官方评测
            if not args.dataset:
                raise RuntimeError("dev 模式必须提供 --dataset。")
            base_model = _resolve_base_model(args.base_model)
            family = model_family(base_model)
            template = SFT_TEMPLATE_OVERRIDE or llamafactory_template(family)
            logger.info("base_model=%s family=%s template=%s", base_model, family, template)
            data_dir = output_dir / "sft_data"
            convert_dataset(Path(args.dataset), data_dir, benchmark, base_model)
            sft_model = run_sft(base_model, data_dir, output_dir, template)
            metrics = run_official_eval(benchmark, ptb_task_dir, sft_model, output_dir, args.dev_limit)
    except Exception as exc:  # noqa: BLE001
        # 训练/评测未跑成（基础设施/致命错误）：非零退出，让上层 TrainerWrapper 判 failed=True，
        # 而非伪装成「评测 0 分」。真·0 分只会在评测正常跑完后经 _write_results 落盘。
        logger.error("流程失败：%s", exc, exc_info=True)
        sys.exit(1)

    score = float(metrics[ptb.get_benchmark(benchmark)["score_key"]])
    _write_results(output_dir, score, benchmark, metrics)


if __name__ == "__main__":
    main()
