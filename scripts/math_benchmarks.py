"""数学基准加载与评分函数。

支持的基准：
    小基准（<50题，需多样本 pass@1）：AIME24, AMC23, CHMath
    大基准（≥50题，单样本贪心）：GSM8K, MATH500, OlympiadBench, Gaokao, Kaoyan, GradeSchool, MinervaMath, GPQA
"""
from __future__ import annotations

import json
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# 答案提取与比较工具
# ---------------------------------------------------------------------------
def extract_boxed_answer(text: str) -> str:
    """从 \\boxed{...} 中提取答案。"""
    if not text:
        return ""
    # 找最后一个 \boxed{...}
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    return matches[-1].strip() if matches else ""


def extract_number(text: str) -> str:
    """提取数值答案（用于 GSM8K/AIME 等纯数字答案基准）。"""
    if not text:
        return ""
    # 先尝试从 \\boxed{} 提取
    boxed = extract_boxed_answer(text)
    if boxed:
        text = boxed
    # 提取数字（支持负数、小数、分数）
    match = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return match.group(0) if match else ""


def normalize_answer(ans: str) -> str:
    """标准化答案（去空格、小写、去标点）。"""
    return re.sub(r"[^\w\s]", "", ans.lower().strip())


# ---------------------------------------------------------------------------
# GSM8K
# ---------------------------------------------------------------------------
def load_gsm8k(split: str = "test", cache_dir: str | None = None) -> list[dict]:
    """加载 GSM8K 数据集（HuggingFace: openai/gsm8k）。"""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=split, cache_dir=cache_dir)
    examples = []
    for item in ds:
        question = item["question"]
        answer = item["answer"].split("####")[-1].strip()  # GSM8K 格式：解释 #### 答案
        examples.append({"question": question, "answer": answer})
    return examples


def gsm8k_prompt(question: str) -> str:
    return question


def gsm8k_score(predictions: list[str], examples: list[dict]) -> tuple[float, list[dict]]:
    """GSM8K 评分：数值精确匹配。"""
    correct = 0
    bad_cases = []
    for pred, ex in zip(predictions, examples):
        pred_num = extract_number(pred)
        gt_num = extract_number(ex["answer"])
        if pred_num == gt_num and pred_num:
            correct += 1
        else:
            bad_cases.append({
                "input": ex["question"],
                "prediction": pred,
                "ground_truth": ex["answer"],
                "error_type": "wrong_answer",
            })
    return correct / len(examples) if examples else 0.0, bad_cases


# ---------------------------------------------------------------------------
# MATH-500
# ---------------------------------------------------------------------------
def load_math500(cache_dir: str | None = None) -> list[dict]:
    """加载 MATH-500 数据集（HuggingFace: HuggingFaceTB/MATH-500）。"""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceTB/MATH-500", split="test", cache_dir=cache_dir)
    examples = []
    for item in ds:
        examples.append({
            "question": item["problem"],
            "answer": item["answer"],
        })
    return examples


def math500_prompt(question: str) -> str:
    return question


def math500_score(predictions: list[str], examples: list[dict]) -> tuple[float, list[dict]]:
    """MATH-500 评分：LaTeX 语义等价（简化版：提取 boxed 内容比较）。"""
    correct = 0
    bad_cases = []
    for pred, ex in zip(predictions, examples):
        pred_ans = extract_boxed_answer(pred)
        gt_ans = extract_boxed_answer(ex["answer"])
        if normalize_answer(pred_ans) == normalize_answer(gt_ans) and pred_ans:
            correct += 1
        else:
            bad_cases.append({
                "input": ex["question"],
                "prediction": pred,
                "ground_truth": ex["answer"],
                "error_type": "wrong_answer",
            })
    return correct / len(examples) if examples else 0.0, bad_cases


# ---------------------------------------------------------------------------
# MATH train (dev 用)
# ---------------------------------------------------------------------------
def load_math_train(data_path: str | None = None, cache_dir: str | None = None) -> list[dict]:
    """加载 MATH competition train set (7500 条公开训练题)。

    data_path 为本地 jsonl 路径（每行含 question/answer），若为 None 则从 HF 加载。
    """
    if data_path and Path(data_path).exists():
        examples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line.strip())
                examples.append({"question": obj["question"], "answer": obj["answer"]})
        return examples
    else:
        # 从 HuggingFace 加载（hendrycks/competition_math train split）
        from datasets import load_dataset
        ds = load_dataset("hendrycks/competition_math", split="train", cache_dir=cache_dir)
        examples = []
        for item in ds:
            examples.append({"question": item["problem"], "answer": item["solution"]})
        return examples


def math_train_prompt(question: str) -> str:
    return question


def math_train_score(predictions: list[str], examples: list[dict]) -> tuple[float, list[dict]]:
    """MATH train 评分：同 MATH-500。"""
    return math500_score(predictions, examples)


# ---------------------------------------------------------------------------
# AIME 2024（小基准，<50题）
# ---------------------------------------------------------------------------
def load_aime2024(cache_dir: str | None = None) -> list[dict]:
    """加载 AIME 2024 数据集（30 题，HuggingFace: AI-MO/aimo-validation-aime）。"""
    from datasets import load_dataset
    ds = load_dataset("AI-MO/aimo-validation-aime", split="train", cache_dir=cache_dir)
    examples = []
    for item in ds:
        examples.append({
            "question": item["problem"],
            "answer": str(item["answer"]),  # AIME 答案为整数
        })
    return examples


def aime_prompt(question: str) -> str:
    return question


def aime_score(predictions: list[str], examples: list[dict]) -> tuple[float, list[dict]]:
    """AIME 评分：整数精确匹配。"""
    correct = 0
    bad_cases = []
    for pred, ex in zip(predictions, examples):
        pred_num = extract_number(pred)
        gt_num = ex["answer"]
        if pred_num == gt_num and pred_num:
            correct += 1
        else:
            bad_cases.append({
                "input": ex["question"],
                "prediction": pred,
                "ground_truth": gt_num,
                "error_type": "wrong_answer",
            })
    return correct / len(examples) if examples else 0.0, bad_cases


# ---------------------------------------------------------------------------
# AMC23（小基准，<50题）
# ---------------------------------------------------------------------------
def load_amc23(cache_dir: str | None = None) -> list[dict]:
    """加载 AMC 2023 数据集（约 25 题，来源: AI-MO/aimo-validation-amc）。"""
    from datasets import load_dataset
    # 注意：实际 AMC 数据集可能需要手动下载或使用不同的 HF repo
    # 此处使用示例路径，如不存在需替换
    try:
        ds = load_dataset("AI-MO/aimo-validation-amc", split="train", cache_dir=cache_dir)
    except Exception:
        # fallback: 返回空列表（需手动配置数据集路径）
        return []
    examples = []
    for item in ds:
        examples.append({
            "question": item["problem"],
            "answer": str(item["answer"]),
        })
    return examples


def amc23_prompt(question: str) -> str:
    return question


def amc23_score(predictions: list[str], examples: list[dict]) -> tuple[float, list[dict]]:
    """AMC 评分：整数或单字母精确匹配。"""
    correct = 0
    bad_cases = []
    for pred, ex in zip(predictions, examples):
        pred_ans = extract_number(pred) or extract_boxed_answer(pred)
        gt_ans = ex["answer"]
        if normalize_answer(pred_ans) == normalize_answer(gt_ans) and pred_ans:
            correct += 1
        else:
            bad_cases.append({
                "input": ex["question"],
                "prediction": pred,
                "ground_truth": gt_ans,
                "error_type": "wrong_answer",
            })
    return correct / len(examples) if examples else 0.0, bad_cases


# ---------------------------------------------------------------------------
# CHMath（小基准，<50题）
# ---------------------------------------------------------------------------
def load_chmath(cache_dir: str | None = None) -> list[dict]:
    """加载中文数学基准 CHMath（<50题，需手动配置数据集路径）。"""
    _ = cache_dir  # 占位参数，实际使用时从本地加载
    # CHMath 通常为本地 jsonl 文件，此处返回空列表作为占位
    # 实际使用时需替换为真实路径或 HF repo
    return []


def chmath_prompt(question: str) -> str:
    return question


def chmath_score(predictions: list[str], examples: list[dict]) -> tuple[float, list[dict]]:
    """CHMath 评分：同 MATH-500。"""
    return math500_score(predictions, examples)


# ---------------------------------------------------------------------------
# 大基准（≥50题）
# ---------------------------------------------------------------------------
def load_olympiadbench(cache_dir: str | None = None) -> list[dict]:
    """加载 OlympiadBench（HuggingFace: KbsdJames/Olympiad-Bench）。"""
    try:
        from datasets import load_dataset
        ds = load_dataset("KbsdJames/Olympiad-Bench", split="test", cache_dir=cache_dir)
        examples = []
        for item in ds:
            examples.append({"question": item["problem"], "answer": item["answer"]})
        return examples
    except Exception:
        return []


def olympiadbench_prompt(question: str) -> str:
    return question


def olympiadbench_score(predictions: list[str], examples: list[dict]) -> tuple[float, list[dict]]:
    return math500_score(predictions, examples)


def load_gaokao(cache_dir: str | None = None) -> list[dict]:
    """加载高考数学基准（本地 jsonl 或 HF，需配置路径）。"""
    _ = cache_dir  # 占位参数，实际使用时从本地加载
    return []


def gaokao_prompt(question: str) -> str:
    return question


def gaokao_score(predictions: list[str], examples: list[dict]) -> tuple[float, list[dict]]:
    return math500_score(predictions, examples)


def load_kaoyan(cache_dir: str | None = None) -> list[dict]:
    """加载考研数学基准（本地 jsonl 或 HF，需配置路径）。"""
    _ = cache_dir  # 占位参数，实际使用时从本地加载
    return []


def kaoyan_prompt(question: str) -> str:
    return question


def kaoyan_score(predictions: list[str], examples: list[dict]) -> tuple[float, list[dict]]:
    return math500_score(predictions, examples)


def load_gradeschool(cache_dir: str | None = None) -> list[dict]:
    """加载 GradeSchool 数学基准（本地 jsonl 或 HF，需配置路径）。"""
    _ = cache_dir  # 占位参数，实际使用时从本地加载
    return []


def gradeschool_prompt(question: str) -> str:
    return question


def gradeschool_score(predictions: list[str], examples: list[dict]) -> tuple[float, list[dict]]:
    return gsm8k_score(predictions, examples)


def load_minervamath(cache_dir: str | None = None) -> list[dict]:
    """加载 MinervaMath 基准（本地 jsonl 或 HF，需配置路径）。"""
    _ = cache_dir  # 占位参数，实际使用时从本地加载
    return []


def minervamath_prompt(question: str) -> str:
    return question


def minervamath_score(predictions: list[str], examples: list[dict]) -> tuple[float, list[dict]]:
    return math500_score(predictions, examples)


def load_gpqa(cache_dir: str | None = None) -> list[dict]:
    """加载 GPQA 数据集（HuggingFace: Idavidrein/gpqa）。"""
    try:
        from datasets import load_dataset
        ds = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train", cache_dir=cache_dir)
        examples = []
        for item in ds:
            examples.append({
                "question": item["Question"],
                "answer": item["Correct Answer"],
            })
        return examples
    except Exception:
        return []


def gpqa_prompt(question: str) -> str:
    return question


def gpqa_score(predictions: list[str], examples: list[dict]) -> tuple[float, list[dict]]:
    """GPQA 评分：答案精确匹配（通常为多选题字母）。"""
    correct = 0
    bad_cases = []
    for pred, ex in zip(predictions, examples):
        pred_ans = extract_boxed_answer(pred) or pred.strip()[:10]
        gt_ans = ex["answer"]
        if normalize_answer(pred_ans) == normalize_answer(gt_ans):
            correct += 1
        else:
            bad_cases.append({
                "input": ex["question"],
                "prediction": pred,
                "ground_truth": gt_ans,
                "error_type": "wrong_answer",
            })
    return correct / len(examples) if examples else 0.0, bad_cases
