"""其他数学基准的占位符实现。

包含：CHMath, OlympiadBench, Gaokao, Kaoyan, GradeSchool, MinervaMath, GPQA
这些基准需要后续配置具体的数据源（本地 jsonl 或 HuggingFace repo）。
"""
from __future__ import annotations

from .answer_extract import extract_answer, is_equiv


# ---------------------------------------------------------------------------
# CHMath（中文数学，小基准 <50题）
# ---------------------------------------------------------------------------
def load_chmath(cache_dir: str | None = None) -> list[dict]:
    """加载 CHMath 数据集（需手动配置数据源）。"""
    _ = cache_dir
    # TODO: 配置本地 jsonl 或 HF repo
    return []


def chmath_prompt(question: str) -> str:
    return question


def chmath_score(predictions: list[str], examples: list[dict]) -> tuple[float, list[dict]]:
    """CHMath 判分：LaTeX 答案比较。"""
    correct = 0
    bad_cases: list[dict] = []
    for pred_text, ex in zip(predictions, examples):
        pred_answer = extract_answer(pred_text, mode="auto")
        gold = ex["gold_answer"]
        match = pred_answer is not None and is_equiv(pred_answer, gold)
        if match:
            correct += 1
        else:
            bad_cases.append({
                "input": ex["question"],
                "prediction": pred_answer or "(extraction failed)",
                "ground_truth": gold,
                "error_type": "wrong_answer",
            })
    return correct / len(examples) if examples else 0.0, bad_cases


# ---------------------------------------------------------------------------
# OlympiadBench（大基准 ≥50题）
# ---------------------------------------------------------------------------
def load_olympiadbench(cache_dir: str | None = None) -> list[dict]:
    """加载 OlympiadBench。"""
    from datasets import load_dataset
    try:
        ds = load_dataset("KbsdJames/Olympiad-Bench", split="test", cache_dir=cache_dir)
        return [{"question": r["problem"], "gold_answer": r["answer"]} for r in ds]
    except Exception:
        return []


def olympiadbench_prompt(question: str) -> str:
    return question


def olympiadbench_score(predictions: list[str], examples: list[dict]) -> tuple[float, list[dict]]:
    return chmath_score(predictions, examples)  # 复用 LaTeX 判分


# ---------------------------------------------------------------------------
# Gaokao（高考数学，大基准）
# ---------------------------------------------------------------------------
def load_gaokao(cache_dir: str | None = None) -> list[dict]:
    _ = cache_dir
    return []


def gaokao_prompt(question: str) -> str:
    return question


def gaokao_score(predictions: list[str], examples: list[dict]) -> tuple[float, list[dict]]:
    return chmath_score(predictions, examples)


# ---------------------------------------------------------------------------
# Kaoyan（考研数学，大基准）
# ---------------------------------------------------------------------------
def load_kaoyan(cache_dir: str | None = None) -> list[dict]:
    _ = cache_dir
    return []


def kaoyan_prompt(question: str) -> str:
    return question


def kaoyan_score(predictions: list[str], examples: list[dict]) -> tuple[float, list[dict]]:
    return chmath_score(predictions, examples)


# ---------------------------------------------------------------------------
# GradeSchool（小学数学，大基准）
# ---------------------------------------------------------------------------
def load_gradeschool(cache_dir: str | None = None) -> list[dict]:
    _ = cache_dir
    return []


def gradeschool_prompt(question: str) -> str:
    return question


def gradeschool_score(predictions: list[str], examples: list[dict]) -> tuple[float, list[dict]]:
    """GradeSchool 判分：数值答案比较（类似 GSM8K）。"""
    from .answer_extract import normalize_number
    correct = 0
    bad_cases: list[dict] = []
    for pred_text, ex in zip(predictions, examples):
        pred_answer = extract_answer(pred_text, mode="auto")
        gold = ex["gold_answer"]
        match = False
        if pred_answer is not None:
            pred_num = normalize_number(pred_answer)
            gold_num = normalize_number(gold)
            if pred_num is not None and gold_num is not None:
                match = abs(pred_num - gold_num) < 1e-3
            else:
                match = is_equiv(pred_answer, gold)
        if match:
            correct += 1
        else:
            bad_cases.append({
                "input": ex["question"],
                "prediction": pred_answer or "(extraction failed)",
                "ground_truth": gold,
                "error_type": "wrong_answer",
            })
    return correct / len(examples) if examples else 0.0, bad_cases


# ---------------------------------------------------------------------------
# MinervaMath（大基准）
# ---------------------------------------------------------------------------
def load_minervamath(cache_dir: str | None = None) -> list[dict]:
    _ = cache_dir
    return []


def minervamath_prompt(question: str) -> str:
    return question


def minervamath_score(predictions: list[str], examples: list[dict]) -> tuple[float, list[dict]]:
    return chmath_score(predictions, examples)


# ---------------------------------------------------------------------------
# GPQA（科学问答，大基准）
# ---------------------------------------------------------------------------
def load_gpqa(cache_dir: str | None = None) -> list[dict]:
    """加载 GPQA。"""
    from datasets import load_dataset
    try:
        ds = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train", cache_dir=cache_dir)
        return [{"question": r["Question"], "gold_answer": r["Correct Answer"]} for r in ds]
    except Exception:
        return []


def gpqa_prompt(question: str) -> str:
    return question


def gpqa_score(predictions: list[str], examples: list[dict]) -> tuple[float, list[dict]]:
    """GPQA 判分：字符串精确匹配（多选题字母答案）。"""
    correct = 0
    bad_cases: list[dict] = []
    for pred_text, ex in zip(predictions, examples):
        pred_answer = extract_answer(pred_text, mode="auto")
        gold = ex["gold_answer"]
        match = pred_answer is not None and is_equiv(pred_answer, gold)
        if match:
            correct += 1
        else:
            bad_cases.append({
                "input": ex["question"],
                "prediction": pred_answer or "(extraction failed)",
                "ground_truth": gold,
                "error_type": "wrong_answer",
            })
    return correct / len(examples) if examples else 0.0, bad_cases
