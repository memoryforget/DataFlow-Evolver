"""通用数学答案提取与比较工具。

支持：
- \boxed{} 提取
- #### <number> 提取 (GSM8K 格式)
- 末尾数值提取
- 数值/表达式语义等价判定
"""
from __future__ import annotations

import re
from fractions import Fraction


def extract_boxed(text: str) -> str | None:
    """提取 LaTeX \\boxed{...} 中的内容（支持嵌套大括号）。"""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    depth = 0
    start = idx + len("\\boxed{")
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            if depth == 0:
                return text[start:i].strip()
            depth -= 1
    return None


def extract_gsm8k_answer(text: str) -> str | None:
    """提取 GSM8K 格式答案：#### <number>"""
    match = re.search(r"####\s*([^\n]+)", text)
    if match:
        return match.group(1).strip()
    return None


def extract_last_number(text: str) -> str | None:
    """提取文本中最后出现的数值（整数或小数，含负号）。"""
    matches = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if matches:
        return matches[-1].replace(",", "")
    return None


def extract_answer(text: str, mode: str = "auto") -> str | None:
    """根据模式提取答案。

    mode:
        "boxed"  - 只提取 \\boxed{}
        "gsm8k"  - 只提取 #### <number>
        "number" - 只提取末尾数值
        "auto"   - 依次尝试 boxed → gsm8k → number
    """
    if mode == "boxed":
        return extract_boxed(text)
    elif mode == "gsm8k":
        return extract_gsm8k_answer(text)
    elif mode == "number":
        return extract_last_number(text)
    else:
        result = extract_boxed(text)
        if result is not None:
            return result
        result = extract_gsm8k_answer(text)
        if result is not None:
            return result
        return extract_last_number(text)


def normalize_number(s: str) -> float | None:
    """尝试将字符串规范化为浮点数。"""
    s = s.strip().replace(",", "").replace(" ", "")
    s = s.rstrip(".")
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100
        except ValueError:
            return None
    if "/" in s:
        try:
            return float(Fraction(s))
        except (ValueError, ZeroDivisionError):
            pass
    try:
        return float(s)
    except ValueError:
        return None


def normalize_latex(s: str) -> str:
    """轻量 LaTeX 规范化：去除空格、\\text{}、\\mathrm{} 等修饰。"""
    s = s.strip()
    s = re.sub(r"\\(?:text|mathrm|mathbf|mathit|operatorname)\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:left|right|displaystyle|,|;|!|quad|qquad)", "", s)
    s = re.sub(r"\s+", "", s)
    return s


def is_equiv(pred: str, gold: str, tolerance: float = 1e-6) -> bool:
    """判断两个答案是否等价。

    优先尝试数值比较，否则做字符串规范化后精确匹配。
    """
    if pred is None or gold is None:
        return False

    pred_num = normalize_number(pred)
    gold_num = normalize_number(gold)
    if pred_num is not None and gold_num is not None:
        return abs(pred_num - gold_num) < tolerance

    pred_norm = normalize_latex(pred)
    gold_norm = normalize_latex(gold)
    if pred_norm == gold_norm:
        return True

    pred_lower = pred_norm.lower()
    gold_lower = gold_norm.lower()
    return pred_lower == gold_lower
