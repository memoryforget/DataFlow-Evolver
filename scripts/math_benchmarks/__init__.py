"""Math benchmarks 评测工具包。"""
from .answer_extract import extract_answer, extract_boxed, extract_gsm8k_answer, is_equiv
from .gsm8k import load_gsm8k, build_prompt as gsm8k_prompt, score_predictions as gsm8k_score
from .math500 import load_math500, build_prompt as math500_prompt, score_predictions as math500_score
from .math_train import load_math_train, build_prompt as math_train_prompt, score_predictions as math_train_score
from .aime2024 import load_aime2024, build_prompt as aime_prompt, score_predictions as aime_score

__all__ = [
    "extract_answer", "extract_boxed", "extract_gsm8k_answer", "is_equiv",
    "load_gsm8k", "gsm8k_prompt", "gsm8k_score",
    "load_math500", "math500_prompt", "math500_score",
    "load_math_train", "math_train_prompt", "math_train_score",
    "load_aime2024", "aime_prompt", "aime_score",
]
