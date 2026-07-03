"""Math benchmarks 评测工具包。"""
from .answer_extract import extract_answer, extract_boxed, extract_gsm8k_answer, is_equiv
from .gsm8k import load_gsm8k, build_prompt as gsm8k_prompt, score_predictions as gsm8k_score
from .math500 import load_math500, build_prompt as math500_prompt, score_predictions as math500_score
from .math_train import load_math_train, build_prompt as math_train_prompt, score_predictions as math_train_score, SYSTEM_PROMPT
from .aime2024 import load_aime2024, build_prompt as aime_prompt, score_predictions as aime_score
from .amc23 import load_amc23, build_prompt as amc23_prompt, score_predictions as amc23_score
from .additional_benchmarks import (
    load_chmath, chmath_prompt, chmath_score,
    load_olympiadbench, olympiadbench_prompt, olympiadbench_score,
    load_gaokao, gaokao_prompt, gaokao_score,
    load_kaoyan, kaoyan_prompt, kaoyan_score,
    load_gradeschool, gradeschool_prompt, gradeschool_score,
    load_minervamath, minervamath_prompt, minervamath_score,
    load_gpqa, gpqa_prompt, gpqa_score,
)

__all__ = [
    "extract_answer", "extract_boxed", "extract_gsm8k_answer", "is_equiv",
    "load_gsm8k", "gsm8k_prompt", "gsm8k_score",
    "load_math500", "math500_prompt", "math500_score",
    "load_math_train", "math_train_prompt", "math_train_score", "SYSTEM_PROMPT",
    "load_aime2024", "aime_prompt", "aime_score",
    "load_amc23", "amc23_prompt", "amc23_score",
    "load_chmath", "chmath_prompt", "chmath_score",
    "load_olympiadbench", "olympiadbench_prompt", "olympiadbench_score",
    "load_gaokao", "gaokao_prompt", "gaokao_score",
    "load_kaoyan", "kaoyan_prompt", "kaoyan_score",
    "load_gradeschool", "gradeschool_prompt", "gradeschool_score",
    "load_minervamath", "minervamath_prompt", "minervamath_score",
    "load_gpqa", "gpqa_prompt", "gpqa_score",
]

