"""Math benchmarks 评测工具包。"""
from .answer_extract import extract_answer, extract_boxed, is_equiv
from .gsm8k import load_gsm8k, build_prompt as gsm8k_prompt
from .math500 import load_math500, build_prompt as math500_prompt
from .math_train import load_math_train, build_prompt as math_train_prompt
from .aime2024 import load_aime2024, build_prompt as aime_prompt
from .amc23 import load_amc23, build_prompt as amc23_prompt
from .additional_benchmarks import (
    load_chmath, chmath_prompt,
    load_olympiadbench, olympiadbench_prompt,
    load_gaokao, gaokao_prompt,
    load_kaoyan, kaoyan_prompt,
    load_gradeschool, gradeschool_prompt,
    load_minervamath, minervamath_prompt,
    load_gpqa, gpqa_prompt,
)

__all__ = [
    "extract_answer", "extract_boxed", "is_equiv",
    "load_gsm8k", "gsm8k_prompt",
    "load_math500", "math500_prompt",
    "load_math_train", "math_train_prompt",
    "load_aime2024", "aime_prompt",
    "load_amc23", "amc23_prompt",
    "load_chmath", "chmath_prompt",
    "load_olympiadbench", "olympiadbench_prompt",
    "load_gaokao", "gaokao_prompt",
    "load_kaoyan", "kaoyan_prompt",
    "load_gradeschool", "gradeschool_prompt",
    "load_minervamath", "minervamath_prompt",
    "load_gpqa", "gpqa_prompt",
]

