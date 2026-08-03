"""Additional benchmark loaders.

All benchmarks are loaded from fixed paths under:
`data/benchmarks/limo_eval/`

判分统一在 `scripts/vllm_eval_standalone.py` 的 `_grade` 里做，这里只负责把各家
原始字段规整成 `{question, gold_answer, choices?, gold_letter?}`。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .answer_extract import extract_boxed, is_equiv

_OPTION_LABELS = "ABCDE"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PROJECT_LIMO_EVAL_DATA_DIR = _PROJECT_ROOT / "data" / "benchmarks" / "limo_eval"
_DEFAULT_LOCAL_BENCHMARK_PATHS: dict[str, list[str]] = {
    "GSM8K": ["gsm8k/test.jsonl"],
    "AIME2024": ["aime/test.jsonl"],
    "AMC23": ["amc/test.jsonl"],
    "MATH500": ["math/test.jsonl"],
    "CHMath": ["cn_math_2024/test.jsonl"],
    "OlympiadBench": ["olympiadbench/test.jsonl"],
    "Gaokao": ["gaokao/test.jsonl"],
    "Kaoyan": ["kaoyan/test.jsonl"],
    "GradeSchool": ["grade_school_math/test.jsonl"],
    "MinervaMath": ["minerva/test.jsonl"],
    "GPQA": ["gpqa/test.jsonl"],
}


def _project_benchmark_path(
    bench_name: str,
    default_local_paths: list[str] | None,
) -> Path:
    rel_paths = default_local_paths or _DEFAULT_LOCAL_BENCHMARK_PATHS.get(bench_name, [])
    if not rel_paths:
        raise KeyError(f"{bench_name} has no declared benchmark path")
    rel_path = Path(rel_paths[0])
    return rel_path if rel_path.is_absolute() else _PROJECT_LIMO_EVAL_DATA_DIR / rel_path


def _choice_label(idx: int) -> str:
    if idx < len(_OPTION_LABELS):
        return _OPTION_LABELS[idx]
    return f"OPTION_{idx + 1}"


def _candidate_question(row: dict[str, Any]) -> str:
    for key in ("question", "Question", "problem", "Problem", "prompt", "instruction", "query"):
        value = row.get(key)
        if value:
            return str(value).strip()
    return ""


def _candidate_answer(row: dict[str, Any]) -> str:
    for key in (
        "gold_answer",
        "correct_answer",
        "answer",
        "final_answer",
        "target",
        "label",
        "output",
    ):
        value = row.get(key)
        if value not in (None, ""):
            return str(value).strip()
    solution = str(row.get("solution", "")).strip()
    if solution:
        return extract_boxed(solution) or solution
    return ""


def _candidate_gold_letter(row: dict[str, Any]) -> str | None:
    for key in ("gold_letter", "answer_letter", "correct_choice", "correct_option"):
        value = row.get(key)
        if isinstance(value, str):
            letter = value.strip().upper().rstrip(".")
            if letter in _OPTION_LABELS:
                return letter
    return None


def _coerce_choices(row: dict[str, Any]) -> list[tuple[str, str]] | None:
    raw = row.get("choices") or row.get("options")
    if raw is None:
        raw = {k: row[k] for k in _OPTION_LABELS if row.get(k) not in (None, "")}
        if not raw:
            raw = {k: row[k.lower()] for k in _OPTION_LABELS if row.get(k.lower()) not in (None, "")}
        if not raw:
            return None

    choices: list[tuple[str, str]] = []
    if isinstance(raw, dict):
        for label, value in raw.items():
            label_str = str(label).strip().upper()
            value_str = str(value).strip()
            if label_str and value_str:
                choices.append((label_str, value_str))
    elif isinstance(raw, list):
        for idx, item in enumerate(raw):
            if isinstance(item, dict):
                label = str(item.get("label", _choice_label(idx))).strip().upper()
                text = str(
                    item.get("text", item.get("content", item.get("value", "")))
                ).strip()
            else:
                label = _choice_label(idx)
                text = str(item).strip()
            if text:
                choices.append((label, text))

    return choices or None


def _stable_mcq_choices(question: str, correct: str, wrongs: list[str]) -> tuple[list[tuple[str, str]], str]:
    pool = [(correct, True), *[(w, False) for w in wrongs if w]]
    digest = hashlib.md5(question.encode("utf-8")).digest()
    keyed = []
    for idx, (text, is_correct) in enumerate(pool):
        key = hashlib.md5(digest + str(idx).encode("utf-8") + text.encode("utf-8")).hexdigest()
        keyed.append((key, text, is_correct))
    keyed.sort(key=lambda item: item[0])

    labeled: list[tuple[str, str]] = []
    gold_letter = "A"
    for idx, (_, text, is_correct) in enumerate(keyed):
        label = _choice_label(idx)
        labeled.append((label, text))
        if is_correct:
            gold_letter = label
    return labeled, gold_letter


def _normalize_row(row: dict[str, Any]) -> dict[str, Any] | None:
    question = _candidate_question(row)
    if not question:
        return None

    example: dict[str, Any] = {"question": question}

    if "Correct Answer" in row:
        correct = str(row.get("Correct Answer", "")).strip()
        wrongs = [
            str(row.get("Incorrect Answer 1", "")).strip(),
            str(row.get("Incorrect Answer 2", "")).strip(),
            str(row.get("Incorrect Answer 3", "")).strip(),
        ]
        choices, gold_letter = _stable_mcq_choices(question, correct, wrongs)
        example["gold_answer"] = correct
        example["choices"] = choices
        example["gold_letter"] = gold_letter
        return example

    gold_answer = _candidate_answer(row)
    if not gold_answer:
        return None

    example["gold_answer"] = gold_answer
    choices = _coerce_choices(row)
    if choices:
        example["choices"] = choices
        gold_letter = _candidate_gold_letter(row)
        if gold_letter is None:
            for label, choice_text in choices:
                if is_equiv(choice_text, gold_answer):
                    gold_letter = label
                    break
        if gold_letter is not None:
            example["gold_letter"] = gold_letter
    return example


def _load_json_file(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
        return rows

    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        for key in ("data", "examples", "items", "rows"):
            value = data.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    raise ValueError(f"Unsupported data format: {path}")


def _load_local_rows(path: Path) -> list[dict[str, Any]]:
    if path.is_file():
        return _load_json_file(path)

    if path.is_dir():
        files = sorted(path.glob("*.jsonl")) + sorted(path.glob("*.json"))
        rows: list[dict[str, Any]] = []
        for file in files:
            rows.extend(_load_json_file(file))
        return rows

    raise FileNotFoundError(f"Local data path does not exist: {path}")


def _normalize_examples(rows: list[dict[str, Any]], bench_name: str) -> list[dict[str, Any]]:
    examples = []
    for row in rows:
        ex = _normalize_row(row)
        if ex is not None:
            examples.append(ex)
    if not examples:
        raise ValueError(f"{bench_name} loaded but no usable examples were found")
    return examples


def load_benchmark_rows(
    *,
    bench_name: str,
    default_local_paths: list[str] | None = None,
) -> list[dict[str, Any]]:
    path = _project_benchmark_path(bench_name, default_local_paths)
    if not path.exists():
        raise FileNotFoundError(
            f"{bench_name} dataset is missing: {path}. "
            "Place the benchmark file in the project's fixed relative path."
        )
    return _load_local_rows(path)


def load_benchmark_examples(
    *,
    bench_name: str,
    default_local_paths: list[str] | None = None,
) -> list[dict[str, Any]]:
    rows = load_benchmark_rows(
        bench_name=bench_name,
        default_local_paths=default_local_paths,
    )
    return _normalize_examples(rows, bench_name)


def load_chmath(cache_dir: str | None = None) -> list[dict]:
    return load_benchmark_examples(
        bench_name="CHMath",
        default_local_paths=["cn_math_2024/test.jsonl"],
    )


def chmath_prompt(question: str) -> str:
    return question


def load_olympiadbench(cache_dir: str | None = None) -> list[dict]:
    return load_benchmark_examples(
        bench_name="OlympiadBench",
        default_local_paths=["olympiadbench/test.jsonl"],
    )


def olympiadbench_prompt(question: str) -> str:
    return question


def load_gaokao(cache_dir: str | None = None) -> list[dict]:
    return load_benchmark_examples(
        bench_name="Gaokao",
        default_local_paths=["gaokao/test.jsonl"],
    )


def gaokao_prompt(question: str) -> str:
    return question


def load_kaoyan(cache_dir: str | None = None) -> list[dict]:
    return load_benchmark_examples(
        bench_name="Kaoyan",
        default_local_paths=["kaoyan/test.jsonl"],
    )


def kaoyan_prompt(question: str) -> str:
    return question


def load_gradeschool(cache_dir: str | None = None) -> list[dict]:
    return load_benchmark_examples(
        bench_name="GradeSchool",
        default_local_paths=["grade_school_math/test.jsonl"],
    )


def gradeschool_prompt(question: str) -> str:
    return question


def load_minervamath(cache_dir: str | None = None) -> list[dict]:
    return load_benchmark_examples(
        bench_name="MinervaMath",
        default_local_paths=["minerva/test.jsonl"],
    )


def minervamath_prompt(question: str) -> str:
    return question


def load_gpqa(cache_dir: str | None = None) -> list[dict]:
    return load_benchmark_examples(
        bench_name="GPQA",
        default_local_paths=["gpqa/test.jsonl"],
    )


def gpqa_prompt(question: str) -> str:
    return question

