from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from statistics import median
from typing import Any

import yaml

PROTOCOL_VERSION = 3


class RuntimeConfigError(ValueError):
    """Raised when the external runtime configuration is invalid."""


def load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeConfigError(f"configuration root must be an object: {path}")
    return _expand_environment(raw)


def load_request(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeConfigError("request root must be a JSON object")
    if raw.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeConfigError(
            f"unsupported request protocol_version={raw.get('protocol_version')!r}; "
            f"expected {PROTOCOL_VERSION}"
        )
    stage = raw.get("stage")
    if stage not in {"baseline", "periodic"}:
        raise RuntimeConfigError(f"unsupported request stage: {stage!r}")
    training_performed = raw.get("training_performed")
    if not isinstance(training_performed, bool):
        raise RuntimeConfigError("training_performed must be boolean")
    if stage == "baseline":
        if training_performed:
            raise RuntimeConfigError(
                "baseline request must set training_performed=false"
            )
    else:
        if not training_performed:
            raise RuntimeConfigError(
                "periodic request must set training_performed=true"
            )
        dataset_path = raw.get("dataset_path")
        if not isinstance(dataset_path, str) or not dataset_path.strip():
            raise RuntimeConfigError("periodic request must provide dataset_path")
    benchmarks = raw.get("expected_benchmarks")
    if (
        not isinstance(benchmarks, list)
        or not benchmarks
        or not all(isinstance(item, str) and item.strip() for item in benchmarks)
    ):
        raise RuntimeConfigError("expected_benchmarks must be a non-empty string list")
    bad_cases = raw.get("bad_cases_per_benchmark")
    if isinstance(bad_cases, bool) or not isinstance(bad_cases, int) or bad_cases < 1:
        raise RuntimeConfigError("bad_cases_per_benchmark must be a positive integer")
    return raw


def normalize_sft_dataset(
    source: Path,
    destination: Path,
    *,
    user_field: str,
    assistant_field: str,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    skipped_blank = 0
    with (
        source.open(encoding="utf-8") as input_handle,
        destination.open("w", encoding="utf-8", newline="\n") as output_handle,
    ):
        for line_number, line in enumerate(input_handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise RuntimeConfigError(
                    f"dataset line {line_number} is not a JSON object"
                )
            user = record.get(user_field)
            assistant = record.get(assistant_field)
            if not isinstance(user, str) or not user.strip():
                skipped_blank += 1
                continue
            if not isinstance(assistant, str) or not assistant.strip():
                skipped_blank += 1
                continue
            normalized = {"instruction": user, "input": "", "output": assistant}
            output_handle.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            rows += 1
    if rows == 0:
        raise RuntimeConfigError("normalized SFT dataset is empty")
    return {
        "source_path": str(source.resolve()),
        "normalized_path": str(destination.resolve()),
        "rows": rows,
        "skipped_blank_rows": skipped_blank,
        "user_field": user_field,
        "assistant_field": assistant_field,
    }


def audit_sft_token_lengths(
    dataset_path: Path,
    *,
    model_path: str,
    cutoff_len: int,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    lengths: list[int] = []
    for record in _read_jsonl(dataset_path):
        rendered = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": record["instruction"]},
                {"role": "assistant", "content": record["output"]},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        lengths.append(len(tokenizer.encode(rendered, add_special_tokens=False)))
    ordered = sorted(lengths)
    total_tokens = sum(ordered)
    retained_tokens = sum(min(length, cutoff_len) for length in ordered)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": median(ordered),
        "p90": _percentile(ordered, 0.90),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
        "max": ordered[-1],
        "cutoff_len": cutoff_len,
        "rows_over_cutoff": sum(length > cutoff_len for length in ordered),
        "retained_token_fraction": retained_tokens / total_tokens
        if total_tokens
        else 1.0,
    }


def train_sft(
    request: Mapping[str, Any],
    config: Mapping[str, Any],
    runtime_dir: Path,
) -> tuple[Path, dict[str, str]]:
    sft = _mapping(config.get("sft"), "sft")
    source = Path(str(request["dataset_path"])).resolve()
    if not source.is_file():
        raise RuntimeConfigError(f"SFT dataset does not exist: {source}")

    dataset_dir = runtime_dir / "sft_dataset"
    train_file = dataset_dir / "train.jsonl"
    normalization = normalize_sft_dataset(
        source,
        train_file,
        user_field=_string(sft.get("user_field", "instruction"), "sft.user_field"),
        assistant_field=_string(
            sft.get("assistant_field", "output"), "sft.assistant_field"
        ),
    )
    cutoff_len = _positive_int(
        _mapping(sft.get("trainer"), "sft.trainer").get("cutoff_len"),
        "sft.trainer.cutoff_len",
    )
    normalization["token_lengths"] = audit_sft_token_lengths(
        train_file,
        model_path=_string(config.get("base_model"), "base_model"),
        cutoff_len=cutoff_len,
    )
    if (
        bool(sft.get("fail_on_truncation", False))
        and normalization["token_lengths"]["rows_over_cutoff"]
    ):
        raise RuntimeConfigError(
            f"{normalization['token_lengths']['rows_over_cutoff']} rows exceed cutoff_len={cutoff_len}"
        )
    (dataset_dir / "dataset_info.json").write_text(
        json.dumps(
            {
                "downstream_sft": {
                    "file_name": train_file.name,
                    "columns": {
                        "prompt": "instruction",
                        "query": "input",
                        "response": "output",
                    },
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    normalization_path = runtime_dir / "normalization_report.json"
    normalization_path.write_text(
        json.dumps(normalization, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    output_dir = runtime_dir / "sft_model"
    trainer = dict(_mapping(sft.get("trainer"), "sft.trainer"))
    required = {
        "model_name_or_path": _string(config.get("base_model"), "base_model"),
        "trust_remote_code": True,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "full",
        "dataset_dir": str(dataset_dir.resolve()),
        "dataset": "downstream_sft",
        "output_dir": str(output_dir.resolve()),
    }
    forbidden_overrides = sorted(set(required).intersection(trainer))
    if forbidden_overrides:
        raise RuntimeConfigError(
            "sft.trainer must not override runtime-owned keys: "
            + ", ".join(forbidden_overrides)
        )
    trainer_config = {**required, **trainer}
    train_config_path = runtime_dir / "llamafactory_sft.yaml"
    train_config_path.write_text(
        yaml.safe_dump(trainer_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    cli = _string(sft.get("llamafactory_cli"), "sft.llamafactory_cli")
    env = os.environ.copy()
    _prepend_executable_dir(env, cli)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": _string(
                sft.get("cuda_visible_devices", "0,1,2,3,4,5,6,7"),
                "sft.cuda_visible_devices",
            ),
            "FORCE_TORCHRUN": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    env.update(_string_mapping(sft.get("env", {}), "sft.env"))
    run_logged(
        [cli, "train", str(train_config_path.resolve())],
        cwd=runtime_dir,
        env=env,
        stdout_path=runtime_dir / "sft_stdout.log",
        stderr_path=runtime_dir / "sft_stderr.log",
        timeout_sec=_positive_int(sft.get("timeout_sec", 86400), "sft.timeout_sec"),
    )
    if not (output_dir / "config.json").is_file():
        raise RuntimeError(
            f"LlamaFactory completed without a model config: {output_dir}"
        )
    return output_dir, {
        "checkpoint": str(output_dir.resolve()),
        "normalized_dataset": str(train_file.resolve()),
        "normalization_report": str(normalization_path.resolve()),
        "sft_config": str(train_config_path.resolve()),
    }


def evaluate_model(
    model_path: Path,
    expected_benchmarks: Iterable[str],
    bad_cases_per_benchmark: int,
    config: Mapping[str, Any],
    runtime_dir: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    evaluation = _mapping(config.get("evaluation"), "evaluation")
    groups = evaluation.get("groups")
    if not isinstance(groups, list) or not groups:
        raise RuntimeConfigError("evaluation.groups must be a non-empty list")

    benchmark_specs: dict[str, tuple[Mapping[str, Any], str]] = {}
    for group in groups:
        group_map = _mapping(group, "evaluation.groups[]")
        mapping = _mapping(
            group_map.get("benchmarks"), "evaluation.groups[].benchmarks"
        )
        for label, parser_name in mapping.items():
            if label in benchmark_specs:
                raise RuntimeConfigError(
                    f"duplicate benchmark label in evaluation groups: {label}"
                )
            benchmark_specs[str(label)] = (
                group_map,
                _string(parser_name, f"parser name for {label}"),
            )

    expected = list(expected_benchmarks)
    if set(expected) != set(benchmark_specs):
        raise RuntimeConfigError(
            "request/config benchmark mismatch: "
            f"request={sorted(expected)}, config={sorted(benchmark_specs)}"
        )

    evaluation_root = Path(__file__).resolve().parent / "vendor/qwen25_math"
    script = evaluation_root / "math_eval.py"
    if not script.is_file():
        raise RuntimeConfigError(f"vendored Qwen2.5-Math evaluator not found: {script}")
    python = _string(evaluation.get("python"), "evaluation.python")
    global_cuda = _string(
        evaluation.get("cuda_visible_devices", "0"), "evaluation.cuda_visible_devices"
    )
    timeout_sec = _positive_int(
        evaluation.get("timeout_sec", 86400), "evaluation.timeout_sec"
    )

    results: dict[str, Any] = {}
    output_root = runtime_dir / "evaluation"
    for group_index, group in enumerate(groups):
        group_map = _mapping(group, "evaluation.groups[]")
        group_name = _string(
            group_map.get("name", f"group_{group_index}"), "evaluation group name"
        )
        labels_to_parsers = {
            str(label): _string(parser_name, f"parser name for {label}")
            for label, parser_name in _mapping(
                group_map.get("benchmarks"), "evaluation.groups[].benchmarks"
            ).items()
        }
        data_dir = Path(_string(group_map.get("data_dir"), f"{group_name}.data_dir"))
        if not data_dir.is_dir():
            raise RuntimeConfigError(f"diagnostic data_dir does not exist: {data_dir}")
        group_output = output_root / group_name
        group_output.mkdir(parents=True, exist_ok=True)
        parser_names = list(labels_to_parsers.values())
        argv = [
            python,
            str(script),
            "--data_names",
            ",".join(parser_names),
            "--data_dir",
            str(data_dir.resolve()),
            "--model_name_or_path",
            str(model_path.resolve()),
            "--output_dir",
            str(group_output.resolve()),
            "--prompt_type",
            _string(
                group_map.get("prompt_type", "qwen25-math-cot"),
                f"{group_name}.prompt_type",
            ),
            "--split",
            "test",
            "--num_test_sample",
            "-1",
            "--seed",
            str(_nonnegative_int(group_map.get("seed", 0), f"{group_name}.seed")),
            "--temperature",
            str(
                _number(group_map.get("temperature", 0.0), f"{group_name}.temperature")
            ),
            "--n_sampling",
            str(
                _positive_int(
                    group_map.get("n_sampling", 1), f"{group_name}.n_sampling"
                )
            ),
            "--top_p",
            str(_number(group_map.get("top_p", 1.0), f"{group_name}.top_p")),
            "--max_tokens_per_call",
            str(
                _positive_int(
                    group_map.get("max_tokens_per_call", 16384),
                    f"{group_name}.max_tokens_per_call",
                )
            ),
            "--use_vllm",
            "--save_outputs",
            "--overwrite",
        ]
        if bool(group_map.get("apply_chat_template", False)):
            argv.append("--apply_chat_template")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = _string(
            group_map.get("cuda_visible_devices", global_cuda),
            f"{group_name}.cuda_visible_devices",
        )
        env["TOKENIZERS_PARALLELISM"] = "false"
        run_logged(
            argv,
            cwd=evaluation_root,
            env=env,
            stdout_path=group_output / "stdout.log",
            stderr_path=group_output / "stderr.log",
            timeout_sec=timeout_sec,
        )
        metric = _string(group_map.get("metric", "accuracy"), f"{group_name}.metric")
        n_sampling = _positive_int(
            group_map.get("n_sampling", 1), f"{group_name}.n_sampling"
        )
        for label, parser_name in labels_to_parsers.items():
            records_path = _single_output_jsonl(group_output / parser_name)
            records = _read_jsonl(records_path)
            results[label] = summarize_qwen_records(
                records,
                metric=metric,
                n_sampling=n_sampling,
                bad_cases_per_benchmark=bad_cases_per_benchmark,
            )

    return {label: results[label] for label in expected}, {
        "evaluation_output": str(output_root.resolve()),
        "evaluation_config": str(
            Path(_string(config.get("config_path"), "config_path")).resolve()
        ),
        "diagnostic_manifest": str(
            Path(
                _string(
                    evaluation.get("diagnostic_manifest"),
                    "evaluation.diagnostic_manifest",
                )
            ).resolve()
        ),
    }


def summarize_qwen_records(
    records: list[dict[str, Any]],
    *,
    metric: str,
    n_sampling: int,
    bad_cases_per_benchmark: int,
) -> dict[str, Any]:
    if not records:
        raise RuntimeConfigError("Qwen evaluator produced an empty result file")
    correctness: list[bool] = []
    failures: list[dict[str, str]] = []
    for index, record in enumerate(records):
        scores = record.get("score")
        outputs = record.get("code")
        parsed_answers = record.get("pred")
        if not isinstance(scores, list) or len(scores) != n_sampling:
            raise RuntimeConfigError(
                f"Qwen result {index} has {len(scores) if isinstance(scores, list) else 'invalid'} "
                f"scores; expected {n_sampling}"
            )
        if not isinstance(outputs, list) or len(outputs) != n_sampling:
            raise RuntimeConfigError(
                f"Qwen result {index} has invalid generated outputs"
            )
        if not isinstance(parsed_answers, list) or len(parsed_answers) != n_sampling:
            raise RuntimeConfigError(
                f"Qwen result {index} has invalid parsed answers"
            )
        if metric == "accuracy":
            correct = bool(scores[0])
        elif metric == f"pass@{n_sampling}":
            correct = any(bool(score) for score in scores)
        else:
            raise RuntimeConfigError(
                f"unsupported metric {metric!r}; use 'accuracy' or 'pass@{n_sampling}'"
            )
        correctness.append(correct)
        if not correct:
            question = _render_text(record.get("question"), "question")
            reference = _render_text(record.get("gt"), "ground truth")
            response = _render_text(
                outputs[0], "model response", empty_sentinel="[empty model response]"
            )
            parsed_answer = _render_text(
                parsed_answers[0], "parsed answer", empty_sentinel=""
            )
            failures.append(
                {
                    "question": question,
                    "model_response": response,
                    "parsed_answer": parsed_answer,
                    "correct_answer": reference,
                }
            )
    score = sum(correctness) / len(correctness)
    return {
        "score": score,
        "metric": metric,
        "incorrect_count": len(failures),
        "bad_cases": failures[: min(bad_cases_per_benchmark, len(failures))],
    }


def run_logged(
    argv: list[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    stdout_path: Path,
    stderr_path: Path,
    timeout_sec: int,
) -> float:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with (
        stdout_path.open("w", encoding="utf-8") as stdout_handle,
        stderr_path.open("w", encoding="utf-8") as stderr_handle,
    ):
        completed = subprocess.run(
            argv,
            cwd=str(cwd),
            env=dict(env),
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"command exited with code {completed.returncode}; see {stdout_path} and {stderr_path}"
        )
    return elapsed


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RuntimeConfigError(f"non-object JSONL record in {path}")
                records.append(value)
    return records


def _single_output_jsonl(directory: Path) -> Path:
    candidates = sorted(directory.glob("*.jsonl"))
    if len(candidates) != 1:
        raise RuntimeConfigError(
            f"expected one Qwen output JSONL under {directory}, found {len(candidates)}"
        )
    return candidates[0]


def _percentile(ordered: list[int], quantile: float) -> float:
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _prepend_executable_dir(env: dict[str, str], executable: str) -> None:
    """Keep console-script subprocesses in the same Python environment."""
    executable_dir = str(Path(executable).resolve().parent)
    current_path = env.get("PATH", "")
    env["PATH"] = (
        os.pathsep.join((executable_dir, current_path))
        if current_path
        else executable_dir
    )


def _expand_environment(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    return value


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeConfigError(f"{field_name} must be an object")
    return value


def _string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeConfigError(f"{field_name} must be a non-empty string")
    return value.strip()


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RuntimeConfigError(f"{field_name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeConfigError(f"{field_name} must be a non-negative integer")
    return value


def _number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeConfigError(f"{field_name} must be numeric")
    return float(value)


def _string_mapping(value: Any, field_name: str) -> dict[str, str]:
    mapping = _mapping(value, field_name)
    if any(
        not isinstance(key, str) or not isinstance(item, str)
        for key, item in mapping.items()
    ):
        raise RuntimeConfigError(f"{field_name} must map strings to strings")
    return dict(mapping)


def _render_text(value: Any, field_name: str, empty_sentinel: str | None = None) -> str:
    if isinstance(value, str):
        text = value
    elif value is None:
        text = ""
    else:
        text = json.dumps(value, ensure_ascii=False)
    if not text.strip():
        if empty_sentinel is not None:
            return empty_sentinel
        raise RuntimeConfigError(f"Qwen result has empty {field_name}")
    return text
