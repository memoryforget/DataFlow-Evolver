from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import yaml

SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts/downstream"
sys.path.insert(0, str(SCRIPT_ROOT))

from math_runtime import (
    RuntimeConfigError,
    _prepend_executable_dir,
    load_request,
    normalize_sft_dataset,
    summarize_qwen_records,
)


def test_console_script_environment_precedes_parent_path(tmp_path: Path) -> None:
    executable = tmp_path / "env/bin/llamafactory-cli"
    executable.parent.mkdir(parents=True)
    executable.touch()
    original_path = os.pathsep.join(("first", "second"))
    env = {"PATH": original_path}

    _prepend_executable_dir(env, str(executable))

    assert env["PATH"] == os.pathsep.join(
        (str(executable.parent.resolve()), original_path)
    )


def test_normalize_sft_dataset_drops_blank_and_extra_fields(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "normalized.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps({"instruction": "q1", "output": "a1", "source": "x"}),
                json.dumps({"instruction": "q2", "output": "", "source": "y"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = normalize_sft_dataset(
        source, destination, user_field="instruction", assistant_field="output"
    )

    assert report["rows"] == 1
    assert report["skipped_blank_rows"] == 1
    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "instruction": "q1",
        "input": "",
        "output": "a1",
    }


def test_pass_at_k_and_bad_cases_are_computed_per_question() -> None:
    records = [
        {
            "question": "q1",
            "gt": "1",
            "code": ["bad", "good"],
            "pred": ["0", "1"],
            "score": [False, True],
        },
        {
            "question": "q2",
            "gt": "2",
            "code": ["bad-a", "bad-b"],
            "pred": ["wrong-a", "wrong-b"],
            "score": [False, False],
        },
    ]

    result = summarize_qwen_records(
        records, metric="pass@2", n_sampling=2, bad_cases_per_benchmark=3
    )

    assert result["score"] == 0.5
    assert result["incorrect_count"] == 1
    assert result["bad_cases"] == [
        {
            "question": "q2",
            "model_response": "bad-a",
            "parsed_answer": "wrong-a",
            "correct_answer": "2",
        }
    ]


def test_empty_failed_generation_is_explicit_not_silently_dropped() -> None:
    result = summarize_qwen_records(
        [
            {
                "question": "q",
                "gt": "7",
                "code": [""],
                "pred": [""],
                "score": [False],
            }
        ],
        metric="accuracy",
        n_sampling=1,
        bad_cases_per_benchmark=3,
    )
    assert result["bad_cases"][0]["model_response"] == "[empty model response]"
    assert result["bad_cases"][0]["parsed_answer"] == ""


def test_periodic_request_requires_dataset(tmp_path: Path) -> None:
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "protocol_version": 3,
                "stage": "periodic",
                "training_performed": True,
                "expected_benchmarks": ["gsm8k"],
                "bad_cases_per_benchmark": 3,
                "dataset_path": None,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeConfigError, match="dataset_path"):
        load_request(request)


def test_sampled_competitions_share_one_seeded_pass_at_32_group() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs/downstream_math_runtime.example.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    groups = config["evaluation"]["groups"]
    sampled = next(group for group in groups if group["name"] == "sampled_competition")

    assert sampled["benchmarks"] == {
        "amc23": "amc23",
        "aime24": "aime24",
        "aime25": "aime25",
    }
    assert sampled["metric"] == "pass@32"
    assert sampled["n_sampling"] == 32
    assert sampled["seed"] == 0


def test_vendored_evaluator_forwards_seed_to_vllm() -> None:
    evaluator = SCRIPT_ROOT / "vendor/qwen25_math/math_eval.py"
    source = evaluator.read_text(encoding="utf-8")

    assert "seed=args.seed" in source
