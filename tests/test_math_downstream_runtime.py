from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import yaml

SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts/downstream"
sys.path.insert(0, str(SCRIPT_ROOT))

import math_runtime
from math_runtime import (
    RuntimeConfigError,
    _merge_evaluation_shards,
    _prepend_executable_dir,
    _validate_evaluation_parallelism,
    evaluate_model,
    load_request,
    normalize_sft_dataset,
    summarize_qwen_records,
    synchronize_model_eos_contract,
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


def test_synchronize_model_eos_contract_uses_saved_tokenizer_eos(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "eos_token": "<|im_end|>",
                "added_tokens_decoder": {
                    "151643": {"content": "<|endoftext|>"},
                    "151645": {"content": "<|im_end|>"},
                },
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "config.json").write_text(
        json.dumps({"eos_token_id": 151643}), encoding="utf-8"
    )
    (model_dir / "generation_config.json").write_text(
        json.dumps({"eos_token_id": 151643}), encoding="utf-8"
    )

    report = synchronize_model_eos_contract(model_dir)

    assert report["eos_token"] == "<|im_end|>"
    assert report["eos_token_id"] == 151645
    assert json.loads((model_dir / "config.json").read_text())["eos_token_id"] == 151645
    assert (
        json.loads((model_dir / "generation_config.json").read_text())["eos_token_id"]
        == 151645
    )


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
            "response_token_count": None,
            "finish_reason": "",
            "length_truncated": False,
            "sample_index": 0,
        }
    ]


def test_length_limited_generation_is_marked_without_clipping_response() -> None:
    long_response = "reasoning " * 1000
    result = summarize_qwen_records(
        [
            {
                "question": "q",
                "gt": "7",
                "code": [long_response],
                "pred": ["6"],
                "score": [False],
                "finish_reason": ["length"],
                "output_token_count": [16383],
            }
        ],
        metric="accuracy",
        n_sampling=1,
        bad_cases_per_benchmark=3,
        max_tokens_per_call=16384,
    )

    bad_case = result["bad_cases"][0]
    assert bad_case["model_response"] == long_response
    assert bad_case["response_token_count"] == 16383
    assert bad_case["finish_reason"] == "length"
    assert bad_case["length_truncated"] is True


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


def test_sampled_competitions_share_one_seeded_avg_at_4_group() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs/downstream_math_runtime.example.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["evaluation"]["cuda_visible_devices"] == "0,1,2,3"
    assert config["evaluation"]["data_parallel_size"] == 4
    assert config["evaluation"]["tensor_parallel_size"] == 1
    assert config["evaluation"]["pipeline_parallel_size"] == 1
    groups = config["evaluation"]["groups"]
    sampled = next(group for group in groups if group["name"] == "sampled_competition")

    assert sampled["benchmarks"] == {
        "amc23": "amc23",
        "aime24": "aime24",
        "aime25": "aime25",
    }
    assert sampled["metric"] == "avg@4"
    assert sampled["n_sampling"] == 4
    assert sampled["seed"] == 0
    assert sampled["max_model_len"] == 32768


def test_vendored_evaluator_forwards_seed_to_vllm() -> None:
    evaluator = SCRIPT_ROOT / "vendor/qwen25_math/math_eval.py"
    source = evaluator.read_text(encoding="utf-8")

    assert "seed=args.seed" in source
    assert "max_model_len=args.max_model_len" in source
    assert "--data_parallel_rank" in source
    assert "examples[shard_start:shard_end]" in source
    assert "data_parallel_size=args.data_parallel_size" not in source
    assert '"num_samples": 0' in source
    assert "tensor_parallel_size=args.tensor_parallel_size" in source
    assert '"finish_reason"' in source
    assert '"output_token_count"' in source
    assert "resolve_stop_token_ids(tokenizer)" in source
    assert '"qwen2" in args.model_name_or_path.lower()' not in source
    assert "qwen25-math-cot already contains complete Qwen ChatML markers" in source


def test_qwen25_math_cot_rejects_double_chat_templating(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    config_path = tmp_path / "runtime.yaml"
    config_path.touch()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.touch()
    config = {
        "config_path": str(config_path),
        "evaluation": {
            "python": "python",
            "diagnostic_manifest": str(manifest_path),
            "cuda_visible_devices": "0",
            "data_parallel_size": 1,
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "groups": [
                {
                    "name": "core",
                    "data_dir": str(data_dir),
                    "benchmarks": {"gsm8k": "gsm8k"},
                    "metric": "accuracy",
                    "n_sampling": 1,
                    "prompt_type": "qwen25-math-cot",
                    "apply_chat_template": True,
                }
            ],
        },
    }

    with pytest.raises(RuntimeConfigError, match="complete Qwen ChatML"):
        evaluate_model(
            tmp_path / "model",
            ["gsm8k"],
            2,
            config,
            tmp_path / "runtime",
        )


def test_evaluation_parallelism_requires_exact_gpu_topology() -> None:
    devices = _validate_evaluation_parallelism(
        "0,1,2,3",
        data_parallel_size=4,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        field_name="evaluation.cuda_visible_devices",
    )
    assert devices == ["0", "1", "2", "3"]

    with pytest.raises(RuntimeConfigError, match="requires.*4"):
        _validate_evaluation_parallelism(
            "0,1,2",
            data_parallel_size=4,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            field_name="evaluation.cuda_visible_devices",
        )


def test_evaluation_shards_merge_in_rank_order(tmp_path: Path) -> None:
    shards = []
    for rank, records in enumerate(
        ([{"idx": 0}, {"idx": 1}], [{"idx": 2}], [{"idx": 3}, {"idx": 4}])
    ):
        shard = tmp_path / f"shard_{rank:03d}"
        benchmark = shard / "gsm8k"
        benchmark.mkdir(parents=True)
        (benchmark / "part.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        shards.append(shard)

    destination = tmp_path / "merged/gsm8k.jsonl"
    _merge_evaluation_shards(
        shards, parser_name="gsm8k", destination=destination
    )

    assert [json.loads(line)["idx"] for line in destination.read_text().splitlines()] == [
        0,
        1,
        2,
        3,
        4,
    ]


def test_evaluate_model_runs_four_single_gpu_workers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    config_path = tmp_path / "runtime.yaml"
    config_path.touch()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.touch()
    calls: list[tuple[int, str, list[str]]] = []

    def fake_run_logged(argv: list[str], **kwargs: object) -> float:
        rank = int(argv[argv.index("--data_parallel_rank") + 1])
        output_dir = Path(argv[argv.index("--output_dir") + 1]) / "gsm8k"
        output_dir.mkdir(parents=True)
        (output_dir / "part.jsonl").write_text(
            json.dumps(
                {
                    "idx": rank,
                    "question": f"q{rank}",
                    "gt": str(rank),
                    "code": [str(rank)],
                    "pred": [str(rank)],
                    "score": [True],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        env = kwargs["env"]
        assert isinstance(env, dict)
        calls.append((rank, env["CUDA_VISIBLE_DEVICES"], argv))
        return 0.0

    monkeypatch.setattr(math_runtime, "run_logged", fake_run_logged)
    config = {
        "config_path": str(config_path),
        "evaluation": {
            "python": "python",
            "diagnostic_manifest": str(manifest_path),
            "cuda_visible_devices": "0,1,2,3",
            "data_parallel_size": 4,
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "groups": [
                {
                    "name": "core",
                    "data_dir": str(data_dir),
                    "benchmarks": {"gsm8k": "gsm8k"},
                    "metric": "accuracy",
                    "n_sampling": 1,
                }
            ],
        },
    }

    results, _ = evaluate_model(
        tmp_path / "model",
        ["gsm8k"],
        3,
        config,
        tmp_path / "runtime",
    )

    assert results["gsm8k"]["score"] == 1.0
    ordered_calls = sorted(calls)
    assert [(rank, gpu) for rank, gpu, _ in ordered_calls] == [
        (0, "0"),
        (1, "1"),
        (2, "2"),
        (3, "3"),
    ]
    assert all(
        argv[argv.index("--data_parallel_size") + 1] == "4"
        for _, _, argv in ordered_calls
    )
