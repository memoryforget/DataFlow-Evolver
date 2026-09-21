import importlib.util
import json
import sys
import types
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ADAPTER_PATH = (
    REPO
    / "downstream_runtime"
    / "education"
    / "scripts"
    / "official_adapters.py"
)
SUITE_PATH = (
    REPO
    / "downstream_runtime"
    / "education"
    / "scripts"
    / "run_benchmark_suite.py"
)


def load_adapter():
    spec = importlib.util.spec_from_file_location(
        "education_official_adapters_test", ADAPTER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_suite():
    spec = importlib.util.spec_from_file_location(
        "education_benchmark_suite_test", SUITE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def render_official_answer(answer_type, standard_answer):
    if answer_type == "single_choice":
        return f"Reasoning mentions A. Final answer: {standard_answer[0]}"
    if answer_type == "multi_question_choice":
        return " ".join(f"【答案】{answer}" for answer in standard_answer)
    if answer_type == "multi_choice":
        return f"【答案】 {standard_answer[0]}"
    if answer_type == "five_out_of_seven":
        return " ".join(standard_answer)
    raise AssertionError(answer_type)


def test_gaokao_adapter_uses_all_four_official_answer_protocols(tmp_path):
    adapter = load_adapter()
    gaokao_root = adapter.ROOT / "benchmarks" / "gaokao"
    prompt_rows = json.loads(
        (gaokao_root / "Bench" / "Obj_Prompt.json").read_text(encoding="utf-8")
    )["examples"]
    objective_types = {row["keyword"]: row["type"] for row in prompt_rows}
    assert set(objective_types.values()) == {
        "single_choice",
        "multi_question_choice",
        "multi_choice",
        "five_out_of_seven",
    }

    predictions = []
    expected_by_keyword = {}
    for keyword, answer_type in objective_types.items():
        source_file = gaokao_root / "Data" / "Objective_Questions" / f"{keyword}.json"
        source = json.loads(source_file.read_text(encoding="utf-8"))
        example = source["example"][0]
        standard_answer = example["answer"]
        if isinstance(standard_answer, str):
            standard_answer = (
                json.loads(standard_answer.replace("'", '"'))
                if standard_answer.startswith("[")
                else [standard_answer]
            )
        expected_by_keyword[keyword] = standard_answer
        predictions.append(
            {
                "prompt": example["question"],
                "prediction": render_official_answer(answer_type, standard_answer),
                "official": {
                    "question_type": "Objective_Questions",
                    "source_file": str(source_file),
                    "index": example["index"],
                },
            }
        )

    result = adapter.score_gaokao(predictions, tmp_path)
    assert result["score_kind"] == "official"
    assert result["metrics"]["objective"]["scoring_rate"] == 1.0
    for keyword, expected in expected_by_keyword.items():
        scored = json.loads(
            (tmp_path / "gaokao_obj_input" / f"{keyword}.json").read_text(
                encoding="utf-8"
            )
        )
        assert scored["example"][0]["model_answer"] == expected


class FakeError(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.calls = 0

    def create(self, **_kwargs):
        self.calls += 1
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return type(
            "Response",
            (),
            {"choices": [type("Choice", (), {"message": type("Message", (), {"content": outcome})()})()]},
        )()


class FakeClient:
    def __init__(self, outcomes):
        self.chat = type("Chat", (), {"completions": FakeCompletions(outcomes)})()


def test_judge_retries_transient_errors(monkeypatch):
    adapter = load_adapter()
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    client = FakeClient([FakeError("overloaded", 503), "【总分】3分"])
    text, error = adapter._judge_with_retry(client, "judge", "prompt")
    assert error is None
    assert text == "【总分】3分"
    assert client.chat.completions.calls == 2


def test_judge_retries_unparseable_replies(monkeypatch):
    adapter = load_adapter()
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    client = FakeClient(["判分理由完整，但缺少最终格式", "【总分】：3分"])
    text, error = adapter._judge_with_retry(
        client,
        "judge",
        "prompt",
        response_validator=(
            lambda reply: adapter._parse_gaokao_total_score(reply) is not None
        ),
        validation_feedback="Return 【总分】N分.",
    )
    assert error is None
    assert adapter._parse_gaokao_total_score(text) == 3.0
    assert client.chat.completions.calls == 2


def test_judge_returns_last_unparseable_reply_after_retry_limit(monkeypatch):
    adapter = load_adapter()
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    client = FakeClient(["still no total"] * adapter._JUDGE_RETRY_MAX)
    text, error = adapter._judge_with_retry(
        client,
        "judge",
        "prompt",
        response_validator=(
            lambda reply: adapter._parse_gaokao_total_score(reply) is not None
        ),
    )
    assert text == "still no total"
    assert isinstance(error, ValueError)
    assert client.chat.completions.calls == adapter._JUDGE_RETRY_MAX


def test_judge_does_not_retry_permanent_errors(monkeypatch):
    adapter = load_adapter()
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    client = FakeClient([FakeError("invalid_api_key", 401)])
    text, error = adapter._judge_with_retry(client, "judge", "prompt")
    assert text is None
    assert isinstance(error, FakeError)
    assert client.chat.completions.calls == 1


def test_partial_official_metrics_are_usable_but_not_complete():
    suite = load_suite()
    metrics = {"score_kind": "partial", "metrics": {"accuracy": 0.5}}
    assert suite._score_is_usable("partial", metrics)
    assert suite._score_is_usable("official", metrics)
    assert not suite._score_is_usable("scorer_error", metrics)
    assert not suite._score_is_usable("partial", None)


def test_mathfish_malformed_only_subset_has_undefined_format_metrics(monkeypatch):
    mathfish_root = REPO / "downstream_runtime" / "education" / "benchmarks" / "mathfish"
    monkeypatch.syspath_prepend(str(mathfish_root))
    from mathfish.evaluators.tagging_evaluator import TaggingEvaluator

    evaluator = TaggingEvaluator.__new__(TaggingEvaluator)
    subset = [
        (
            {
                "only_malformed": {
                    "true": [["A"]],
                    "pred": ["not a parsed option list"],
                    "example_ids": ["example_grade_level"],
                }
            },
            "dimension",
        )
    ]
    metrics = evaluator.calculate_subsetted_stats(subset)["dimension"][
        "only_malformed"
    ]
    assert metrics["acc"] == 0
    assert metrics["option_acc"] is None
    assert metrics["f1"] is None


def test_gaokao_ignores_subjective_rows_without_a_judge(tmp_path):
    adapter = load_adapter()
    gaokao_root = adapter.ROOT / "benchmarks" / "gaokao"
    source_file = next(
        (gaokao_root / "Data" / "Objective_Questions").glob("*.json")
    )
    example = json.loads(source_file.read_text(encoding="utf-8"))["example"][0]
    result = adapter.score_gaokao(
        [{
            "prompt": example["question"],
            "prediction": "A",
            "official": {
                "question_type": "Subjective_Questions",
                "source_file": str(source_file),
                "index": example["index"],
            },
        }],
        tmp_path,
    )
    assert result["score_kind"] == "unavailable"
    assert result.get("objective_only") is not True
