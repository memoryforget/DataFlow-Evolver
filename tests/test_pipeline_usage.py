from __future__ import annotations

import json
import textwrap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from dataflow_evolver.core.schemas import PipelineConfig
from dataflow_evolver.engine.execution_wrapper import SubprocessExecutionWrapper
from dataflow_evolver.llm.serving import APILLMServing
from dataflow_evolver.telemetry.pipeline_usage import (
    aggregate_pipeline_usage,
    append_usage_event,
    build_usage_event,
    normalize_usage,
    write_combined_iteration_usage,
    write_usage_payload,
)


CFG = PipelineConfig([{"name": "HttpStep"}], "", "", "")


class _Response:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def json(self):
        return self.payload


def _request(payload, url="https://api.example/v1/chat/completions"):
    return SimpleNamespace(
        body=json.dumps(payload).encode(),
        url=url,
        method="POST",
    )


def test_review_serving_normalizes_env_expanded_numeric_strings(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    serving = APILLMServing(
        api_url="https://api.example/v1/chat/completions",
        model_name="review-model",
        api_key="secret",
        max_workers="10",
        temperature="0.7",
        request_timeout="600",
        max_retries="3",
    )

    assert serving.max_workers == 10
    assert serving.temperature == 0.7
    assert captured["timeout"] == 600.0
    assert captured["max_retries"] == 3


def test_normalize_usage_supports_openai_and_anthropic_fields():
    assert normalize_usage(
        {
            "prompt_tokens": 100,
            "completion_tokens": 30,
            "total_tokens": 130,
            "prompt_tokens_details": {"cached_tokens": 40},
            "completion_tokens_details": {"reasoning_tokens": 12},
        }
    ) == {
        "input_tokens": 100,
        "output_tokens": 30,
        "total_tokens": 130,
        "cached_input_tokens": 40,
        "cache_creation_input_tokens": 0,
        "reasoning_tokens": 12,
    }
    assert normalize_usage(
        {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_read_input_tokens": 7,
            "cache_creation_input_tokens": 3,
        }
    )["total_tokens"] == 15


def test_missing_usage_is_recorded_without_estimation(monkeypatch):
    monkeypatch.setenv("DF_PIPELINE_LLM_EXECUTION_ID", "exec-1")
    monkeypatch.setenv("DF_PIPELINE_LLM_ITERATION", "iteration_001")
    monkeypatch.setenv("DF_PIPELINE_LLM_ATTEMPT", "2")
    event = build_usage_event(
        _request(
            {"model": "m", "messages": [{"role": "user", "content": "x"}]},
            url="https://api.example/v1/chat/completions?api_key=private#fragment",
        ),
        response=_Response({"id": "r", "model": "m"}),
        elapsed_seconds=1.25,
    )
    assert event is not None
    assert event["usage_status"] == "missing"
    assert event["usage"] is None
    assert event["attempt"] == 2
    assert event["url"] == "https://api.example/v1/chat/completions"
    assert "content" not in json.dumps(event)
    assert "private" not in json.dumps(event)


def test_concurrent_jsonl_appends_remain_parseable(tmp_path):
    path = tmp_path / "calls.jsonl"
    with ThreadPoolExecutor(max_workers=16) as executor:
        list(
            executor.map(
                lambda index: append_usage_event(path, {"index": index}),
                range(200),
            )
        )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 200
    assert {row["index"] for row in rows} == set(range(200))


def test_aggregate_counts_successful_retries_and_missing_usage(tmp_path):
    path = tmp_path / "calls.jsonl"
    base = {
        "execution_id": "exec-1",
        "success": True,
        "model": "m",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "cached_input_tokens": 2,
            "cache_creation_input_tokens": 0,
            "reasoning_tokens": 1,
        },
    }
    append_usage_event(path, base)
    append_usage_event(path, {**base, "execution_id": "exec-2"})
    append_usage_event(
        path,
        {"execution_id": "exec-2", "success": False, "model": "m", "usage": None},
    )

    report = aggregate_pipeline_usage(path)
    assert report["request_count"] == 3
    assert report["success_count"] == 2
    assert report["failure_count"] == 1
    assert report["missing_usage_count"] == 1
    assert report["tokens"]["total_tokens"] == 30
    assert report["execution_ids"] == ["exec-1", "exec-2"]


def test_review_serving_capture_records_exact_usage_without_prompt(tmp_path):
    response = SimpleNamespace(
        model="review-model",
        usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))],
    )
    serving = APILLMServing.__new__(APILLMServing)
    serving.model_name = "configured-review-model"
    serving.max_workers = 1
    serving.temperature = 0.0
    serving._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: response)
        )
    )
    events = tmp_path / "review_llm_calls.jsonl"

    with serving.capture_usage(events, phase="iteration_001"):
        assert serving.generate_one("private review prompt") == "{}"

    summary = aggregate_pipeline_usage(events)
    assert summary["request_count"] == 1
    assert summary["tokens"]["input_tokens"] == 100
    assert summary["tokens"]["output_tokens"] == 20
    assert summary["tokens"]["total_tokens"] == 120
    serialized = events.read_text(encoding="utf-8")
    assert json.loads(serialized)["kind"] == "review_llm"
    assert "private review prompt" not in serialized


def test_combined_iteration_usage_includes_review_and_embedding(tmp_path):
    review = {"kind": "review_llm", "tokens": {"total_tokens": 120}}
    embedding = {"kind": "embedding", "tokens": {"total_tokens": 500}}
    write_usage_payload(tmp_path / "review_llm_token_usage.json", review)
    write_usage_payload(tmp_path / "embedding_token_usage.json", embedding)

    combined = write_combined_iteration_usage(tmp_path)

    assert combined["review_llm"]["tokens"]["total_tokens"] == 120
    assert combined["embedding"]["tokens"]["total_tokens"] == 500


def _write_http_pipeline(iteration_dir: Path) -> None:
    iteration_dir.mkdir(parents=True, exist_ok=True)
    code = r'''
import json
import pathlib
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import requests


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        chunks = [
            {
                "id": "response-1",
                "model": "synthetic-model",
                "choices": [{"index": 0, "delta": {"reasoning_content": "derive"}}],
            },
            {
                "id": "response-1",
                "model": "synthetic-model",
                "choices": [{"index": 0, "delta": {"content": "answer"}}],
            },
            {
                "id": "response-1",
                "model": "synthetic-model",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 120,
                    "completion_tokens": 30,
                    "total_tokens": 150,
                    "prompt_tokens_details": {"cached_tokens": 20},
                    "completion_tokens_details": {"reasoning_tokens": 5}
                }
            }
        ]
        events = [f"data: {json.dumps(chunk)}" for chunk in chunks]
        events.append("data: [DONE]")
        body = ("\n\n".join(events) + "\n\n").encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


server = HTTPServer(("127.0.0.1", 0), Handler)
thread = threading.Thread(target=server.handle_request)
thread.start()
response = requests.post(
    f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
    json={
        "model": "synthetic-model",
        "messages": [{"role": "user", "content": "private"}],
        "stream": True,
    },
    timeout=10,
)
response.raise_for_status()
assert response.json()["choices"][0]["message"] == {
    "role": "assistant",
    "reasoning_content": "derive",
    "content": "answer",
}
thread.join(timeout=10)
server.server_close()

output = pathlib.Path(__file__).parent / "result_step1.jsonl"
output.write_text('{"instruction":"q","output":"a"}\n', encoding="utf-8")
'''
    wrapped = (
        "import os\n"
        "class Pipeline:\n"
        "    def compile(self):\n"
        "        return None\n"
        "    def forward(self):\n"
        f"{textwrap.indent(code, '        ')}\n"
        "pipeline = Pipeline()\n"
        "pipeline.compile()\n"
        "if os.environ.get('DF_COMPILE_ONLY'):\n"
        "    raise SystemExit(0)\n"
        "pipeline.forward()\n"
    )
    (iteration_dir / "pipeline.py").write_text(wrapped, encoding="utf-8")


def test_execution_wrapper_collects_real_requests_usage_and_accumulates(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    raw_entry = workspace / "raw.jsonl"
    raw_entry.write_text('{"instruction":"raw"}\n', encoding="utf-8")
    iteration = workspace / "runs/run_a/iteration_001"
    _write_http_pipeline(iteration)
    executor = SubprocessExecutionWrapper(
        str(workspace), timeout_sec=30, raw_entry_path=str(raw_entry)
    )

    first = executor.run(CFG, iteration)
    second = executor.run(CFG, iteration)

    assert first.success, first.traceback
    assert second.success, second.traceback
    events_path = (
        iteration / "sessions/pipeline/pipeline_llm_calls.jsonl"
    )
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert [event["attempt"] for event in events] == [1, 2]
    assert all(event["usage_status"] == "exact" for event in events)
    assert all("private" not in json.dumps(event) for event in events)

    report = json.loads((iteration / "pipeline_llm_token_usage.json").read_text())
    assert report["request_count"] == 2
    assert report["tokens"]["input_tokens"] == 240
    assert report["tokens"]["output_tokens"] == 60
    assert report["tokens"]["total_tokens"] == 300
    combined = json.loads((iteration / "token_usage.json").read_text())
    assert combined["pipeline_llm"]["tokens"]["total_tokens"] == 300
