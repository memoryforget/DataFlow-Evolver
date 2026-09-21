import json
from types import SimpleNamespace

import numpy as np
import openai
import pytest

from dataflow_evolver.agents.review_agent import ReviewAgent
from dataflow_evolver.core.dataset_quality import (
    DatasetQualityEvaluator,
    EmbeddingBatch,
    OpenAIChatEmbeddingBackend,
    ProxyDatasetConfig,
    _sample_local_proxy,
    compute_embedding_diversity,
    compute_rbf_mmd,
)
from dataflow_evolver.core.schemas import DatasetQualityResult, ReviewResult, TaskSpec
from dataflow_evolver.llm.prompts import build_pipeline_prompt
from dataflow_evolver.core.input_corpus import CorpusFile, InputCorpus


class FakeEmbedder:
    model_name = "fake-embedding"

    def __init__(self):
        self.calls = 0

    def embed(self, conversations):
        self.calls += 1
        vectors = []
        for messages in conversations:
            text = "\n".join(message["content"] for message in messages)
            vectors.append(
                [
                    float(len(text)),
                    float(sum(ord(char) for char in text) % 97 + 1),
                    1.0,
                ]
            )
        return EmbeddingBatch(np.asarray(vectors, dtype=np.float32), 0.0)


def _write_jsonl(path, rows):
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_exact_biased_mmd_is_zero_for_identical_embeddings():
    values = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    mmd, terms = compute_rbf_mmd(values, values.copy(), sigma=1.0, biased=True)
    assert mmd == pytest.approx(0.0, abs=1e-12)
    assert terms["candidate_internal"] == pytest.approx(terms["proxy_internal"])
    assert terms["cross"] == pytest.approx(terms["candidate_internal"])


def test_rbf_mmd_matches_closed_form_for_two_singletons():
    candidate = np.asarray([[1.0, 0.0]], dtype=np.float64)
    proxy = np.asarray([[0.0, 1.0]], dtype=np.float64)
    mmd, terms = compute_rbf_mmd(candidate, proxy, sigma=1.0, biased=True)
    expected_cross = np.exp(-1.0)  # squared distance is 2 after L2 normalization.
    assert terms["candidate_internal"] == pytest.approx(1.0)
    assert terms["proxy_internal"] == pytest.approx(1.0)
    assert terms["cross"] == pytest.approx(expected_cross)
    assert mmd == pytest.approx(np.sqrt(2.0 - 2.0 * expected_cross))


def test_mmd_clamps_tiny_negative_squared_value():
    # Identical arrays exercise the floating-point cancellation path; the result must
    # remain finite and non-negative even if the three means differ by roundoff.
    values = np.asarray([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float64)
    mmd, _ = compute_rbf_mmd(values, values.copy())
    assert np.isfinite(mmd)
    assert mmd >= 0.0


def test_embedding_diversity_separates_identical_and_orthogonal_vectors():
    identical = np.asarray([[1.0, 0.0]] * 4, dtype=np.float64)
    orthogonal = np.eye(4, dtype=np.float64)

    collapsed = compute_embedding_diversity(identical)
    diverse = compute_embedding_diversity(orthogonal)

    assert collapsed["cosine_vendi_score"] == pytest.approx(1.0)
    assert collapsed["cosine_vendi_ratio"] == pytest.approx(0.25)
    assert collapsed["nearest_neighbor_cosine_mean"] == pytest.approx(1.0)
    assert diverse["cosine_vendi_score"] == pytest.approx(4.0)
    assert diverse["cosine_vendi_ratio"] == pytest.approx(1.0)
    assert diverse["nearest_neighbor_cosine_mean"] == pytest.approx(0.0)


def test_dataset_evaluator_uses_configured_sample_size_and_cache(tmp_path):
    candidate = tmp_path / "candidate.jsonl"
    proxy = tmp_path / "proxy.jsonl"
    _write_jsonl(
        candidate,
        [
            {"instruction": f"candidate question {i}", "output": f"answer {i}"}
            for i in range(8)
        ],
    )
    _write_jsonl(
        proxy,
        [
            {"question": f"proxy question {i}", "response": f"solution {i}"}
            for i in range(8)
        ],
    )
    embedder = FakeEmbedder()
    evaluator = DatasetQualityEvaluator(
        proxy=ProxyDatasetConfig(
            source="local",
            path=str(proxy),
            user_field="question",
            assistant_field="response",
            name="math-proxy",
        ),
        embedder=embedder,
        sample_size=5,
        seed=42,
        cache_dir=tmp_path / "cache",
    )

    first = evaluator.evaluate(
        str(candidate), candidate_user_field="instruction", candidate_assistant_field="output"
    )
    second = evaluator.evaluate(
        str(candidate), candidate_user_field="instruction", candidate_assistant_field="output"
    )

    assert first.status == "ok"
    assert first.candidate_total == 8
    assert first.candidate_sample_size == 5
    assert first.proxy_sample_size == 5
    assert first.embedding_dim == 3
    assert first.das == pytest.approx(-first.mmd)
    assert first.embedding_diversity["cosine_vendi_score"] >= 1.0
    assert "nearest_neighbor_cosine_p95" in first.embedding_diversity
    assert set(first.kernel_terms) == {
        "candidate_internal", "proxy_internal", "cross", "mmd_squared"
    }
    assert second.mmd == pytest.approx(first.mmd)
    assert embedder.calls == 2  # Candidate + proxy only once; second evaluation is cached.
    assert first.embedding_usage["usage_status"] == "unavailable"
    assert first.embedding_wall_time_seconds >= 0.0
    assert first.embedding_usage["wall_time_seconds"] >= 0.0
    assert second.embedding_usage["usage_status"] == "cache_only"
    assert second.embedding_usage["request_count"] == 0
    assert second.embedding_usage["tokens"]["total_tokens"] == 0
    assert second.embedding_wall_time_seconds >= 0.0


def test_candidate_only_diversity_skips_proxy_and_das(tmp_path):
    candidate = tmp_path / "candidate.jsonl"
    _write_jsonl(
        candidate,
        [
            {"instruction": f"question {i}", "output": f"answer {i}"}
            for i in range(5)
        ],
    )
    embedder = FakeEmbedder()
    evaluator = DatasetQualityEvaluator(
        proxy=None,
        embedder=embedder,
        sample_size=5,
        proxy_comparison_enabled=False,
    )

    result = evaluator.evaluate(
        str(candidate), candidate_user_field="instruction", candidate_assistant_field="output"
    )

    assert result.status == "ok"
    assert result.proxy_name == "disabled (candidate-only diversity)"
    assert result.mmd is None
    assert result.das is None
    assert result.kernel_terms == {}
    assert "cosine_vendi_score" in result.embedding_diversity
    assert "nearest_neighbor_cosine_p95" in result.embedding_diversity
    assert embedder.calls == 1


def test_candidate_only_diversity_accepts_multiturn_messages(tmp_path):
    candidate = tmp_path / "candidate.jsonl"
    _write_jsonl(
        candidate,
        [
            {
                "messages": [
                    {"role": "system", "content": "Tutor carefully."},
                    {"role": "user", "content": f"question {i}"},
                    {"role": "assistant", "content": f"hint {i}"},
                    {"role": "user", "content": f"follow-up {i}"},
                    {"role": "assistant", "content": f"answer {i}"},
                ]
            }
            for i in range(5)
        ],
    )
    embedder = FakeEmbedder()
    evaluator = DatasetQualityEvaluator(
        proxy=None,
        embedder=embedder,
        sample_size=5,
        proxy_comparison_enabled=False,
    )

    result = evaluator.evaluate(
        str(candidate), candidate_user_field="messages", candidate_assistant_field="messages"
    )

    assert result.status == "ok"
    assert result.metric == "candidate_embedding_diversity"
    assert result.proxy_sample_size == 0
    assert result.mmd is None
    assert result.das is None
    assert embedder.calls == 1


def test_openai_embedding_backend_aggregates_exact_api_usage(monkeypatch):
    class FakeClient:
        async def post(self, _path, *, cast_to, body):
            del cast_to
            index = int(body["messages"][0]["content"][-1])
            return SimpleNamespace(
                data=[SimpleNamespace(embedding=[float(index), 1.0])],
                usage={"prompt_tokens": 10 + index, "total_tokens": 10 + index},
            )

        async def close(self):
            return None

    monkeypatch.setattr(openai, "AsyncOpenAI", lambda **_kwargs: FakeClient())
    backend = OpenAIChatEmbeddingBackend(
        model_name="embedding-model",
        base_url="http://127.0.0.1:8000/v1",
        max_concurrent_requests=2,
    )

    batch = backend.embed(
        [
            [{"role": "user", "content": "item0"}],
            [{"role": "user", "content": "item1"}],
        ]
    )

    assert batch.vectors.shape == (2, 2)
    assert batch.usage["usage_status"] == "exact"
    assert batch.usage["request_count"] == 2
    assert batch.usage["success_count"] == 2
    assert batch.usage["tokens"]["input_tokens"] == 21
    assert batch.usage["tokens"]["total_tokens"] == 21


def test_dataset_evaluator_reads_local_parquet_snapshot_directory(tmp_path):
    datasets = pytest.importorskip("datasets")
    candidate = tmp_path / "candidate.jsonl"
    proxy_dir = tmp_path / "ODA-Math-460k" / "data"
    proxy_dir.mkdir(parents=True)
    _write_jsonl(
        candidate,
        [
            {"instruction": f"candidate question {i}", "output": f"answer {i}"}
            for i in range(6)
        ],
    )
    proxy_rows = [
        {"question": f"proxy question {i}", "response": f"solution {i}"}
        for i in range(6)
    ]
    datasets.Dataset.from_list(proxy_rows).to_parquet(proxy_dir / "train-00000.parquet")
    evaluator = DatasetQualityEvaluator(
        proxy=ProxyDatasetConfig(
            source="local",
            path=str(proxy_dir.parent),
            user_field="question",
            assistant_field="response",
        ),
        embedder=FakeEmbedder(),
        sample_size=5,
        seed=42,
    )

    result = evaluator.evaluate(
        str(candidate), candidate_user_field="instruction", candidate_assistant_field="output"
    )

    assert result.status == "ok"
    assert result.proxy_sample_size == 5
    assert result.mmd is not None


def test_local_proxy_reads_educhat_jsonl_directory_as_one_reservoir(tmp_path):
    proxy_dir = tmp_path / "educhat"
    proxy_dir.mkdir()
    _write_jsonl(
        proxy_dir / "part-b.jsonl",
        [
            {"data": ["question b1", "answer b1"]},
            {"data": ["question b2", "answer b2"]},
        ],
    )
    _write_jsonl(
        proxy_dir / "part-a.jsonl",
        [
            {"data": ["question a1", "answer a1"]},
            {
                "data": ["question a2", "answer a2", "question a3", "answer a3"],
                "system_prompt": "You are a tutor.",
            },
        ],
    )

    first = _sample_local_proxy(
        str(proxy_dir),
        sample_size=3,
        seed=42,
        user_field="data",
        assistant_field="data",
    )
    second = _sample_local_proxy(
        str(proxy_dir),
        sample_size=3,
        seed=42,
        user_field="data",
        assistant_field="data",
    )

    messages, total, valid = first
    assert first == second
    assert total == 4
    assert valid == 4
    assert len(messages) == 3
    assert all(dialog[-1]["role"] == "assistant" for dialog in messages)
    assert all(dialog[0]["role"] == "user" for dialog in messages)
    assert all(
        message["content"] != "You are a tutor."
        for dialog in messages
        for message in dialog
    )


def test_local_proxy_rejects_malformed_educhat_turn_lists(tmp_path):
    proxy = tmp_path / "proxy.jsonl"
    _write_jsonl(
        proxy,
        [
            {"data": ["question", "answer"]},
            {"data": ["question", "answer", "orphan question"]},
            {"data": ["question", ""]},
        ],
    )

    messages, total, valid = _sample_local_proxy(
        str(proxy),
        sample_size=10,
        seed=42,
        user_field="data",
        assistant_field="data",
    )

    assert total == 3
    assert valid == 1
    assert messages == [
        [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
    ]


def test_local_proxy_rejects_sharegpt_conversation_ending_with_user(tmp_path):
    proxy = tmp_path / "proxy.jsonl"
    _write_jsonl(
        proxy,
        [
            {
                "messages": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "partial answer"},
                    {"role": "user", "content": "unanswered follow-up"},
                ]
            }
        ],
    )

    messages, total, valid = _sample_local_proxy(
        str(proxy),
        sample_size=10,
        seed=42,
        user_field="messages",
        assistant_field="messages",
    )

    assert total == 1
    assert valid == 0
    assert messages == []


def test_dataset_evaluator_reports_insufficient_samples_without_embedding(tmp_path):
    candidate = tmp_path / "candidate.jsonl"
    proxy = tmp_path / "proxy.jsonl"
    _write_jsonl(candidate, [{"instruction": "q", "output": "a"}])
    _write_jsonl(proxy, [{"question": "q", "response": "a"}] * 5)
    embedder = FakeEmbedder()
    evaluator = DatasetQualityEvaluator(
        proxy=ProxyDatasetConfig(
            source="local",
            path=str(proxy),
            user_field="question",
            assistant_field="response",
        ),
        embedder=embedder,
        sample_size=5,
        require_full_sample=True,
    )

    result = evaluator.evaluate(
        str(candidate), candidate_user_field="instruction", candidate_assistant_field="output"
    )

    assert result.status == "insufficient_samples"
    assert result.candidate_sample_size == 1
    assert "5" in (result.error or "")
    assert embedder.calls == 0


class StubDatasetEvaluator:
    def evaluate(self, *args, **kwargs):
        return DatasetQualityResult(
            enabled=True,
            status="ok",
            proxy_name="proxy",
            candidate_sample_size=5,
            proxy_sample_size=5,
            mmd=0.2,
            das=-0.2,
            embedding_diversity={
                "cosine_vendi_score": 4.0,
                "cosine_vendi_ratio": 0.5,
                "nearest_neighbor_cosine_p95": 0.8,
            },
        )


def test_review_agent_records_parent_mmd_delta():
    class NeverServing:
        pass

    reviewer = ReviewAgent.__new__(ReviewAgent)
    reviewer.dataset_quality_evaluator = StubDatasetEvaluator()
    reviewer.dataset_quality_fields = ("instruction", "output")
    parent = ReviewResult(0.9, 0.9, True)
    parent.dataset_quality = DatasetQualityResult(
        enabled=True,
        status="ok",
        mmd=0.25,
        das=-0.25,
        embedding_diversity={
            "cosine_vendi_score": 3.5,
            "cosine_vendi_ratio": 0.4,
            "nearest_neighbor_cosine_p95": 0.9,
        },
    )
    current = ReviewResult(0.9, 0.9, True)

    reviewer.apply_dataset_quality(current, "unused.jsonl", parent)

    assert current.dataset_quality.delta_mmd_vs_parent == pytest.approx(-0.05)
    assert current.dataset_quality.delta_embedding_diversity_vs_parent == pytest.approx(
        {
            "cosine_vendi_score": 0.5,
            "cosine_vendi_ratio": 0.1,
            "nearest_neighbor_cosine_p95": -0.1,
        }
    )


def test_pipeline_prompt_exposes_das_without_proxy_examples(tmp_path):
    entry = tmp_path / "raw.jsonl"
    entry.write_text('{"raw":"x"}\n', encoding="utf-8")
    corpus_file = CorpusFile(
        path=entry,
        rel="raw.jsonl",
        size_bytes=entry.stat().st_size,
        rows=1,
        fields=("raw",),
        preview_lines=('{"raw":"x"}',),
    )
    corpus = InputCorpus(
        entry_path=str(entry),
        root=tmp_path,
        files=(corpus_file,),
        entry_file=corpus_file,
    )
    review = ReviewResult(0.9, 0.9, True, review_score=0.8)
    review.dataset_quality = DatasetQualityResult(
        enabled=True,
        status="ok",
        proxy_name="ODA-Math-460k",
        candidate_sample_size=5000,
        proxy_sample_size=5000,
        mmd=0.2,
        das=-0.2,
        delta_mmd_vs_parent=-0.03,
        kernel_terms={
            "candidate_internal": 0.4,
            "proxy_internal": 0.3,
            "cross": 0.35,
        },
        embedding_diversity={
            "cosine_vendi_score": 123.0,
            "cosine_vendi_ratio": 0.0246,
            "nearest_neighbor_cosine_mean": 0.81,
            "nearest_neighbor_cosine_p95": 0.93,
        },
        delta_embedding_diversity_vs_parent={
            "cosine_vendi_ratio": 0.004,
            "nearest_neighbor_cosine_p95": -0.02,
        },
    )

    prompt = build_pipeline_prompt(
        TaskSpec("math", {"instruction": "str", "output": "str"}, []),
        corpus,
        str(tmp_path / "iteration_002"),
        parent_review=review,
    )

    assert "ΔMMD -0.030000" in prompt
    assert "normalized Vendi 0.024600 (Δ +0.004000)" in prompt
    assert "NN cosine p95 0.930000 (Δ -0.020000)" in prompt
    assert "MMD 核项" not in prompt
    assert "ODA-Math-460k" not in prompt
    assert "NN cosine mean" not in prompt


def test_embedding_quality_is_calibrated_and_blended_into_review_score():
    class NeverServing:
        pass

    reviewer = ReviewAgent(
        NeverServing(),
        dataset_quality_evaluator=StubDatasetEvaluator(),
        dataset_quality_weight=0.25,
    )
    result = reviewer.parse(
        {
            "correctness_score": 1.0,
            "relevance_score": 1.0,
            "difficulty_score": 1.0,
            "schema_score": 1.0,
        }
    )
    reviewer.apply_dataset_quality(result, "unused.jsonl")

    das_score = 1.0 - 0.2 / np.sqrt(2.0)
    nn_score = (1.0 - 0.8) / 2.0
    expected_embedding = 0.40 * das_score + 0.30 * 0.5 + 0.30 * nn_score
    assert result.dataset_quality.embedding_quality_score == pytest.approx(expected_embedding)
    assert result.review_score == pytest.approx(0.75 + 0.25 * expected_embedding)
    assert result.score_components["embedding_weight"] == pytest.approx(0.25)
