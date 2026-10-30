import json

from dataflow_evolver.core.schemas import PipelineConfig
from dataflow_evolver.engine.execution_observation import write_execution_observation


def test_observation_uses_actual_step_artifacts_and_bounded_samples(tmp_path):
    raw = tmp_path / "raw.jsonl"
    step = tmp_path / "step.jsonl"
    raw.write_text('{"x": 1}\n{"x": 2}\n{"x": 3}\n', encoding="utf-8")
    step.write_text(
        '{"x": 1, "y": 2}\n{"x": 2, "y": 4, "late": true}\n',
        encoding="utf-8",
    )
    (tmp_path / "step_manifest.json").write_text(
        json.dumps({"steps": [{
            "logical_index": 0,
            "operator_name": "map",
            "output_artifact": str(step),
            "output_rows": 99,
            "output_fields": ["declared_only"],
        }]}),
        encoding="utf-8",
    )
    path = write_execution_observation(
        tmp_path,
        config=PipelineConfig([{"name": "map"}], "", "", ""),
        raw_entry_path=str(raw),
        final_dataset_path=str(step),
        status="ok",
        sample_rows_per_step=1,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    observed = payload["observed_steps"][0]
    assert observed["input_rows"] == 3
    assert observed["output_rows"] == 2
    assert observed["manifest_output_rows"] == 99
    assert observed["output_fields"] == ["late", "x", "y"]
    assert observed["manifest_output_fields"] == ["declared_only"]
    assert observed["sample_rows"] == [{"x": 1, "y": 2}]
    assert payload["final_output"]["rows"] == 2
    assert payload["final_output"]["fields"] == ["late", "x", "y"]


def test_observation_reports_declared_operator_mismatch(tmp_path):
    raw = tmp_path / "raw.jsonl"
    step = tmp_path / "step.jsonl"
    raw.write_text('{"x": 1}\n', encoding="utf-8")
    step.write_text('{"x": 1}\n', encoding="utf-8")
    (tmp_path / "step_manifest.json").write_text(
        json.dumps({"steps": [{
            "logical_index": 0,
            "operator_name": "actual",
            "output_artifact": str(step),
        }]}),
        encoding="utf-8",
    )
    path = write_execution_observation(
        tmp_path,
        config=PipelineConfig([{"name": "declared"}, {"name": "extra"}], "", "", ""),
        raw_entry_path=str(raw),
        final_dataset_path=str(step),
        status="ok",
    )
    kinds = {item["kind"] for item in json.loads(path.read_text())["mismatches"]}
    assert kinds == {"operator_order_mismatch", "operator_count_mismatch"}


def test_observation_streams_low_cardinality_nested_distributions(tmp_path):
    raw = tmp_path / "raw.jsonl"
    step = tmp_path / "step.jsonl"
    raw.write_text('{}\n{}\n{}\n', encoding="utf-8")
    rows = [
        {
            "messages": [],
            "extra_info": json.dumps({"category": "subject", "sample_id": "a"}),
            "pipeline_meta": {"category": "subject", "selected": True},
        },
        {
            "messages": [],
            "extra_info": json.dumps({"category": "diagnostic", "sample_id": "b"}),
            "pipeline_meta": {"category": "diagnostic", "selected": True},
        },
        {
            "messages": [],
            "extra_info": json.dumps({"category": "diagnostic", "sample_id": "c"}),
            "pipeline_meta": {"category": "diagnostic", "selected": False},
        },
    ]
    step.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    (tmp_path / "step_manifest.json").write_text(
        json.dumps({"steps": [{
            "logical_index": 0,
            "operator_name": "select",
            "output_artifact": str(step),
        }]}),
        encoding="utf-8",
    )

    path = write_execution_observation(
        tmp_path,
        config=PipelineConfig([{"name": "select"}], "", "", ""),
        raw_entry_path=str(raw),
        final_dataset_path=str(step),
        status="ok",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    distributions = payload["observed_steps"][0]["field_distributions"]
    assert distributions["pipeline_meta.category"]["value_counts"] == {
        "diagnostic": 2,
        "subject": 1,
    }
    assert distributions["extra_info.category"]["value_counts"] == {
        "diagnostic": 2,
        "subject": 1,
    }
    assert distributions["pipeline_meta.selected"]["value_counts"] == {
        "true": 2,
        "false": 1,
    }
    assert payload["final_output"]["field_distributions"] == distributions
