import json

import pytest

from dataflow_evolver.core.source_audit import audit_mode_dataset


def _write(tmp_path, rows):
    path = tmp_path / "data.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _source():
    return {
        "source_id": "seed-data", "kind": "external_dataset",
        "location": "/data/seed.jsonl", "license": "MIT", "split": "train",
    }


def test_external_mode_rejects_synthetic_rows(tmp_path):
    path = _write(tmp_path, [
        {"_record_id": "e1", "_source_type": "external", "_source_id": "seed-data"},
        {"_record_id": "s1", "_source_type": "synthetic", "_source_id": "generator"},
    ])
    result = audit_mode_dataset(path, {"mode": "external", "sources": [_source()]})
    assert result.external_fraction == pytest.approx(0.5)
    assert result.compliance_score < 1.0
    assert any("100%" in issue for issue in result.issues)


def test_synthetic_mode_rejects_external_rows(tmp_path):
    path = _write(tmp_path, [
        {"_record_id": "e1", "_source_type": "external", "_source_id": "seed-data"},
    ])
    result = audit_mode_dataset(path, {"mode": "synthetic", "sources": []})
    assert result.compliance_score == 0.0
    assert result.issues


def test_hybrid_enforces_ratio_and_seed_links(tmp_path):
    recipe = {
        "mode": "hybrid", "sources": [_source()],
        "mixing_policy": {"external_min_fraction": 0.25, "synthetic_min_fraction": 0.5},
    }
    good = _write(tmp_path, [
        {"_record_id": "e1", "_source_type": "external", "_source_id": "seed-data"},
        {"_record_id": "s1", "_source_type": "synthetic", "_source_id": "generator", "_seed_id": "e1"},
        {"_record_id": "s2", "_source_type": "synthetic", "_source_id": "generator", "_seed_id": "e1"},
    ])
    result = audit_mode_dataset(good, recipe)
    assert result.compliance_score == pytest.approx(1.0)
    assert result.seed_link_rate == 1.0

    broken = _write(tmp_path, [
        {"_record_id": "e1", "_source_type": "external", "_source_id": "seed-data"},
        {"_record_id": "s1", "_source_type": "synthetic", "_source_id": "generator", "_seed_id": "missing"},
    ])
    result = audit_mode_dataset(broken, recipe)
    assert result.compliance_score == 0.0
    assert any("追溯率" in issue for issue in result.issues)


def test_duplicate_record_ids_are_rejected(tmp_path):
    path = _write(tmp_path, [
        {"_record_id": "same", "_source_type": "synthetic", "_source_id": "generator"},
        {"_record_id": "same", "_source_type": "synthetic", "_source_id": "generator"},
    ])
    result = audit_mode_dataset(path, {"mode": "synthetic", "sources": []})
    assert result.record_id_valid_rate == pytest.approx(0.5)
    assert any("全局唯一" in issue for issue in result.issues)
