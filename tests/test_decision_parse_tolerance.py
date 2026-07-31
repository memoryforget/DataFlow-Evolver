"""decision.json 键名漂移的宽容解析。

decision.json 由 PipelineAgent（LLM）生成，实测同一次运行里键名就会漂移：
  - node_2 写了 "operators" + "description" + "design_rationale" → 旧解析只认 "ops"，
    整份算子列表与 rationale 被静默丢弃，日志报"0 个算子"，global memory 只剩 field_flow；
  - node_3 写了 "ops" 字符串数组 → purpose 恒为空串。
这里锁住同义键的兼容，避免再次静默丢信息。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataflow_evolver.agents.pipeline_agent import _parse_decision  # noqa: E402


def test_canonical_schema():
    ops, flow, rationale = _parse_decision({
        "ops": [{"name": "GenOp", "purpose": "生成题目", "params": {"n": 3}}],
        "field_flow": "Empty → question",
        "reason": "从零合成",
    })
    assert ops == [{"name": "GenOp", "custom": True, "purpose": "生成题目", "params": {"n": 3}}]
    assert flow == "Empty → question"
    assert rationale == "从零合成"


def test_node2_style_keys_are_not_dropped():
    """实测 node_2 的写法：operators / description / design_rationale。"""
    ops, flow, rationale = _parse_decision({
        "operators": [
            {"name": "GSM8KQuestionGenerator", "description": "从零生成应用题"},
            {"name": "AnswerFormatValidator", "description": "提取并过滤答案"},
        ],
        "field_flow": "Empty → question → question + answer",
        "design_rationale": "先合成题目再校验答案格式",
    })
    assert [o["name"] for o in ops] == ["GSM8KQuestionGenerator", "AnswerFormatValidator"]
    assert ops[0]["purpose"] == "从零生成应用题"        # description → purpose
    assert rationale == "先合成题目再校验答案格式"
    assert flow == "Empty → question → question + answer"


def test_string_list_ops_still_supported():
    ops, _, _ = _parse_decision({"ops": ["OpA", "OpB"]})
    assert [o["name"] for o in ops] == ["OpA", "OpB"]
    assert all(o["purpose"] == "" for o in ops)      # 字符串写法本就无 purpose 可取


def test_mapping_style_ops():
    ops, _, _ = _parse_decision({"ops": {"OpA": {"purpose": "甲"}, "OpB": {"desc": "乙"}}})
    assert [(o["name"], o["purpose"]) for o in ops] == [("OpA", "甲"), ("OpB", "乙")]


def test_alternate_flow_and_name_keys():
    ops, flow, _ = _parse_decision({
        "operator_list": [{"class": "OpX", "role": "清洗"}],
        "flow": "a → b",
    })
    assert ops == [{"name": "OpX", "custom": True, "purpose": "清洗", "params": {}}]
    assert flow == "a → b"


def test_missing_and_malformed_are_safe():
    assert _parse_decision({}) == ([], "", "")
    assert _parse_decision({"ops": None}) == ([], "", "")
    ops, _, _ = _parse_decision({"ops": [123, None, {"name": "OpOK"}]})
    assert [o["name"] for o in ops] == ["OpOK"]      # 非法条目跳过，不抛异常


def test_unknown_op_name_falls_back_to_placeholder():
    ops, _, _ = _parse_decision({"ops": [{"purpose": "没给名字"}]})
    assert ops[0]["name"] == "?"
