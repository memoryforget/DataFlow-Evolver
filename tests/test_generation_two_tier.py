"""两档生成规模单测：pipeline prompt 指示从 env 读条数；executor 支持 per-call env_overrides。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataflow_evolver.core.input_corpus import InputCorpus
from dataflow_evolver.core.schemas import PipelineConfig, TaskProfile
from dataflow_evolver.engine.execution_wrapper import SubprocessExecutionWrapper
from dataflow_evolver.llm.prompts import build_pipeline_prompt

SCRATCH = InputCorpus(entry_path="empty.jsonl")   # 无输入语料 = 从零合成


def test_pipeline_prompt_instructs_env_target_samples():
    profile = TaskProfile("QA", {"instruction": "str", "output": "str"}, [], "sft")
    sized = build_pipeline_prompt(profile, SCRATCH, "node_1", target_samples=200)
    assert "DF_TARGET_SAMPLES" in sized                 # 指示从环境变量读取
    assert "os.environ" in sized
    assert "200" in sized                                # 缺省回退值
    # 默认不注入
    base = build_pipeline_prompt(profile, SCRATCH, "node_1")
    assert "DF_TARGET_SAMPLES" not in base


def _write_pipeline(node_path: Path):
    """写一个极简 pipeline.py：从 DF_TARGET_SAMPLES 读条数并产出对应行数的 jsonl。"""
    node_path.mkdir(parents=True, exist_ok=True)
    (node_path / "pipeline.py").write_text(
        "import os, json\n"
        "n = int(os.environ.get('DF_TARGET_SAMPLES', '3'))\n"
        "with open(os.path.join(os.path.dirname(__file__), 'out_step1.jsonl'), 'w') as f:\n"
        "    for i in range(n):\n"
        "        f.write(json.dumps({'i': i}) + '\\n')\n",
        encoding="utf-8",
    )


def _count_rows(path: str) -> int:
    return sum(1 for line in open(path, encoding="utf-8") if line.strip())


def test_executor_env_overrides_controls_output_size(tmp_path):
    agent_root = tmp_path / "agent_root"
    node_dir = "node_1"
    _write_pipeline(agent_root / node_dir)
    executor = SubprocessExecutionWrapper(agent_root=str(agent_root))
    cfg = PipelineConfig(operators=[], field_flow="", code="", rationale="")

    # 探针档：注入 5
    r_probe = executor.run(cfg, node_dir, env_overrides={"DF_TARGET_SAMPLES": "5"})
    assert r_probe.success
    assert _count_rows(r_probe.dataset_path) == 5

    # 放大档：同一 pipeline.py 注入 12（无需重新生成代码）
    r_full = executor.run(cfg, node_dir, env_overrides={"DF_TARGET_SAMPLES": "12"})
    assert r_full.success
    assert _count_rows(r_full.dataset_path) == 12


def test_executor_default_when_no_override(tmp_path):
    agent_root = tmp_path / "agent_root"
    node_dir = "node_2"
    _write_pipeline(agent_root / node_dir)
    executor = SubprocessExecutionWrapper(agent_root=str(agent_root))
    cfg = PipelineConfig(operators=[], field_flow="", code="", rationale="")
    r = executor.run(cfg, node_dir)          # 不注入 → pipeline 回退默认 3
    assert r.success
    assert _count_rows(r.dataset_path) == 3
