import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataflow_evolve.engine.execution_wrapper import SubprocessExecutionWrapper
from dataflow_evolve.engine.trainer_wrapper import DownstreamTrainerWrapper

ROOT = Path(__file__).resolve().parents[1]
REF_SCRIPT = ROOT / "scripts" / "reference_train_eval.py"


def test_trainer_parses_reference_results(tmp_path):
    ds = tmp_path / "ds.jsonl"
    ds.write_text(
        '{"instruction":"q1","answer":"a1"}\n'
        '{"instruction":"q2","answer":""}\n',
        encoding="utf-8",
    )
    cmd = f'{sys.executable} "{REF_SCRIPT}"'
    wrapper = DownstreamTrainerWrapper(cmd, timeout_sec=60)
    result = wrapper.run(str(ds), str(tmp_path / "out"))
    assert result.failed is False
    assert result.metric_name == "non_empty_ratio"
    assert abs(result.score - 0.5) < 1e-9
    assert len(result.bad_cases) == 1
    assert result.bad_cases[0].error_type == "empty_answer"


def test_trainer_missing_script_degrades(tmp_path):
    ds = tmp_path / "ds.jsonl"
    ds.write_text('{"answer":"x"}\n', encoding="utf-8")
    wrapper = DownstreamTrainerWrapper("nonexistent_cmd_xyz --flag", timeout_sec=10)
    result = wrapper.run(str(ds), str(tmp_path / "out"))
    assert result.failed is True
    assert result.error_msg is not None


def test_execution_missing_pipeline_file(tmp_path):
    (tmp_path / "node_1").mkdir()
    wrapper = SubprocessExecutionWrapper(agent_root=str(tmp_path), timeout_sec=30)
    from dataflow_evolve.core.schemas import PipelineConfig

    cfg = PipelineConfig(operators=[], field_flow="", code="", rationale="")
    result = wrapper.run(cfg, "node_1")
    assert result.success is False
    assert "不存在" in (result.traceback or "")


def test_execution_runs_simple_pipeline(tmp_path):
    node = tmp_path / "node_1"
    node.mkdir()
    # 一个最小"流水线"：写出一个 jsonl 到 node 目录
    (node / "pipeline.py").write_text(
        "import json, pathlib\n"
        "out = pathlib.Path(__file__).parent / 'result.jsonl'\n"
        "out.write_text(json.dumps({'instruction':'a','answer':'b'}) + '\\n', encoding='utf-8')\n",
        encoding="utf-8",
    )
    wrapper = SubprocessExecutionWrapper(agent_root=str(tmp_path), timeout_sec=30)
    from dataflow_evolve.core.schemas import PipelineConfig

    cfg = PipelineConfig(operators=[], field_flow="", code="", rationale="")
    result = wrapper.run(cfg, "node_1")
    assert result.success is True
    assert result.dataset_path is not None
    assert result.dataset_path.endswith("result.jsonl")


def test_execution_captures_crash(tmp_path):
    node = tmp_path / "node_1"
    node.mkdir()
    (node / "pipeline.py").write_text("raise ValueError('boom')\n", encoding="utf-8")
    wrapper = SubprocessExecutionWrapper(agent_root=str(tmp_path), timeout_sec=30)
    from dataflow_evolve.core.schemas import PipelineConfig

    cfg = PipelineConfig(operators=[], field_flow="", code="", rationale="")
    result = wrapper.run(cfg, "node_1")
    assert result.success is False
    assert "ValueError" in (result.traceback or "") or "boom" in (result.traceback or "")


def test_execution_prefers_nonempty_when_last_step_empty(tmp_path):
    node = tmp_path / "node_1"
    (node / "cache").mkdir(parents=True)
    # 模拟 DataFlow 多 step 落盘：step1 有数据，step2（更新）为空
    (node / "pipeline.py").write_text(
        "import pathlib, time\n"
        "c = pathlib.Path(__file__).parent / 'cache'\n"
        "(c / 's_step1.jsonl').write_text('{\\\"q\\\":\\\"a\\\"}\\n', encoding='utf-8')\n"
        "time.sleep(0.05)\n"
        "(c / 's_step2.jsonl').write_text('\\n', encoding='utf-8')\n",
        encoding="utf-8",
    )
    wrapper = SubprocessExecutionWrapper(agent_root=str(tmp_path), timeout_sec=30)
    from dataflow_evolve.core.schemas import PipelineConfig

    cfg = PipelineConfig(operators=[], field_flow="", code="", rationale="")
    result = wrapper.run(cfg, "node_1")
    assert result.success is True
    assert result.dataset_path.endswith("s_step1.jsonl")


def test_execution_picks_max_step_over_mtime(tmp_path):
    """两步都非空时，应按 step 号取最大（最终算子），而非按 mtime（review M2）。

    故意让 step1 比 step2 写得更晚（mtime 更新），证明选择不依赖 mtime。
    """
    node = tmp_path / "node_1"
    (node / "cache").mkdir(parents=True)
    (node / "pipeline.py").write_text(
        "import pathlib, time\n"
        "c = pathlib.Path(__file__).parent / 'cache'\n"
        "(c / 's_step2.jsonl').write_text('{\\\"q\\\":\\\"final\\\"}\\n', encoding='utf-8')\n"
        "time.sleep(0.05)\n"
        "(c / 's_step1.jsonl').write_text('{\\\"q\\\":\\\"mid\\\"}\\n', encoding='utf-8')\n",
        encoding="utf-8",
    )
    wrapper = SubprocessExecutionWrapper(agent_root=str(tmp_path), timeout_sec=30)
    from dataflow_evolve.core.schemas import PipelineConfig

    cfg = PipelineConfig(operators=[], field_flow="", code="", rationale="")
    result = wrapper.run(cfg, "node_1")
    assert result.success is True
    assert result.dataset_path.endswith("s_step2.jsonl")


# ---------------------------------------------------------------------------
# SSHTrainerWrapper：双机训练（本机 rank0 + SSH 拉起 rank1）。stub 训练命令，无需 GPU。
# ---------------------------------------------------------------------------
import subprocess  # noqa: E402
import time  # noqa: E402

from dataflow_evolve.engine.trainer_wrapper import SSHTrainerWrapper  # noqa: E402


def _ssh_wrapper(cmd: str, *, nnodes: int, timeout_sec: int, node1_host: str = "node1") -> SSHTrainerWrapper:
    return SSHTrainerWrapper(
        cmd,
        node1_host=node1_host,
        master_addr="127.0.0.1",
        gpus_per_node=0,
        nnodes=nnodes,
        timeout_sec=timeout_sec,
        python_path=sys.executable,
    )


def test_ssh_trainer_single_node_end_to_end(tmp_path):
    """nnodes=1：跳过 SSH，仅本机 rank0 拉起 reference_train_eval.py → 解析 TrainResult。"""
    ds = tmp_path / "ds.jsonl"
    ds.write_text(
        '{"instruction":"q1","answer":"a1"}\n'
        '{"instruction":"q2","answer":""}\n',
        encoding="utf-8",
    )
    cmd = f'{sys.executable} "{REF_SCRIPT}"'
    wrapper = _ssh_wrapper(cmd, nnodes=1, timeout_sec=60)
    result = wrapper.run(str(ds), str(tmp_path / "out"))
    assert result.failed is False
    assert result.metric_name == "non_empty_ratio"
    assert abs(result.score - 0.5) < 1e-9
    assert len(result.bad_cases) == 1
    assert result.bad_cases[0].error_type == "empty_answer"


def test_ssh_trainer_missing_script_degrades(tmp_path):
    """rank0 命令不可执行 → 降级 failed（不抛出）。"""
    ds = tmp_path / "ds.jsonl"
    ds.write_text('{"answer":"x"}\n', encoding="utf-8")
    wrapper = SSHTrainerWrapper(
        "nonexistent_cmd_xyz --flag", node1_host="node1", master_addr="127.0.0.1",
        gpus_per_node=0, nnodes=1, timeout_sec=10, python_path=None,
    )
    result = wrapper.run(str(ds), str(tmp_path / "out"))
    assert result.failed is True
    assert result.error_msg is not None


def test_ssh_trainer_passes_cluster_env_to_launcher(tmp_path, monkeypatch):
    """nnodes=2：框架只本地拉一次 node0 脚本，并把集群信息注入环境变量。"""
    ds = tmp_path / "ds.jsonl"
    ds.write_text('{"instruction":"q","answer":"a"}\n', encoding="utf-8")
    out = tmp_path / "out"
    captured = {}

    real_popen = subprocess.Popen

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env")
        assert not (isinstance(cmd, list) and cmd and cmd[0] == "ssh")
        return real_popen(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    cmd = f'{sys.executable} "{REF_SCRIPT}"'
    wrapper = _ssh_wrapper(cmd, nnodes=2, timeout_sec=60, node1_host="user@10.0.0.2")
    result = wrapper.run(str(ds), str(out))

    assert result.failed is False
    launcher_cmd = captured.get("cmd")
    env = captured.get("env")
    assert launcher_cmd is not None and env is not None
    assert str(REF_SCRIPT) in " ".join(str(x) for x in launcher_cmd)
    assert env["NODE_RANK"] == "0"
    assert env["MASTER_ADDR"] == "127.0.0.1"
    assert env["NNODES"] == "2"
    assert env["NODE1_HOST"] == "user@10.0.0.2"
    assert env["SFT_REMOTE_PYTHON"] == sys.executable


def test_ssh_trainer_times_out(tmp_path):
    """rank0 训练超时 → 短超时后降级 failed（不抛出）。"""
    ds = tmp_path / "ds.jsonl"
    ds.write_text('{"instruction":"q","answer":"a"}\n', encoding="utf-8")
    # 用一个会睡很久的命令模拟长训练
    slow = tmp_path / "slow.py"
    slow.write_text("import time; time.sleep(30)\n", encoding="utf-8")
    cmd = f'{sys.executable} "{slow}"'
    wrapper = _ssh_wrapper(cmd, nnodes=1, timeout_sec=2)
    t0 = time.time()
    result = wrapper.run(str(ds), str(tmp_path / "out"))
    elapsed = time.time() - t0
    assert result.failed is True
    assert result.error_msg is not None
    assert elapsed < 20  # 超时后应及时返回，而非等满 30s
