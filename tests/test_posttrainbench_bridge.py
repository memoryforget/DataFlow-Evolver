"""桥接脚本 + benchmark 映射单测：eval 参数画像、格式对齐转换、results.json 契约。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pytest

import ptb_benchmarks as ptb
import posttrainbench_sft_eval as bridge


# --------------------------- 映射表 ---------------------------
def test_all_seven_benchmarks_present():
    assert set(ptb.list_benchmarks()) == {
        "gsm8k", "aime2025", "gpqamain", "bfcl", "humaneval",
        "healthbench", "arenahardwriting",
    }


def test_build_eval_argv_inspect_family():
    argv = ptb.build_eval_argv("gsm8k", model_path="/m", templates_dir="/t",
                               limit=150, json_output_file="/o.json")
    assert "--max-tokens" in argv and "--max-new-tokens" not in argv
    assert "--limit" in argv and argv[argv.index("--limit") + 1] == "150"
    assert argv[argv.index("--model-path") + 1] == "/m"


def test_build_eval_argv_server_family_and_full_limit():
    argv = ptb.build_eval_argv("healthbench", model_path="/m", templates_dir="/t",
                               limit=-1, json_output_file="/o.json")
    assert "--max-new-tokens" in argv and "--max-tokens" not in argv
    assert "--judge-workers" in argv
    assert "--limit" not in argv                # limit=-1 → 全量，不传 --limit


def test_resolve_benchmark_name():
    assert ptb.resolve_benchmark_name(None, "/x/src/eval/tasks/bfcl") == "bfcl"
    assert ptb.resolve_benchmark_name("GSM8K", None) == "gsm8k"
    with pytest.raises(ValueError):
        ptb.resolve_benchmark_name(None, "/x/unknown/dir")


def test_templates_dir_derivation():
    td = ptb.templates_dir_for("/root/src/eval/tasks/gsm8k")
    assert td.name == "templates"
    assert td.parent.name == "eval"


# --------------------------- format_requirement 三档（借鉴 ANDES） ---------------------------
def test_format_requirement_mapping():
    # 每个 benchmark 都必须显式声明档位，且取值合法
    valid = {ptb.FORMAT_UNSTRUCTURED, ptb.FORMAT_CODE, ptb.FORMAT_TOOL_CALL}
    for name in ptb.list_benchmarks():
        fr = ptb.get_benchmark(name)["format_requirement"]
        assert fr in valid, (name, fr)
    assert ptb.format_requirement_for("humaneval") == ptb.FORMAT_CODE
    assert ptb.format_requirement_for("bfcl") == ptb.FORMAT_TOOL_CALL
    assert ptb.format_requirement_for("gsm8k") == ptb.FORMAT_UNSTRUCTURED


def test_format_instruction_text():
    # unstructured → 空串；code/tool_call → 带对应 fence 提示
    assert ptb.format_instruction_for("gsm8k") == ""
    assert "```python" in ptb.format_instruction_for("humaneval")
    assert "```json" in ptb.format_instruction_for("bfcl")


def test_composed_task_text_appends_only_for_structured():
    # unstructured 不改动原文案
    assert ptb.composed_task_text("gsm8k") == ptb.get_benchmark("gsm8k")["task_text"]
    # code/tool_call 追加【输出格式要求】
    he = ptb.composed_task_text("humaneval")
    assert he.startswith(ptb.get_benchmark("humaneval")["task_text"])
    assert "【输出格式要求】" in he and "```python" in he
    assert "```json" in ptb.composed_task_text("bfcl")


def test_convert_dataset_injects_format_into_system_prompt(tmp_path):
    # humaneval(code)：system prompt 应被追加 ```python fence 结构约束
    src = tmp_path / "syn.jsonl"
    src.write_text(
        json.dumps({"instruction": "Write add(a,b)",
                    "output": "```python\ndef add(a,b):\n    return a+b\n```"}) + "\n",
        encoding="utf-8",
    )
    train_file = bridge.convert_dataset(src, tmp_path / "d", "humaneval", base_model="dummy-qwen")
    rows = json.loads(train_file.read_text(encoding="utf-8"))
    sysval = rows[0]["conversations"][0]["value"]
    assert "```python" in sysval
    # gsm8k(unstructured)：不追加 fence 约束
    src2 = tmp_path / "syn2.jsonl"
    src2.write_text(json.dumps({"instruction": "1+1?", "output": "ANSWER: 2"}) + "\n", encoding="utf-8")
    tf2 = bridge.convert_dataset(src2, tmp_path / "d2", "gsm8k", base_model="dummy-qwen")
    rows2 = json.loads(tf2.read_text(encoding="utf-8"))
    assert "```" not in rows2[0]["conversations"][0]["value"]


# --------------------------- 模型族 / 模板 ---------------------------
@pytest.mark.parametrize("path,fam", [
    ("/models/Qwen3-1.7B", "qwen"),
    ("/models/gemma-3-4b", "gemma"),
    ("/models/SmolLM3-3B", "smollm"),
])
def test_model_family_and_template(path, fam):
    assert bridge.model_family(path) == fam
    assert bridge.llamafactory_template(fam) in {"qwen", "gemma", "smollm", "llama3"}


# --------------------------- 数据转换（格式对齐） ---------------------------
def test_convert_dataset_gsm8k_alignment(tmp_path):
    src = tmp_path / "syn.jsonl"
    src.write_text(
        '{"instruction":"John has 3 apples...","output":"...\\nANSWER: 5"}\n'
        '{"question":"2+2?","answer":"...\\nANSWER: 4"}\n', encoding="utf-8",
    )
    out = tmp_path / "data"
    train_file = bridge.convert_dataset(src, out, "gsm8k", base_model="dummy-qwen")
    rows = json.loads(train_file.read_text(encoding="utf-8"))
    assert len(rows) == 2
    conv = rows[0]["conversations"]
    assert [c["from"] for c in conv] == ["system", "human", "gpt"]
    # system prompt 对齐 GSM8K（含 ANSWER 约定）
    assert "ANSWER" in conv[0]["value"]
    assert conv[1]["value"].startswith("John has")
    # dataset_info 为 sharegpt
    info = json.loads((out / "dataset_info.json").read_text(encoding="utf-8"))
    assert info["train"]["formatting"] == "sharegpt"


def test_convert_dataset_tool_calling_includes_tools(tmp_path):
    src = tmp_path / "syn.jsonl"
    src.write_text(
        json.dumps({
            "instruction": "What's the weather in SF?",
            "output": '{"name":"get_weather","arguments":{"city":"SF"}}',
            "tools": [{"name": "get_weather", "parameters": {"city": "str"}}],
        }) + "\n", encoding="utf-8",
    )
    out = tmp_path / "data"
    train_file = bridge.convert_dataset(src, out, "bfcl", base_model="dummy-qwen")
    rows = json.loads(train_file.read_text(encoding="utf-8"))
    assert "tools" in rows[0]
    info = json.loads((out / "dataset_info.json").read_text(encoding="utf-8"))
    assert info["train"]["columns"].get("tools") == "tools"


def test_convert_dataset_empty_raises(tmp_path):
    src = tmp_path / "syn.jsonl"
    src.write_text('{"foo":"bar"}\n', encoding="utf-8")   # 无 instruction/output
    with pytest.raises(ValueError):
        bridge.convert_dataset(src, tmp_path / "data", "gsm8k", base_model="dummy-qwen")


# --------------------------- 跨 conda env 子进程修正 ---------------------------
def _fake_env(root: Path, *, with_nvcc: bool) -> str:
    """造一个假 conda env 前缀，返回其 python 路径。"""
    (root / "lib").mkdir(parents=True, exist_ok=True)
    (root / "bin").mkdir(parents=True, exist_ok=True)
    py = root / "bin" / "python"
    py.write_text("", encoding="utf-8")
    if with_nvcc:
        (root / "bin" / "nvcc").write_text("", encoding="utf-8")
    return str(py)


def test_env_fixups_prepend_own_lib_and_set_cuda_home(tmp_path):
    py = _fake_env(tmp_path / "df-x", with_nvcc=True)
    env = {"LD_LIBRARY_PATH": "/other/lib"}
    bridge.apply_env_prefix_fixups(env, py)
    # 自己的 lib 必须在最前，才能压过父进程继承来的旧 libstdc++
    assert env["LD_LIBRARY_PATH"] == f"{tmp_path / 'df-x' / 'lib'}:/other/lib"
    assert env["CUDA_HOME"] == str(tmp_path / "df-x")
    assert env["CUDA_PATH"] == str(tmp_path / "df-x")


def test_env_fixups_without_nvcc_leaves_cuda_home_unset(tmp_path):
    py = _fake_env(tmp_path / "df-y", with_nvcc=False)
    env = {}
    bridge.apply_env_prefix_fixups(env, py)
    assert env["LD_LIBRARY_PATH"] == str(tmp_path / "df-y" / "lib")
    assert "CUDA_HOME" not in env


def test_env_fixups_are_idempotent_and_respect_valid_cuda_home(tmp_path):
    py = _fake_env(tmp_path / "df-z", with_nvcc=True)
    other = _fake_env(tmp_path / "df-w", with_nvcc=True)
    env = {"CUDA_HOME": str(Path(other).parent.parent)}
    bridge.apply_env_prefix_fixups(env, py)
    bridge.apply_env_prefix_fixups(env, py)       # 二次调用不应重复追加
    assert env["LD_LIBRARY_PATH"] == str(tmp_path / "df-z" / "lib")
    # 已有可用 CUDA_HOME（含 nvcc）不被覆盖
    assert env["CUDA_HOME"] == str(tmp_path / "df-w")


# --------------------------- results.json 契约 ---------------------------
def test_write_results_contract(tmp_path):
    bridge._write_results(tmp_path, 0.7321, "gsm8k")
    data = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert data["score"] == pytest.approx(0.7321)
    assert data["metric_name"] == "gsm8k"
    assert data["bad_cases"] == []           # 恒为空：不落 per-sample test 内容
