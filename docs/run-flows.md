# PTB Official Default Run Flow

`configs/default.yaml` now uses the PostTrainBench official evaluation path by
default for the current PTB testing profile. The framework itself can still be
adapted to other downstream trainer/evaluator backends through the trainer
wrapper contract.

## 1. Runtime Environments

The framework still uses three isolated environments:

- **df-agent**: runs `python -m dataflow_evolver.main`
- **df-train**: runs LlamaFactory SFT through `scripts/posttrainbench_sft_eval.py`
- **df-eval**: runs PTB official `evaluate.py` and vLLM/inspect_ai

`trainer.train_python` and `trainer.eval_python` in `configs/default.yaml` point
to the train and eval environments. The bridge forwards the eval interpreter as
`SFT_EVAL_PYTHON`.

Install df-agent with both the orchestration and generated-pipeline runtime
extras. If `agent_sdk.external_data: true`, also install `external-data`:

```bash
conda activate df-agent
uv pip install -e ".[agent]"
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e ".[agent-runtime]"
uv pip install -e ".[dev]"
uv pip install -e ".[external-data]"
```

`external-data` provides HuggingFace/ModelScope dataset search and data-discovery
tools in the df-agent process.

## 2. Current Default PTB Loop

Every node follows the same feedback path:

```text
PipelineAgent writes pipeline.py
  -> SubprocessExecutionWrapper generates synthetic SFT data
  -> ReviewAgent checks schema, relevance, quality, hard metrics, decontamination
  -> CappedTrainer fixes sample count
  -> posttrainbench_sft_eval.py runs SFT
  -> PTB evaluate.py --limit ${PTB_DEV_LIMIT:-150} returns official dev score
  -> TrainScore feeds MCTS backprop
```

Final evaluation uses the same bridge in `--mode test`, which calls official
`evaluate.py --limit -1`.

## 3. No Input Corpus By Default

PTB does not provide training input data to the agent. Therefore the default
config leaves `project.input_dir` empty, so `input_corpus.discover()` falls back
to the internal empty entry and the pipeline synthesizes data from scratch.

For explicit ablations only, set:

```bash
export PTB_INPUT_DIR=/path/to/input
export PTB_INPUT_ENTRY=some_file.jsonl
```

Do not use benchmark test questions or answers as training input.

## 4. Hugging Face Access

Official inspect_ai tasks may load benchmark data from Hugging Face. The bridge
sets `HF_ENDPOINT=https://hf-mirror.com` when `HF_ENDPOINT` is not already set.
This follows HF-Mirror's non-invasive environment-variable route.

If data is already cached locally, the mirror is harmless. If the mirror is not
reachable, use a working cache, direct HF access, or a proxy.

## 5. Common Environment

```bash
export DF_API_KEY=...
export ANTHROPIC_API_KEY=...
export PTB_BASE_MODEL=/vepfs-mlp2/c20250602/500050/models/Qwen3-1.7B-Base
export PTB_TASK_DIR=/vepfs-mlp2/c20250602/500050/lh/xqf/PostTrainBench/src/eval/tasks/gsm8k
export PTB_BENCHMARK=gsm8k
export PTB_DEV_LIMIT=150
```

Then run:

```bash
/vepfs-mlp2/c20250602/500050/lh/xqf/miniconda3/envs/df-agent/bin/python \
  -m dataflow_evolver.main \
  --config configs/default.yaml \
  --task "合成 GSM8K 风格的小学数学应用题 SFT 数据（含 CoT、末行 ANSWER: <数值>）以 post-train 该模型"
```

## 6. Dev Limit

`PTB_DEV_LIMIT=150` is the default because GSM8K official `evaluate.py` also
defaults to 150 for fast iteration. Use smaller values only for smoke tests:

```bash
export PTB_DEV_LIMIT=50
```

Final test remains full official evaluation with `--limit -1`.

## 7. Outputs

Each node writes training/eval artifacts under its `train_out/` directory:

- `sft_model/`
- `metrics.json` from official PTB evaluation
- `results.json` in the framework contract:

```json
{
  "score": 0.1234,
  "metric_name": "gsm8k",
  "bad_cases": [],
  "metrics": {
    "accuracy": 0.1234,
    "stderr": 0.0091
  }
}
```

`bad_cases` stays empty because the bridge intentionally avoids reading
per-sample benchmark logs back into the optimization loop.
