# math_mix_v3_periodic_seed42_formal_v1 实验手册

> 本文件仅保存在当前实验机器的 Git 忽略目录，不 commit、不 push。不得记录 API key 或 token。

## 实验身份

- 仓库：`/jizhicfs/xiaqifeng/DataFlow-Evolver`
- 分支：`refactor/open-dataflow-pipeline-runtime`
- 启动时代码提交：`8e7f12a`
- Run：`math_mix_v3_periodic_seed42_formal_v1`
- 启动时间：2026-08-20T18:11:10+08:00（Asia/Shanghai）
- 最大迭代：5；下游评测间隔：每 2 个 iteration
- 固定输入：`data/math_reasoning_mix_v3/gsm8k_math_dapo3333_bigmath3333_numina3333_train_5k_seed42_decontam.jsonl`
- 输入 5,000 行，目标严格 1,000 条 SFT 记录

## GPU 分配

| 用途 | GPU | 配置 |
|---|---:|---|
| 全参数 SFT | 0-6 | `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6` |
| 下游评测 | 0-3 | 4 个独立 DP rank，每卡一个完整 7B 模型 |
| Embedding | 7 | Qwen3-Embedding-8B 持久服务 |
| 非 SFT 阶段空闲 | 4-6 | 不参与 baseline/评测 |

评测 rank 必须使用独立的 `/tmp/dataflow-evolver-vllm-evaluation/runtime_<pid>/group_<group>/rank_<rank>/{vllm,triton,torchinductor}`。禁止多个 rank 共用 FUSE 编译缓存。

## API 配置

### Pipeline Agent（Codex）
\n- Pipeline 执行失败后默认最多进行 2 次代码自修复。配置字段为 loop.execution_repairs=2，可通过环境变量 DF_EXECUTION_REPAIRS 覆盖。

- URL：`https://relay01.01tree.ai/codex/v1`
- Model：`gpt-5.4`；provider：`tree-relay`；wire API：`responses`
- Key env：`CODEX_API_KEY`；同步 env：`DF_AGENT_API_KEY`
- `DF_AGENT_API_KEY_ENV=CODEX_API_KEY`
- `CODEX_API_KEY_ENV=CODEX_API_KEY`
- Effort：`xhigh`；max turns：50
- Run timeout：3,600 秒（1 小时）
- Codex 总尝试：4（日志 1/4 至 4/4）
- Retry base delay：10 秒；`resume_on_retry=true`
- 框架代码自修复：2 次

### Review Agent

- URL：`https://relay01.01tree.ai/codex/v1`
- Model：`gpt-5.4`；API：OpenAI-compatible `chat.completions`
- Key env：`DF_API_KEY`；URL env：`DF_API_URL`
- Max workers：10；request timeout：600 秒；max retries：3
- Relay 可能短暂返回 `503 proxy_unavailable`。本次预检第一次 503，第一次间隔重试成功。

### Pipeline LLM（teacher，不是 Pipeline Agent）

- URL：`http://28.59.7.41:8000/v1/chat/completions`
- URL env：`DF_PIPELINE_API_URL`；model：`Qwen3.8-27B`
- 实际后端模型：Qwen3.8-27B。服务同时接受 `teacher` 和 `Qwen3.8-27B`，正式实验使用实际模型名 `Qwen3.8-27B`。
- Teacher aliases are both callable. Raw responses contain message.reasoning and message.content; the formal task keeps only complete clean message.content and requires finish_reason=stop.
- Key env：`DF_PIPELINE_API_KEY`，本地默认 `EMPTY`
- Max workers：100
- Temperature：0.6；top_p：0.95；max_tokens：16,384
- Connect timeout：10 秒；read timeout：1,800 秒；max retries：3
- `enable_thinking=true`；`reasoning_output_format=think`
- 要求 finish reason=`stop`，reasoning 和 content 均非空

### Embedding API

- URL：`http://127.0.0.1:8000/v1`；env：`DF_DAS_EMBEDDING_URL`
- Model：`model/Qwen3-Embedding-8B`；GPU 7；dtype=bfloat16
- Max model length：40,960；GPU memory utilization：0.90
- Max concurrent requests：64；max retries：3
- Truncate prompt tokens：40,960；truncation side：right
- DAS sample size：1,000；sample seed：42；max fail rate：0.02
- DataFlow-Evolver 只创建 HTTP client，不负责启动或停止服务

## Embedding 启动与验证

```bash
mkdir -p /tmp/tmux-dfe
chmod 700 /tmp/tmux-dfe
tmux -S /tmp/tmux-dfe/socket new-session -d -s dfe-embedding \
  "export CUDA_VISIBLE_DEVICES=7; exec bash /jizhicfs/xiaqifeng/DataFlow-Evolver/downstream_runtime/start_embedding_server.sh > /tmp/dfe-embedding.log 2>&1"
```

本机缓存根为 `/tmp/dataflow-evolver-embedding`，含 `vllm`、`triton`、`torchinductor`、`cuda`、`xdg`。

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"/jizhicfs/xiaqifeng/DataFlow-Evolver/model/Qwen3-Embedding-8B","input":"readiness check"}'
```

本次首次加载约 5 分钟，真实请求返回 HTTP 200 和非空向量。服务应贯穿全部 5 轮。

## Bad case

三类优先级：`wrong_answer > parse_failure > runaway`。每 benchmark 最多 2 个。各类别内部固定 seed 42 随机打乱；按优先级从每个非空类别先取 1 个，不足再按相同优先级补足。三类完整 count 始终传给 Pipeline Agent。

## 下游评测

- Benchmarks：gsm8k、math、amc23、aime24、aime25、minerva_math、gaokao2024_mix、olympiadbench
- Greedy core：GPU 0-3，4 ranks，accuracy，temperature=0，n_sampling=1
- Sampled competition：GPU 0-3，4 ranks，amc23/aime24/aime25，avg@4，temperature=0.6，n_sampling=4；反馈同时记录 correct_count/total_count、question_count 和 generation-level bad-case counts
- 两组 `max_tokens_per_call=16384`、`max_model_len=32768`
- 评测 timeout：172,800 秒

## SFT

- Base：`/jizhicfs/xiaqifeng/models/Qwen2.5-7B`
- LLaMA-Factory：`/jizhicfs/xiaqifeng/miniconda3/envs/dfe-sft/bin/llamafactory-cli`
- GPU 0-6；DeepSpeed ZeRO-2；template=qwen2_5_base
- Qwen2.5-7B Base 保留 ChatML system/user 边界；assistant 只用原生 <|endoftext|>（151643）结束，禁止把 EOS 替换为 <|im_end|>。
- cutoff_len=16,384；epochs=1
- per-device batch=1；gradient accumulation=4
- learning rate=5e-6；cosine；warmup=0.1
- BF16/TF32；seed/data_seed=42；timeout=172,800 秒

## 启动前预检

必须确认以下变量存在但不得打印值：`CODEX_API_KEY`、`DF_AGENT_API_KEY`、`DF_API_KEY`、`DF_AGENT_API_KEY_ENV`、`CODEX_API_KEY_ENV`、`CODEX_BASE_URL`、`DF_API_URL`、`DF_DAS_EMBEDDING_URL`。

必须真实验证：embedding health/vector、Pipeline LLM `/v1/models`、Review Agent 最小生成、Pipeline Agent 最小 Codex Responses 调用。任何一项失败，不启动实验。

```bash
export no_proxy="${no_proxy},28.59.7.41,127.0.0.1,localhost"
```

## GitHub SSH 代理推送（仅记录，不表示本文档要推送）

`vpn.sh` 只设置 HTTP/HTTPS proxy，Git SSH 还需 ProxyCommand。

```bash
source /jizhicfs/xiaqifeng/vpn.sh
cat >/tmp/git_ssh_http_proxy.sh <<'SH'
#!/bin/sh
exec /usr/bin/ssh \
  -o StrictHostKeyChecking=accept-new \
  -o ConnectTimeout=60 \
  -o "ProxyCommand=/jizhicfs/xiaqifeng/miniconda3/envs/dfe-review/bin/python /tmp/http_connect_proxy.py %h %p" \
  "$@"
SH
chmod 700 /tmp/git_ssh_http_proxy.sh

PUSH_URL=ssh://git@ssh.github.com:443/memoryforget/DataFlow-Evolver.git
GIT_SSH=/tmp/git_ssh_http_proxy.sh \
  git -c remote.origin.pushurl="$PUSH_URL" push origin <branch>
```

代理允许 `ssh.github.com:443`，拒绝 `github.com:22` 和 `github.com:36000`。

## Artifacts 策略

必须保留 `artifacts/downstream_eval`（benchmark 固定数据、manifest、去污染 references）。可删除不兼容旧 baseline cache、失败 run 的 experiment/workspace 目录、失败 DAS cache 和旧 `/tmp` 编译缓存。不得删除输入 corpus、模型或 `artifacts/downstream_eval`。

## 监控与中止条件

持续监控主进程、embedding tmux/health、GPU 边界、rank 缓存、Pipeline rollout、artifact validation、compile/forward、Review、SFT 和下游评测。

出现以下情况立即中止：`Stale file handle`、CUDA OOM、embedding 退出、GPU 7 被训练/评测占用、rank 共享缓存、API 配置缺失或持续不可用、Pipeline Agent 接近上下文上限仍重复探索、自修复后 validation/compile 仍失败、最终数据不是严格 1,000 条。

## 文档创建时状态

- 正式 run 已启动，iteration 0 baseline 运行中
- `bad_cases_per_benchmark=2`
- 评测 ranks 分别绑定 GPU 0/1/2/3，缓存隔离
- GPU 7 embedding 健康
- Review Agent、Pipeline Agent、Pipeline LLM、embedding 已完成真实预检
- 尚未发现 Stale file handle、OOM 或日志异常
