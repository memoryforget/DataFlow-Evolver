# PTB官方GSM8K评估报告
日期：2026年8月4日

## 评估范围
本报告记录了PostTrainBench官方针对以下模型开展的GSM8K复现评估：
- 基础模型：`/vepfs-mlp2/c20250602/500050/models/Qwen3-1.7B-Base`
- Node0 监督微调（SFT）模型：`workspace/_agent_root/node_0/train_out/sft_model`

本次评估运行使用脚本 `PostTrainBench/src/eval/tasks/gsm8k/evaluate.py`，该脚本调用 `inspect_evals/gsm8k` 评估套件，采用Hugging Face平台上 `openai/gsm8k` 数据集的测试分片完成测试。

## 环境问题修复
此前PTB运行失败，原因是 `inspect_evals/gsm8k` 需要从Hugging Face加载 `openai/gsm8k` 数据集，但运行环境无法直接访问该站点。

修复方案为：启动本地mihomo代理，并在运行评估器时配置如下HTTP代理环境变量：
```bash
HTTP_PROXY=http://127.0.0.1:7890
HTTPS_PROXY=http://127.0.0.1:7890
ALL_PROXY=http://127.0.0.1:7890
```

本次未使用 `ALL_PROXY=socks5://...` 格式的代理，原因是当前 `df-eval` 环境未安装 `socksio` 依赖，httpx网络库不支持SOCKS协议代理。

## 评估结果
| 模型 | 评估协议 | 样本数量 | 准确率 | 标准误 |
| --- | --- | ---: | ---: | ---: |
| Qwen3-1.7B-Base | PTB官方GSM8K | 1319 | 0.1236 | 0.0091 |
| node0 SFT模型 | PTB官方GSM8K | 1319 | 0.1312 | 0.0093 |

## 结果解读
PTB官方评估结果确认，在该评估协议下，基础模型在GSM8K数据集上的准确率约为12%，符合基础模型的预期表现，与此前自定义DataFlow-Evolver评估器得出的70%以上的结果存在明显差异。

此前的高分结果来自完全不同的评估链路：
- 数据源：`DataFlow-Evolver/data/benchmarks/limo_eval/gsm8k/test.jsonl`
- 提示词：自定义提示词「请逐步推理，并将最终答案放在\boxed{}中」
- 评分逻辑：采用自定义的答案提取顺序（boxed标识 → ####标识 → 选项 → 末尾数字），并搭配自定义的数学等价性校验规则

PTB采用的官方评估协议差异如下：
- 数据源：Hugging Face平台 `openai/gsm8k` 数据集的测试分片
- 提示词：Inspect官方GSM8K提示词，要求最终一行以 `ANSWER: $ANSWER` 格式输出答案
- 评分器：`inspect_ai.scorer.match(numeric=True)`
- 少样本设置：Inspect官方GSM8K默认配置10个少样本示例

因此两组分数不具备直接可比性。若用于官方汇报，请采用上述PTB的评估数值。

## 清理说明
本次问题排查过程中创建的临时指标目录如下：
- `workspace/ptb_eval_gsm8k_base_official`
- `workspace/ptb_eval_gsm8k_node0_official`

以上目录仅为本次诊断生成，保存本报告后可删除。