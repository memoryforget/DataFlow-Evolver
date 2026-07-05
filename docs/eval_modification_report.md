# 评测脚本修改完成报告

## 修改日期
2026-07-05

## 修改目标
使 DataFlow-Evolve 的评测脚本与 LIMO 论文保持一致，提升答案比较的准确性。

## 已完成的修改

### 1. ✅ 集成 LIMO Grader（优先级：高）

#### 复制的文件
- `scripts/math_benchmarks/grader.py` - LIMO 的完整答案比较逻辑
- `scripts/math_benchmarks/math_normalization.py` - 符号等价判定依赖
- `scripts/math_benchmarks/parser.py` - 答案规范化工具（简化版）

#### 核心功能
- **符号等价判定**：使用 `sympy`、`latex2sympy`、`parse_latex` 进行数学表达式等价判定
- **多答案支持**：逗号分隔的多解（排序后比较）
- **矩阵支持**：`\begin{pmatrix}...\end{pmatrix}` 和 `\begin{bmatrix}...\end{bmatrix}`
- **方程等价**：`a=b` 等价于 `a-b=0`
- **超时保护**：使用 `multiprocessing.Process` + 3秒超时，防止复杂表达式卡住
- **递归比较**：支持嵌套结构（深度限制5层）
- **数值容差**：`abs_tol=1e-4`（与 LIMO 一致）

### 2. ✅ 统一评测模式（优先级：中）

#### 修改前
- **小基准**（<50题）：4样本 pass@1（temperature=0.6, top_p=0.95）
- **大基准**（≥50题）：单样本贪心（temperature=0）

#### 修改后
- **所有 benchmark**：统一使用单样本贪心（temperature=0）
- 与 LIMO 论文完全一致

#### 删除的代码
- `TEST_SMALL_BENCHMARKS` 常量定义
- `TEST_LARGE_BENCHMARKS` 常量定义
- 小基准 pass@1 计算逻辑（~120行代码）
- `math.comb` 相关的 pass@k 计算

### 3. ✅ 修改评测脚本逻辑

#### `run_eval()` 函数变化

**修改前**：
```python
from math_benchmarks import (
    ..., gsm8k_score, math500_score, ...
)

# 调用每个 benchmark 的 score_fn
accuracy, bad_cases = score_fn(predictions, examples)
```

**修改后**：
```python
from math_benchmarks.grader import check_is_correct
from math_benchmarks.answer_extract import extract_answer

# 统一使用 LIMO grader
for pred_text, ex in zip(predictions, examples):
    pred_answer = extract_answer(pred_text, mode="auto")
    gold_answer = ex.get("answer") or ex.get("gold_answer", "")
    is_correct = check_is_correct(pred_answer, gold_answer, timeout=True)
```

#### 新增字段
在 bad_cases 中新增：
- `extracted_answer`：从生成文本中提取的答案
- 方便后续诊断分析

### 4. ✅ 添加依赖

#### `pyproject.toml` 修改
```toml
[project.optional-dependencies]
train = [
    # ... 原有依赖 ...
    # --- 数学评测：答案等价判定（LIMO grader 依赖） ---
    "sympy>=1.12",
    "latex2sympy2>=1.9",
    "regex>=2023.0",
]
```

#### 安装方式
```bash
# 在 train 环境中安装
pip install -e ".[train]"
```

### 5. ✅ 保留的修改（之前完成）

- ✅ 数据集长度过滤（CUTOFF_LEN=16384）
- ✅ 使用真实 tokenizer 精确计算 token 数
- ✅ 异常处理 + 回退机制

## 预期影响

### 准确率提升
- **估计提升 2-5%**：避免符号等价误判（如 `\frac{1}{2}` vs `0.5`）
- 减少假阴性（正确答案被判错）

### 与 LIMO 论文可比性
- 评测模式完全一致
- 答案比较逻辑完全一致
- 结果可直接与论文对比

### 潜在问题
- **历史结果不可比**：新旧评测器结果会有差异（预期是新版更准确）
- **计算开销略增**：符号等价判定比字符串匹配慢（但有3秒超时保护）

## 测试建议

### 1. 小规模验证
```bash
# 准备一个小测试集（10-20条）
python scripts/llamafactory_math_sft_eval.py \
    --dataset test_small.jsonl \
    --output test_output/ \
    --mode dev
```

### 2. 对比测试
在已有模型上重新评测，对比新旧结果：
- 记录哪些样本的判定结果发生了变化
- 分析变化是否符合预期（应该是新版更准确）

### 3. 回归测试
在 MATH-train 上完整评测，确保：
- 脚本运行正常
- 结果合理（准确率在预期范围内）
- 无崩溃或超时

## 回滚方案

如果发现问题，可以通过 git 回滚：
```bash
git checkout HEAD~1 scripts/llamafactory_math_sft_eval.py
git checkout HEAD~1 pyproject.toml
rm scripts/math_benchmarks/grader.py
rm scripts/math_benchmarks/math_normalization.py
rm scripts/math_benchmarks/parser.py
```

## 文件清单

### 新增文件
- `scripts/math_benchmarks/grader.py` (516行)
- `scripts/math_benchmarks/math_normalization.py`
- `scripts/math_benchmarks/parser.py` (169行)
- `docs/eval_comparison_limo.md` (对比分析文档)

### 修改文件
- `scripts/llamafactory_math_sft_eval.py` (删除 ~150行，修改 ~50行)
- `pyproject.toml` (添加 3 个依赖)

### 未修改
- `scripts/math_benchmarks/answer_extract.py` (保留，作为辅助工具)
- 其他 benchmark 加载器（gsm8k.py, math500.py 等）

## 后续建议

1. **更新文档**：在 README 中说明评测方式已更新
2. **记录基准**：在新版评测器上重新建立基准结果
3. **CI 集成**：考虑在 CI 中运行小规模评测，确保代码变更不影响评测

## 参考

- LIMO 论文：https://arxiv.org/abs/2502.03387
- LIMO 评测代码：`C:\Users\Administrator\Desktop\LIMO\eval\`
- 对比分析文档：`docs/eval_comparison_limo.md`
