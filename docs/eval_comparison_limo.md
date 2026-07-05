# LIMO 评测脚本对比分析

## 1. System Prompt 对比

### LIMO 原版
```python
system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
```

### DataFlow-Evolve 当前
```python
system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
```

**结论：✅ 完全一致**

---

## 2. 答案提取逻辑对比

### LIMO 使用的提取器
- 位置：`LIMO/eval/utils/parser.py`
- 核心函数：
  - `extract_answer()` - 从生成文本中提取 `\\boxed{}` 内容
  - 支持：LaTeX 嵌套括号、分数修复、sqrt 修复
  - 回退策略：boxed → 末尾数字

### DataFlow-Evolve 当前
- 位置：`scripts/math_benchmarks/answer_extract.py`
- 核心函数：
  - `extract_boxed()` - 提取 `\\boxed{}` 内容（支持嵌套）
  - `extract_gsm8k_answer()` - 提取 `#### <number>`
  - `extract_choice_letter()` - 提取多选题字母
  - `extract_last_number()` - 提取末尾数值
  - `extract_answer(mode="auto")` - 依次尝试 boxed → gsm8k → choice → number

**结论：✅ 逻辑相似，DataFlow 版本更全面（支持多种格式）**

---

## 3. 答案比较逻辑对比

### LIMO 使用的比较器
- 位置：`LIMO/eval/utils/grader.py`
- 核心函数：`math_equal(prediction, reference, ...)`
- 特性：
  1. **字符串精确匹配**（大小写不敏感）
  2. **数值比较**：`isclose(abs_tol=1e-4)`
  3. **符号等价判定**：使用 `sympy`、`latex2sympy`、`parse_latex`
  4. **特殊格式支持**：
     - 多选题字母答案（A-E）
     - 逗号分隔的多答案（排序后比较）
     - 区间/向量 `[a, b]` vs `(a, b)` 
     - 矩阵 `\begin{pmatrix}...\end{pmatrix}`
     - 方程等价（`a=b` vs `a-b=0`）
  5. **超时保护**：使用 `multiprocessing.Process` + 3秒超时
  6. **递归比较**：支持嵌套结构（深度限制5层）

### DataFlow-Evolve 当前
- 位置：`scripts/math_benchmarks/answer_extract.py`
- 核心函数：`is_equiv(pred, gold, tolerance=1e-6)`
- 特性：
  1. **数值比较**：`abs(pred - gold) < 1e-6`
  2. **LaTeX 规范化**：去除 `\\text{}`、`\\mathrm{}`、空格
  3. **字符串匹配**（大小写不敏感）

**结论：⚠️ LIMO 的比较逻辑更强大和完善**

### 关键差异

| 特性 | LIMO | DataFlow-Evolve | 影响 |
|------|------|-----------------|------|
| 符号等价 | ✅ sympy + latex2sympy | ❌ 仅字符串规范化 | **高** - 数学表达式可能被误判 |
| 多答案支持 | ✅ 逗号分隔 + 排序 | ❌ 不支持 | **中** - 多解题目会失败 |
| 矩阵支持 | ✅ pmatrix/bmatrix | ❌ 不支持 | **低** - 线性代数题目 |
| 方程等价 | ✅ a=b ≈ a-b=0 | ❌ 不支持 | **低** - 方程题目 |
| 超时保护 | ✅ 3秒 + 进程隔离 | ❌ 无保护 | **中** - 复杂表达式可能卡住 |
| 数值容差 | `1e-4` | `1e-6` | **低** - 容差差异 |

---

## 4. 数据加载逻辑对比

### LIMO
```python
# LIMO/eval/utils/data_loader.py
def load_data(data_name, split, data_dir):
    # 支持：gsm8k, math, aime, amc, olympiadbench, gpqa, minerva
    # 格式：HuggingFace datasets 或 JSONL
    ...
```

### DataFlow-Evolve
```python
# scripts/math_benchmarks/*.py
# 每个 benchmark 有独立的加载器
def load_gsm8k(cache_dir=None):
    dataset = load_dataset("openai/gsm8k", "main", split="test", cache_dir=cache_dir)
    ...

def load_math500(cache_dir=None):
    dataset = load_dataset("HuggingFaceTB/MATH-500", split="test", cache_dir=cache_dir)
    ...
```

**结论：✅ 逻辑一致，都使用 HuggingFace datasets**

---

## 5. 评测模式对比

### LIMO
- **单样本贪心**（temperature=0）用于大多数 benchmark
- **多样本 pass@k**（temperature>0, n>1）可选

### DataFlow-Evolve
- **小基准**（<50题）：4样本 pass@1（temperature=0.6）
- **大基准**（≥50题）：单样本贪心（temperature=0）

**结论：⚠️ 策略不同**

LIMO 论文中的策略是：
- **全部使用单样本贪心**（temperature=0）
- 只在特定实验中使用 pass@k

DataFlow 当前的小基准 pass@1 策略与 LIMO 论文**不一致**。

---

## 6. 推荐修改

### 优先级 1：答案比较逻辑（高影响）

**建议：使用 LIMO 的 `math_equal` 函数**

```python
# 在 scripts/math_benchmarks/answer_extract.py 中添加
# 或创建新文件 scripts/math_benchmarks/grader.py

# 从 LIMO 复制核心函数：
# - math_equal()
# - symbolic_equal()
# - numeric_equal()
# - parse_digits()
# - 及相关依赖

# 需要安装依赖：
# pip install sympy latex2sympy2 regex
```

### 优先级 2：评测模式一致性（中影响）

**建议：统一使用单样本贪心**

```python
# 修改 scripts/llamafactory_math_sft_eval.py

# 删除小基准/大基准区分
# 所有 benchmark 统一使用：
params = SamplingParams(
    temperature=0,
    max_tokens=MAX_NEW_TOKENS,
    stop_token_ids=[151643, 151645],
)
```

### 优先级 3：数值容差（低影响）

**建议：调整容差为 1e-4**

```python
# 如果使用 LIMO 的 math_equal，这已经包含在内
# 如果保留当前实现，修改：
def numeric_equal(prediction: float, reference: float):
    return isclose(reference, prediction, abs_tol=1e-4)  # 从 1e-6 改为 1e-4
```

---

## 7. 实施步骤

### Step 1: 复制 LIMO grader
```bash
# 创建新文件
cp C:/Users/Administrator/Desktop/LIMO/eval/utils/grader.py \
   C:/Users/Administrator/Desktop/DataFlow-Evolve/scripts/math_benchmarks/

# 创建依赖文件
cp C:/Users/Administrator/Desktop/LIMO/eval/utils/math_normalization.py \
   C:/Users/Administrator/Desktop/DataFlow-Evolve/scripts/math_benchmarks/
```

### Step 2: 修改 llamafactory_math_sft_eval.py
```python
# 导入 LIMO grader
from math_benchmarks.grader import check_is_correct

# 在 run_eval() 中替换比较逻辑：
# 原来：
from math_benchmarks import ..._score
# 改为：
# 在 _score 函数内部使用 check_is_correct(pred, gold)
```

### Step 3: 统一评测模式
```python
# 删除小基准 pass@1 逻辑
# 所有 benchmark 统一单样本贪心
```

### Step 4: 安装依赖
```bash
pip install sympy latex2sympy2 regex word2number
```

---

## 8. 风险评估

| 修改项 | 风险 | 缓解措施 |
|--------|------|----------|
| 替换答案比较逻辑 | **中** - 可能改变历史评测结果 | 保留旧版本，对比新旧结果 |
| 修改评测模式 | **低** - 统一为 LIMO 论文标准 | 文档记录变更 |
| 添加新依赖 | **低** - sympy/regex 稳定 | 添加到 requirements.txt |

---

## 9. 测试建议

1. **小规模对比测试**
   ```bash
   # 运行 10 条样本，对比新旧评测器结果
   python scripts/test_grader_comparison.py --samples 10
   ```

2. **完整回归测试**
   ```bash
   # 在已有模型上重新评测，对比准确率差异
   python scripts/llamafactory_math_sft_eval.py --dataset test_dataset.jsonl --output test_output/
   ```

3. **记录结果差异**
   - 记录哪些样本的判定结果发生了变化
   - 分析是否符合预期（LIMO grader 更准确）

---

## 10. 总结

### 当前状态
- ✅ System prompt 一致
- ✅ 答案提取逻辑基本一致
- ⚠️ **答案比较逻辑较弱**（核心问题）
- ⚠️ 评测模式不一致

### 推荐行动
1. **必须修改**：引入 LIMO 的 `math_equal` 函数（提升评测准确性）
2. **建议修改**：统一评测模式为单样本贪心（与 LIMO 论文一致）
3. **可选优化**：数值容差调整为 1e-4

### 预期影响
- 评测准确率可能提升 2-5%（避免符号等价误判）
- 与 LIMO 论文结果更具可比性
- 减少假阴性（正确答案被判错）
