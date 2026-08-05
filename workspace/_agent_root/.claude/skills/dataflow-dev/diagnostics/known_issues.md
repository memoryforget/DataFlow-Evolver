# DataFlow 已知问题数据库

> **文件状态**：可追加更新
> **格式约定**：每条 Issue 包含：编号、标题、症状关键词、根因、解决方案、修复代码示例
> **使用方式**：诊断时先按"症状关键词"匹配，命中则直接给出根因和方案

---

## Issue #001 — 配置参数命名警告（非错误）

**标题**：`Unexpected key 'xxx' in operator` 警告

**症状关键词**：
- `Unexpected key`
- `in operator`
- `UserWarning`

**根因**：

`OPERATOR_REGISTRY.register()` 会校验 Pipeline 配置文件（YAML/JSON）中传入的 key 是否在算子的 `__init__` 参数列表中。若 key 不匹配（如大小写不同、多余参数），会抛出 `UserWarning`，但**不会中断执行**。

**解决方案**：

检查 Pipeline 配置文件中对应算子的参数名拼写，确保与算子 `__init__` 参数名完全一致（区分大小写）。

**修复示例**：

```yaml
# ❌ 错误
operators:
  - type: MyFilter
    input_Key: "text"   # 大小写错误

# ✅ 正确
operators:
  - type: MyFilter
    input_key: "text"
```

---

## Issue #002 — 算子注册缺失

**标题**：`No object named 'Xxx' found in 'operators' registry`

**症状关键词**：
- `No object named`
- `found in 'operators' registry`
- `RegistryError`

**根因**：

新建算子文件后，未在对应模块的 `__init__.py` 的 `TYPE_CHECKING` 块中注册。LazyLoader 依赖此注册才能找到类。

**解决方案**：

在对应模块的 `__init__.py` 中添加 import：

```python
# dataflow/operators/<module>/__init__.py

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .filter.my_new_filter import MyNewFilter
```

**修复示例**：

```python
# dataflow/operators/general_text/__init__.py
if TYPE_CHECKING:
    from .filter.my_new_filter import MyNewFilter  # ← 新增这行
```

---

## Issue #003 — Pipeline key 不一致

**标题**：`Key Matching Error` / `does not match any output keys`

**症状关键词**：
- `Key Matching Error`
- `does not match any output keys`
- `KeyError` + Pipeline 上下文

**根因**：

Pipeline 中前一个算子的 `output_key` 与后一个算子的 `input_key` 不一致，或 `storage.write()` 写入的列名与后续 `storage.read()` 读取的列名不匹配。

**解决方案**：

逐步检查 `forward()` 中每个算子调用的 `input_key` / `output_key`，确保相邻步骤的字段名严格一致。

**修复示例**：

```python
# ❌ 错误：step1 输出 "result"，step2 读取 "output"
self.step1_op.run(storage=self.storage.step(), output_key="result")
self.step2_op.run(storage=self.storage.step(), input_key="output")  # 找不到 "output"

# ✅ 正确：保持一致
self.step1_op.run(storage=self.storage.step(), output_key="cleaned_text")
self.step2_op.run(storage=self.storage.step(), input_key="cleaned_text")
```

---

## Issue #004 — 缺少 storage.step()

**标题**：`You must call storage.step() before`

**症状关键词**：
- `You must call storage.step() before`
- `storage has not been stepped`
- `AssertionError` + storage 上下文

**根因**：

在 Pipeline `forward()` 中调用算子时，没有在每个 `run()` 调用时传递 `storage=self.storage.step()`，或使用了错误的用法（如 `self.storage` 而非 `self.storage.step()`）。

**解决方案**：

- Pipeline `forward()` 中：每次 `op.run()` 都传 `storage=self.storage.step()`
- 独立测试脚本中：先调用 `storage.step()`，再传给 `op.run()`

**修复示例**：

```python
# ❌ 错误（Pipeline 中）
self.op.run(storage=self.storage, input_key="text")

# ✅ 正确（Pipeline 中）
self.op.run(storage=self.storage.step(), input_key="text")

# ✅ 正确（独立测试脚本中）
storage = FileStorage(...)
storage.step()  # 手动推进一步
op.run(storage=storage, input_key="text")
```

---

## Issue #005 — DummyStorage 不支持 get_keys_from_dataframe

**标题**：`DummyStorage` + `AttributeError` / `TypeError`

**症状关键词**：
- `DummyStorage`
- `AttributeError: 'DummyStorage' object has no attribute`
- `TypeError` + DummyStorage 上下文

**根因**：

`DummyStorage` 是一个测试用 stub，不实现 `get_keys_from_dataframe()` 及其他完整 Storage 方法。在 Pipeline 中使用 `DummyStorage` 会导致算子调用这些方法时抛出错误。

**解决方案**：

- Pipeline 中**必须**使用 `FileStorage`，禁止使用 `DummyStorage`
- `DummyStorage` 仅用于极简单元测试，不涉及 dataframe 操作时才可使用

**修复示例**：

```python
# ❌ 错误（Pipeline 中使用 DummyStorage）
from dataflow.utils.storage import DummyStorage
storage = DummyStorage()

# ✅ 正确（Pipeline 中使用 FileStorage）
from dataflow.utils.storage import FileStorage
storage = FileStorage(
    first_entry_file_name="./data/input.jsonl",
    cache_path="./cache",
    file_name_prefix="dataflow_cache_step",
    cache_type="jsonl",
)
```

---

## Issue #006 — re.split() 捕获组产生 None

**标题**：`AttributeError: 'NoneType' object has no attribute 'strip'` + `re.split`

**症状关键词**：
- `AttributeError: 'NoneType' object has no attribute 'strip'`
- `re.split`
- `NoneType` + 字符串操作

**根因**：

在 `re.split()` 的 pattern 中使用了**捕获组** `(...)`（包括可选捕获组 `(...)?`）。Python 的 `re.split()` 在使用捕获组时，会把匹配到的分隔符（或 `None`，若是可选组未匹配）插入结果列表中，导致列表中存在 `None` 值，后续调用 `.strip()` 等字符串方法时报错。

**解决方案**：

将所有 `re.split()` pattern 中的捕获组 `(...)` 改为**非捕获组** `(?:...)`。

**修复示例**：

```python
import re

text = "Step 1: foo\nStep 2: bar"

# ❌ 错误：捕获组导致 None 混入结果
parts = re.split(r"(Step \d+:)", text)
# ['', 'Step 1:', ' foo\n', 'Step 2:', ' bar']
# 结果中包含分隔符本身，可能混入 None（尤其是可选组）

# ❌ 更危险：可选捕获组
parts = re.split(r"(\n+)?##", text)
# None 会出现在 \n 未匹配时

# ✅ 正确：非捕获组
parts = re.split(r"(?:Step \d+:)", text)
parts = re.split(r"(?:\n+)?##", text)

# 或者直接过滤 None
parts = [p for p in re.split(r"(Step \d+:)", text) if p is not None]
```

---

## Issue #007 — @prompt_restrict 装饰器位置错误

**标题**：`@prompt_restrict` 未生效或报错

**症状关键词**：
- `prompt_restrict`
- 装饰器未生效
- Prompt 类型限制失效

**根因**：

`@prompt_restrict(MyPrompt)` 装饰器必须**紧贴类定义上方**。若在 `@prompt_restrict` 和 `class MyOperator` 之间插入了其他装饰器（如 `@OPERATOR_REGISTRY.register()`），则 `@prompt_restrict` 作用于了 `register()` 的返回值而非类本身，导致限制失效。

**解决方案**：

确保装饰器顺序为：`@OPERATOR_REGISTRY.register()` 在最外层，`@prompt_restrict` 紧贴类定义：

```python
# ❌ 错误：@prompt_restrict 被 @register 隔开
@OPERATOR_REGISTRY.register()
@prompt_restrict(MyPrompt)
class MyOperator(OperatorABC):
    ...

# ✅ 正确：@prompt_restrict 紧贴类
@OPERATOR_REGISTRY.register()
class MyOperator(OperatorABC):
    ...
# 注：实际上 @prompt_restrict 应在类定义上方、@register 下方
# 请查阅最新 DataFlow 源码确认正确的叠加顺序
```

**注**：该 Issue 的准确修复依赖具体 DataFlow 版本，建议查阅最新源码中 `@prompt_restrict` 的实际用法示例。

---

## Issue #008 — LazyLoader 子包路径 import 失败

**标题**：`ModuleNotFoundError` + `dataflow.operators.reasoning.refine`

**症状关键词**：
- `ModuleNotFoundError`
- `dataflow.operators.reasoning.refine`
- `ImportError` + 算子类名

**根因**：

DataFlow 使用 LazyLoader 机制，算子类通过父模块的 `__init__.py` 注册并懒加载。直接 import 子包路径（如 `from dataflow.operators.reasoning.refine.cot_llm_judge_refiner import CoTLLMJudgeRefiner`）会绕过 LazyLoader，在部分场景下导致 `ModuleNotFoundError`。

**解决方案**：

必须从父模块 import：

```python
# ✅ 正确
from dataflow.operators.reasoning import CoTLLMJudgeRefiner

# ❌ 错误
from dataflow.operators.reasoning.refine.cot_llm_judge_refiner import CoTLLMJudgeRefiner
```

---

## Issue #009 — json_schema 严格模式缺 additionalProperties

**标题**：`Invalid schema for response_format` / `'additionalProperties' is required to be supplied and to be false`

**症状关键词**：
- `Invalid schema for response_format`
- `additionalProperties' is required`
- `API request failed ... status=400` + `response_format`
- 算子（如 `PromptedGenerator`）传了 `json_schema=` 后输出全为 `null`、下游 step 被过滤成空（`Final shape: (0, N)`），但**进程仍 exit 0**（错误被 `APILLMServing_request` 吞掉、重试耗尽后返回 None）

**根因**：

`APILLMServing_request` 在构造 `response_format` 时硬编码 `"strict": True`（见 `dataflow/serving/api_llm_serving_request.py`）。OpenAI 的严格结构化输出要求 schema 里**每一个 `type: object` 都必须显式写 `"additionalProperties": false`**。若算子传入的 `json_schema` 缺这个字段，服务端返回 HTTP 400，5 次重试全失败 → 该行结果写成 `null`。

**注意**：这是**数据质量级**失败而非崩溃级——流水线不会抛栈、不会非零退出，因此不会触发执行期自修正；只会在产出里表现为字段全空。务必在生成阶段就写对 schema。

**解决方案**：

凡是传给算子的 `json_schema`，每个 object 层级都补 `"additionalProperties": False`。

**修复示例**：

```python
# ❌ 错误：strict 模式下会被服务端 400 拒绝
json_schema={
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "answer": {"type": "string"},
    },
    "required": ["question", "answer"],
}

# ✅ 正确：每个 object 都显式声明 additionalProperties: False
json_schema={
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "question": {"type": "string"},
        "answer": {"type": "string"},
    },
    "required": ["question", "answer"],
}
```

> 嵌套 object 同理：内层 `"type": "object"` 也要各自带 `"additionalProperties": False`。

---

## Issue #010 — APILLMServing_request 调用方式错误

**标题**：`AttributeError: 'APILLMServing_request' object has no attribute 'request'` / `TypeError: 'APILLMServing_request' object is not callable`

**症状关键词**：
- `AttributeError: 'APILLMServing_request' object has no attribute 'request'`
- `TypeError: 'APILLMServing_request' object is not callable`
- LLM 调用返回 None
- `messages[1].content: got object, want string` (API 400 错误)
- 生成结果全为空或占位文本

**根因**：

`APILLMServing_request` 是一个类，需要实例化后调用 `generate_from_input()` 方法。常见错误：
1. 尝试调用不存在的 `.request()` 方法
2. 直接把实例当函数调用 `self.llm_serving(...)`
3. 传入错误的参数格式（messages 而非 user_inputs）

**解决方案**：

正确的调用方式：

```python
# 1. 在算子 __init__ 中接收 serving 实例
def __init__(self, llm_serving, ...):
    super().__init__()
    self.llm_serving = llm_serving  # 必须命名为 llm_serving

# 2. 在 run() 中调用 generate_from_input()
def run(self, storage, input_key="text", output_key="result"):
    df = storage.read("dataframe")
    
    # 准备输入
    user_inputs = df[input_key].tolist()  # list[str]
    system_prompt = "You are a helpful assistant"
    
    # 正确调用
    responses = self.llm_serving.generate_from_input(
        user_inputs=user_inputs,
        system_prompt=system_prompt
    )
    
    # responses 是 list[str]，与 user_inputs 一一对应
    # 处理 None 值（API 失败时返回 None）
    df[output_key] = [r if r is not None else "" for r in responses]
    
    storage.write(df)
    return [output_key]
```

**完整 API 签名**：

```python
APILLMServing_request.generate_from_input(
    user_inputs: list[str],           # 必需：用户输入列表
    system_prompt: str = "",           # 可选：系统提示
    json_schema: dict | None = None,   # 可选：结构化输出 schema
    **kwargs                           # 其他参数传递给 API
) -> list[str]
```

**注意事项**：
- 输入必须是 `list[str]`，不是 `messages` 格式
- 返回值是 `list[str]`，与输入一一对应
- 如果 API 调用失败，对应位置返回 `None`
- **必须在算子内部处理 `None` 值**，否则下游算子会崩溃

**修复示例**：

```python
# ❌ 错误 1：调用不存在的方法
response = self.llm_serving.request(prompt=user_input)

# ❌ 错误 2：把实例当函数
response = self.llm_serving(user_input, system_prompt)

# ❌ 错误 3：传入 messages 格式
response = self.llm_serving.generate_from_input(
    messages=[{"role": "user", "content": "..."}]  # 错误格式
)

# ✅ 正确
responses = self.llm_serving.generate_from_input(
    user_inputs=["question 1", "question 2"],
    system_prompt="You are a math tutor"
)
```

---

## Issue #011 — FileStorage.write() 参数数量错误

**标题**：`TypeError: FileStorage.write() takes 2 positional arguments but 3 were given`

**症状关键词**：
- `FileStorage.write() takes 2 positional arguments but 3 were given`
- `storage.write()` 报错
- Pipeline 执行崩溃在 `storage.write(df, ...)` 行

**根因**：

`FileStorage.write()` 只接受一个参数（DataFrame），不接受第二个参数指定存储格式。这是最常见的 API 误用错误。

**解决方案**：

正确的 API 签名：

```python
storage.write(df)  # ✓ 只传 DataFrame
```

存储格式在 FileStorage 初始化时通过 `cache_type` 参数指定：

```python
storage = FileStorage(
    first_entry_file_name="input.jsonl",
    cache_path="./cache",
    cache_type="jsonl"  # 在这里指定格式，不是在 write() 时指定
)
```

**修复示例**：

```python
# ❌ 错误：传入两个参数
storage.write(df, 'dataframe')
storage.write(df, 'jsonl')
storage.write(df, format='jsonl')

# ✅ 正确：只传 DataFrame
storage.write(df)
```

**完整 API 签名**：

```python
FileStorage.write(df: pd.DataFrame) -> None
```

---

## Issue #012 — compile 阶段误读尚未生成的 step cache

**标题**：`pipeline.compile()` 中报 `dataflow_cache_step_stepN.jsonl does not exist`

**症状关键词**：
- traceback 位于 `pipeline.compile()` 调用期间
- `FileNotFoundError: ... dataflow_cache_step_stepN.jsonl does not exist`
- `forward()` 在算子调用后执行了 `storage.read(...)`、`to_json(...)` 或其他手工导出

**根因**：

`PipelineABC.compile()` 会临时把成员算子包装为 `AutoOP`，再调用 `forward()` 记录算子拓扑和参数。
这个阶段不会执行算子的 `run()` 方法，因此任何 step cache 都还没有生成。若 `forward()` 在算子调用之外
直接读取最终 storage 或导出文件，compile 阶段必然读到不存在的 cache。

**解决方案**：

PipelineABC 的 `forward()` 只描述算子调用；每个算子在自身 `run()` 中读取上一步并通过
`storage.write(df)` 写出本步。最后一个算子的 step 即最终结果，DataFlow-Evolver 的
ExecutionWrapper 会自动发现编号最大的非空 step JSONL。

```python
class MyPipeline(PipelineABC):
    def forward(self):
        self.op_step1.run(storage=self.storage.step(), ...)
        self.op_step2.run(storage=self.storage.step(), ...)
        # 不要在这里 storage.read(...) 或手工 to_json(...)


if __name__ == "__main__":
    pipeline = MyPipeline()
    pipeline.compile()
    if os.environ.get("DF_COMPILE_ONLY"):
        sys.exit(0)
    pipeline.forward()
```

**禁止的伪修复**：把 `DF_COMPILE_ONLY` 判断移动到 `pipeline.compile()` 之前。那只会跳过 compile，
命令虽然退出码为 0，却没有验证任何流水线语义。

---

## 快速匹配表

| 报错关键词 | Issue 编号 |
|---|---|
| `Unexpected key 'xxx' in operator` | #001 |
| `No object named 'Xxx' found in 'operators' registry` | #002 |
| `Key Matching Error` / `does not match any output keys` | #003 |
| `You must call storage.step() before` | #004 |
| `DummyStorage` + `AttributeError` / `TypeError` | #005 |
| `'NoneType' object has no attribute 'strip'` + `re.split` | #006 |
| `prompt_restrict` 未生效 | #007 |
| `ModuleNotFoundError` + `dataflow.operators.reasoning.refine` | #008 |
| `Invalid schema for response_format` / `additionalProperties' is required` | #009 |
| `'APILLMServing_request' object has no attribute 'request'` | #010 |
| `'APILLMServing_request' object is not callable` | #010 |
| `FileStorage.write() takes 2 positional arguments but 3 were given` | #011 |
| `dataflow_cache_step_stepN.jsonl does not exist` 出现在 `pipeline.compile()` | #012 |

---

*本文件由 dataflow-dev skill 维护，新增 Issue 时请同步更新 SKILL.md 诊断流程的快速匹配表。*
