# DataFlow 算子 API 参考模板

## 完整示例：LLM 驱动的生成类算子

基于 `dataflow/operators/reasoning/generate/reasoning_answer_generator.py`

```python
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
import pandas as pd

@OPERATOR_REGISTRY.register()  # 自定义算子跳过此行
class ReasoningAnswerGenerator(OperatorABC):
    """
    为给定问题生成 CoT 推理答案
    """
    def __init__(self, llm_serving: LLMServingABC):
        super().__init__()  # 必须调用，初始化 self.logger
        self.llm_serving = llm_serving  # 必须命名为 llm_serving
        
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return "为给定问题生成推理答案"
        else:
            return "Generate reasoning answers for questions"
    
    def run(
        self, 
        storage: DataFlowStorage,  # 第一个参数
        input_key: str = "question",  # input_ 开头
        output_key: str = "generated_cot"  # output_ 开头
    ):
        """
        执行答案生成
        
        Returns:
            list[str]: 输出字段名列表，如 ['generated_cot']
        """
        # 1. 读取数据
        df = storage.read("dataframe")
        
        # 2. 验证输入列存在
        if input_key not in df.columns:
            raise ValueError(f"Column '{input_key}' not found")
        
        # 3. 准备 LLM 输入
        questions = df[input_key].tolist()  # list[str]
        
        # 4. 调用 LLM（关键 API）
        answers = self.llm_serving.generate_from_input(
            user_inputs=questions,  # list[str]，不是 messages 格式
            system_prompt="You are a math tutor. Provide step-by-step reasoning."
        )
        
        # 5. 处理响应（API 失败时返回 None）
        df[output_key] = [ans if ans is not None else "" for ans in answers]
        
        # 6. 写回存储（关键：只有一个参数！）
        storage.write(df)
        
        # 7. 返回输出字段名
        return [output_key]
```

## 关键 API 签名速查

### FileStorage API
```python
# 读取
df = storage.read("dataframe")  # 返回 pandas.DataFrame

# 写入（只有一个参数！）
storage.write(df)  # ✓ 正确
# storage.write(df, 'dataframe')  # ✗ 错误 - 会崩溃
```

### LLM Serving API
```python
# 正确调用
responses = self.llm_serving.generate_from_input(
    user_inputs=["question 1", "question 2"],  # list[str]
    system_prompt="You are a helpful assistant"  # str
)
# 返回 list[str]，与 user_inputs 一一对应
# API 失败时对应位置返回 None

# 错误调用
# self.llm_serving.request(...)  # ✗ 方法不存在
# self.llm_serving(...)  # ✗ 不是 callable
```

### 错误处理模式
```python
# 模式 A: 过滤 None 值
responses = self.llm_serving.generate_from_input(user_inputs, system_prompt)
valid_responses = [r if r is not None else "" for r in responses]
df[output_key] = valid_responses

# 模式 B: 跳过失败行
results = []
for i, r in enumerate(responses):
    if r is not None:
        results.append(r)
    else:
        self.logger.warning(f"LLM failed for row {i}")
        results.append("")  # 或使用默认值
```

## 常见错误对照表

| 错误写法 | 正确写法 |
|---------|---------|
| `storage.write(df, 'dataframe')` | `storage.write(df)` |
| `self.llm_serving.request(prompt)` | `self.llm_serving.generate_from_input(user_inputs, system_prompt)` |
| `self.llm_serving(prompt)` | `self.llm_serving.generate_from_input(user_inputs, system_prompt)` |
| `from dataflow.operators.reasoning.generate.xxx import Xxx` | `from dataflow.operators.reasoning import Xxx` |
| `def run(self, storage)` 无返回值 | `def run(self, storage): ... return ['output_key']` |
