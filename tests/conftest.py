"""pytest 全局配置：保证单元测试离线、快速、可重复。

强制 HuggingFace 离线模式。posttrainbench_sft_eval.convert_dataset 会用
AutoTokenizer.from_pretrained(base_model) 统计 token 长度，失败则降级为字符粗估；
测试传的是 dummy 模型名，若 transformers 已安装且允许联网，它会把 dummy 名当作 Hub
repo id 去解析并长时间重试，导致测试挂死。置为离线后它快速失败并走既定降级分支。

必须在模块导入期设置：huggingface_hub 在 import 时就把这些变量读成模块常量，
放进 fixture 会晚于 test 模块的收集与导入而失效。setdefault 保留外部显式覆盖的余地。
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
