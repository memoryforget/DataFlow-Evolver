"""pytest 全局配置：禁止可选模型后端意外联网，保证测试快速、可重复。"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
