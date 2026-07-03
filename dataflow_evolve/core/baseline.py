"""强基线锚点：奖励策略（绝对/相对增益）与基线流水线源码适配。

设计依据 `docs/design-decisions.md` §2（奖励锚点 / §2.8 冷启动奖励 / penalty 一致性）。

两块互补的小功能集中在此：
1. `RewardPolicy`：把「绝对分奖励（冷启动）」与「Relative Gain 奖励（暖启动）」统一成一个
   对象，并据 `reward_mode` 给出**随模式切换的失败 penalty**，保证失败节点恒为最差。
2. `adapt_baseline_source`：把 DataFlow 自带的强基线流水线源码（硬编码 openai.com/gpt-4o 与
   `empty.jsonl`）以**最小可行方式**（纯正则字符串替换）适配到代理 API + 本地空种子 + 节点缓存目录。
   不重写流水线结构，只改 4 处运行时常量；复用执行子进程注入的 `DF_PIPELINE_API_KEY`。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from dataflow_evolve.utils.logging import get_logger

logger = get_logger("core.baseline")

# 两种奖励模式（与 configs/default.yaml 的 baseline.reward_mode 对应）
REWARD_ABSOLUTE = "absolute"
REWARD_RELATIVE = "relative_to_baseline"


@dataclass
class RewardPolicy:
    """统一封装奖励与失败 penalty 的计算，随 `reward_mode` 自洽切换。

    - 绝对分（冷启动）：`reward = node_dev`，节点分恒 ≥0，`penalty = 0` 即最差。
    - Relative Gain（暖启动）：`reward = node_dev − baseline_dev`，>0 即打败基线；
      但「能跑却逊于基线」的节点奖励为负，会比崩溃节点（0）更低 → 排序错乱。
      故 `penalty = −baseline_dev`（映射到绝对分 0），保证失败恒为最差（见 §2.8）。

    作为纯奖励平移，锚点不改变 UCB 的兄弟排序与 argmax（§2.4 技术点）——
    暖启动用 Relative Gain 与冷启动用绝对分在搜索/选点上等价，只差一个解释性常数。
    """

    reward_mode: str = REWARD_ABSOLUTE
    baseline_dev_score: float | None = None

    def __post_init__(self) -> None:
        if self.reward_mode not in (REWARD_ABSOLUTE, REWARD_RELATIVE):
            raise ValueError(f"未知 reward_mode：{self.reward_mode!r}")
        if self.reward_mode == REWARD_RELATIVE and self.baseline_dev_score is None:
            raise ValueError("relative_to_baseline 模式必须提供 baseline_dev_score")

    @property
    def is_relative(self) -> bool:
        return self.reward_mode == REWARD_RELATIVE

    def reward(self, node_dev_score: float) -> float:
        """把节点的 dev 分换算为回传奖励。"""
        if self.is_relative:
            return node_dev_score - (self.baseline_dev_score or 0.0)
        return node_dev_score

    @property
    def penalty(self) -> float:
        """失败节点（生成/执行/评审/训练失败）的奖励，随模式切换以保证恒为最差。"""
        if self.is_relative:
            return -(self.baseline_dev_score or 0.0)
        return 0.0


# ---------------------------------------------------------------------------
# 基线流水线源码适配（最小可行：仅替换运行时常量，不动结构）
# ---------------------------------------------------------------------------
def _replace_kwarg_literal(src: str, kwarg: str, value: str) -> tuple[str, int]:
    """把 `kwarg="..."` / `kwarg='...'` 的字符串字面量整体替换为 `kwarg="value"`。

    返回 (新源码, 替换次数)。用于 api_url / model_name / first_entry_file_name / cache_path。
    """
    pattern = re.compile(rf"{re.escape(kwarg)}\s*=\s*([\"'])(?:\\.|(?!\1).)*\1")
    replacement = f'{kwarg}="{value}"'
    new_src, n = pattern.subn(replacement, src)
    return new_src, n


def adapt_baseline_source(
    src: str,
    *,
    api_url: str,
    model_name: str,
    key_env: str,
    empty_jsonl_path: str,
    cache_path: str,
) -> str:
    """把强基线流水线源码适配到代理 API / 本地空种子 / 节点缓存目录。

    最小可行实现（见模块 docstring）：纯正则替换 4 处常量 + 注入 `key_name_of_api_key`。
    `APILLMServing_request` 默认从环境变量 `DF_API_KEY` 取密钥，而执行子进程注入的是
    `DF_PIPELINE_API_KEY`，故必须把 `key_name_of_api_key` 显式改写为后者。

    路径里的反斜杠（Windows）会被规整为正斜杠，避免写进源码字符串时被当作转义。
    """
    empty_jsonl_path = empty_jsonl_path.replace("\\", "/")
    cache_path = cache_path.replace("\\", "/")

    out, n_url = _replace_kwarg_literal(src, "api_url", api_url)
    out, n_model = _replace_kwarg_literal(out, "model_name", model_name)
    out, n_entry = _replace_kwarg_literal(out, "first_entry_file_name", empty_jsonl_path)
    out, n_cache = _replace_kwarg_literal(out, "cache_path", cache_path)

    # 注入/改写 key_name_of_api_key：若已存在则替换其值，否则在每个 APILLMServing_request( 后插入
    if re.search(r"key_name_of_api_key\s*=", out):
        out, _ = _replace_kwarg_literal(out, "key_name_of_api_key", key_env)
    else:
        out, n_key = re.subn(
            r"APILLMServing_request\(",
            f'APILLMServing_request(\n                key_name_of_api_key="{key_env}",',
            out,
        )
    logger.info(
        "适配基线源码：api_url=%d model_name=%d first_entry=%d cache_path=%d",
        n_url, n_model, n_entry, n_cache,
    )
    if n_url == 0 or n_entry == 0:
        raise ValueError(
            f"基线源码适配失败：api_url 替换 {n_url} 次、first_entry 替换 {n_entry} 次。"
            "基线流水线格式可能已变更，请核对是否仍使用 APILLMServing_request 与 first_entry_file_name。"
        )
    return out
