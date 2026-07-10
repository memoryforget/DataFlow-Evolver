"""Resume 冒烟测试（真实 Claude Agent SDK）。

验证 `ClaudeAgentOptions(resume=<session_id>)` 是否真的载入上一次会话的历史续跑，
以及 `agent_sdk._run` 是否正确捕获 session_id。用"nonce 记忆"法证明：

  phase1（全新会话）：让 agent 把一个随机 nonce 写进 smoke_a.txt，并记下会话 session_id；
  phase2（resume 该会话）：提示里【不出现 nonce】，只让它"把上次写进 smoke_a.txt 的那一行
                            原样写进 smoke_b.txt"。

若 smoke_b.txt 的内容 == phase1 生成的 nonce，则证明 resume 载入了历史（agent 只能从被
resume 的对话历史里得知那个 nonce）。否则说明 resume 未生效 / transcript 未落盘。

前置：需要 ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL 就绪、代理可用（会消耗少量真实 token）。
运行：  python scripts/smoke_resume.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataflow_evolver.llm.agent_sdk import _run  # 真实 SDK 调用 + session_id 捕获

# 与正式 run 一致的模型（避免模型名不被代理识别导致误判）；可用 DF_SMOKE_MODEL 覆盖。
_MODEL = os.environ.get("DF_SMOKE_MODEL", "claude-opus-4-8")


def _build_min_options(cwd: str, resume: str | None = None):
    """最小 options：放开内置工具（含 Write），无头自动批准；可选 resume。"""
    from claude_agent_sdk import ClaudeAgentOptions

    return ClaudeAgentOptions(
        cwd=cwd,
        model=_MODEL,
        tools={"type": "preset", "preset": "claude_code"},
        permission_mode="bypassPermissions",
        max_turns=8,
        resume=resume,
    )


def main() -> int:
    nonce = f"NONCE-{uuid.uuid4().hex[:12]}"
    with tempfile.TemporaryDirectory(prefix="smoke_resume_") as d:
        cwd = str(Path(d))

        # ---- phase 1：全新会话，写 nonce 到 smoke_a.txt ----
        p1 = (
            f"Create a file named smoke_a.txt in the current directory whose only content "
            f"is exactly this single line: {nonce}\n"
            f"After writing it, reply with the word DONE."
        )
        print(f"[phase1] 全新会话，nonce={nonce}")
        r1 = asyncio.run(_run(p1, _build_min_options(cwd)))
        print(f"[phase1] success={r1.success} session_id={r1.session_id} error={r1.error}")
        if not r1.success:
            print("[FAIL] phase1 运行失败（代理/网络问题？）")
            return 1
        if not r1.session_id:
            print("[FAIL] phase1 未捕获到 session_id —— 无法 resume")
            return 1
        a = Path(cwd) / "smoke_a.txt"
        if not a.exists():
            print("[FAIL] phase1 没写出 smoke_a.txt")
            return 1
        print(f"[phase1] smoke_a.txt 内容={a.read_text(encoding='utf-8').strip()!r}")

        # ---- phase 2：resume 该会话，提示里不含 nonce ----
        p2 = (
            "Earlier in THIS conversation you created smoke_a.txt with a single line. "
            "Now create a new file smoke_b.txt in the current directory whose only content "
            "is EXACTLY that same line you wrote before (reuse the value from your memory of "
            "this conversation; do not read the file). Then reply with the exact value you used."
        )
        print(f"[phase2] resume={r1.session_id}（提示不含 nonce）")
        r2 = asyncio.run(_run(p2, _build_min_options(cwd, resume=r1.session_id)))
        print(f"[phase2] success={r2.success} session_id={r2.session_id} error={r2.error}")
        if not r2.success:
            print("[FAIL] phase2 resume 运行失败")
            return 1
        b = Path(cwd) / "smoke_b.txt"
        if not b.exists():
            print("[FAIL] phase2 没写出 smoke_b.txt")
            return 1
        b_content = b.read_text(encoding="utf-8").strip()
        print(f"[phase2] smoke_b.txt 内容={b_content!r}")

        # ---- 判定 ----
        if nonce in b_content:
            print(f"\n[PASS] resume 生效：phase2 仅凭会话历史复现了 nonce {nonce} ✓")
            return 0
        print(
            f"\n[FAIL] smoke_b.txt 未包含 phase1 的 nonce（期望含 {nonce}，实际 {b_content!r}）——"
            "resume 可能未载入历史 / transcript 未落盘。"
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
