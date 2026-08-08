from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CLAUDE_SKILL = REPO_ROOT / "workspace/.claude/skills/dataflow-dev"
CODEX_SKILL = REPO_ROOT / "workspace/.agents/skills/dataflow-dev"


def _files(root: Path) -> set[Path]:
    return {path.relative_to(root) for path in root.rglob("*") if path.is_file()}


def test_codex_skill_is_a_complete_platform_adapted_copy():
    """Keep both native discovery trees in sync without cross-tree indirection."""
    assert _files(CODEX_SKILL) == _files(CLAUDE_SKILL)
    for relative in _files(CLAUDE_SKILL):
        claude = (CLAUDE_SKILL / relative).read_text(encoding="utf-8")
        codex = (CODEX_SKILL / relative).read_text(encoding="utf-8")
        assert codex.replace(
            ".agents/skills/dataflow-dev",
            ".claude/skills/dataflow-dev",
        ) == claude, f"skill copy drifted: {relative}"


def test_codex_skill_never_redirects_to_claude_tree():
    for relative in _files(CODEX_SKILL):
        text = (CODEX_SKILL / relative).read_text(encoding="utf-8")
        assert ".claude/skills/dataflow-dev" not in text, relative
