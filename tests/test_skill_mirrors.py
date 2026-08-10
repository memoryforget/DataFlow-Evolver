from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CLAUDE_SKILL = REPO_ROOT / "workspace/.claude/skills/dataflow-dev"
CODEX_SKILL = REPO_ROOT / "workspace/.agents/skills/dataflow-dev"
OPENCODE_SKILL = REPO_ROOT / "workspace/.opencode/skills/dataflow-dev"


def _files(root: Path) -> set[Path]:
    return {path.relative_to(root) for path in root.rglob("*") if path.is_file()}


def test_backend_skills_are_complete_platform_adapted_copies():
    """Keep all native discovery trees in sync without cross-tree indirection."""
    for root, native_path in (
        (CODEX_SKILL, ".agents/skills/dataflow-dev"),
        (OPENCODE_SKILL, ".opencode/skills/dataflow-dev"),
    ):
        assert _files(root) == _files(CLAUDE_SKILL)
        for relative in _files(CLAUDE_SKILL):
            claude = (CLAUDE_SKILL / relative).read_text(encoding="utf-8")
            adapted = (root / relative).read_text(encoding="utf-8")
            assert adapted.replace(
                native_path,
                ".claude/skills/dataflow-dev",
            ) == claude, f"skill copy drifted: {root.name}/{relative}"


def test_backend_skills_never_redirect_to_another_tree():
    roots = {
        CLAUDE_SKILL: ".claude/skills/dataflow-dev",
        CODEX_SKILL: ".agents/skills/dataflow-dev",
        OPENCODE_SKILL: ".opencode/skills/dataflow-dev",
    }
    for root, own_path in roots.items():
        forbidden = set(roots.values()) - {own_path}
        for relative in _files(root):
            text = (root / relative).read_text(encoding="utf-8")
            assert not any(path in text for path in forbidden), relative
