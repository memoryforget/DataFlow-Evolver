from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOTS = (
    REPO_ROOT / "workspace/.claude/skills/dataflow-evolver-pipeline",
    REPO_ROOT / "workspace/.agents/skills/dataflow-evolver-pipeline",
    REPO_ROOT / "workspace/.opencode/skills/dataflow-evolver-pipeline",
)


def _files(root: Path) -> set[Path]:
    return {path.relative_to(root) for path in root.rglob("*") if path.is_file()}


def test_specialized_backend_skills_are_byte_identical():
    canonical = SKILL_ROOTS[0]
    canonical_files = _files(canonical)
    assert canonical_files == {
        Path("SKILL.md"),
        Path("references/serving-contract.md"),
    }
    for root in SKILL_ROOTS[1:]:
        assert _files(root) == canonical_files
        for relative in canonical_files:
            assert (root / relative).read_bytes() == (canonical / relative).read_bytes()
