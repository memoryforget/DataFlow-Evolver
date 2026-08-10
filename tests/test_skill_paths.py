from __future__ import annotations

import pytest

from dataflow_evolver.llm.agent_sdk import default_system_prompt_append
from dataflow_evolver.llm.skill_paths import SKILL_ROOTS, skill_root_for_backend


@pytest.mark.parametrize("backend, expected", sorted(SKILL_ROOTS.items()))
def test_backend_uses_native_skill_root(backend: str, expected: str):
    assert skill_root_for_backend(backend) == expected


@pytest.mark.parametrize("backend, expected", sorted(SKILL_ROOTS.items()))
def test_default_prompt_mentions_only_selected_skill_root(
    backend: str, expected: str
):
    prompt = default_system_prompt_append(backend)
    assert f"{expected}/templates/operator_api_reference.md" in prompt
    assert f"{expected}/diagnostics/known_issues.md" in prompt
    for other_backend, other_root in SKILL_ROOTS.items():
        if other_backend != backend:
            assert other_root not in prompt


def test_unknown_backend_has_no_fallback_to_claude():
    with pytest.raises(ValueError, match="unknown agent backend"):
        skill_root_for_backend("other")
