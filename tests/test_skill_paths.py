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
    assert "Activate the project skill" in prompt
    assert "`dataflow-evolver-pipeline`" in prompt
    assert "authoritative API/runtime" in prompt
    assert "contract" in prompt
    assert all(root not in prompt for root in SKILL_ROOTS.values())


def test_unknown_backend_has_no_fallback_to_claude():
    with pytest.raises(ValueError, match="unknown agent backend"):
        skill_root_for_backend("other")
