from pathlib import Path


def test_launcher_adds_node_directory_before_codex_version_check():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "run_gsm8k_5k_compare.sh"
    ).read_text(encoding="utf-8")

    node_setup = script.index('if [[ -n "${NODE_BIN:-}" ]]')
    version_check = script.index('CODEX_VERSION="$("${CODEX_BIN}" --version')

    assert node_setup < version_check
    assert 'export PATH="${NODE_DIR}:${PATH}"' in script
    assert 'export PATH="${CODEX_DIR}:${PATH}"' in script
