import importlib.util
import json
from pathlib import Path

import pytest


RUNTIME_PATH = (
    Path(__file__).resolve().parents[1]
    / "downstream_runtime/education/scripts/education_runtime.py"
)


def load_runtime():
    spec = importlib.util.spec_from_file_location("education_runtime_test", RUNTIME_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_normalize_sft_dataset_preserves_sharegpt_multiturn(tmp_path):
    runtime = load_runtime()
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "normalized.jsonl"
    messages = [
        {"role": "system", "content": "Tutor carefully."},
        {"role": "user", "content": "I am stuck."},
        {"role": "assistant", "content": "Which step is unclear?"},
        {"role": "user", "content": "The second one."},
        {"role": "assistant", "content": "Let us inspect it."},
    ]
    image = tmp_path / "figure.png"
    image.write_bytes(b"image")
    messages[1]["content"] += " <ImageHere>"
    write_jsonl(source, [{"messages": messages, "images": [str(image)]}])

    report = runtime.normalize_sft_dataset(
        source, destination, messages_field="messages", images_field="images"
    )
    normalized = json.loads(destination.read_text(encoding="utf-8"))

    assert normalized["messages"][1]["content"].endswith(" <image>")
    assert normalized["images"] == [str(image)]
    assert report["formatting"] == "sharegpt"
    assert report["rows"] == 1


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "assistant", "content": "answer first"}],
        [
            {"role": "user", "content": "question"},
            {"role": "user", "content": "another question"},
        ],
        [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": ""},
        ],
    ],
)
def test_normalize_sft_dataset_rejects_invalid_dialogue(tmp_path, messages):
    runtime = load_runtime()
    source = tmp_path / "source.jsonl"
    write_jsonl(source, [{"messages": messages}])

    with pytest.raises(runtime.RuntimeConfigError):
        runtime.normalize_sft_dataset(
            source, tmp_path / "normalized.jsonl", messages_field="messages"
        )


def test_normalize_sft_dataset_rejects_image_count_mismatch(tmp_path):
    runtime = load_runtime()
    image = tmp_path / "figure.png"
    image.write_bytes(b"image")
    source = tmp_path / "source.jsonl"
    write_jsonl(
        source,
        [{
            "messages": [
                {"role": "user", "content": "Inspect this image."},
                {"role": "assistant", "content": "Done."},
            ],
            "images": [str(image)],
        }],
    )

    with pytest.raises(runtime.RuntimeConfigError, match="placeholders"):
        runtime.normalize_sft_dataset(
            source,
            tmp_path / "normalized.jsonl",
            messages_field="messages",
            images_field="images",
        )


def test_synchronize_model_eos_contract_uses_tokenizer_json(tmp_path):
    runtime = load_runtime()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "eos_token": "<|im_end|>",
                "tokenizer_class": "TokenizersBackend",
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "tokenizer.json").write_text(
        json.dumps(
            {
                "added_tokens": [
                    {"id": 248044, "content": "<|endoftext|>"},
                    {"id": 248046, "content": "<|im_end|>"},
                ]
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "config.json").write_text(
        json.dumps({"eos_token_id": None, "model_type": "qwen3_5"}), encoding="utf-8"
    )
    (model_dir / "generation_config.json").write_text(
        json.dumps({"eos_token_id": [248046, 248044]}), encoding="utf-8"
    )

    report = runtime.synchronize_model_eos_contract(model_dir)

    assert report["eos_token"] == "<|im_end|>"
    assert report["eos_token_id"] == 248046
    tokenizer_config = json.loads((model_dir / "tokenizer_config.json").read_text())
    assert tokenizer_config["eos_token_id"] == 248046
    assert tokenizer_config["tokenizer_class"] == "Qwen2Tokenizer"
    assert json.loads((model_dir / "config.json").read_text())["eos_token_id"] == 248046
    assert (
        json.loads((model_dir / "generation_config.json").read_text())["eos_token_id"]
        == 248046
    )
