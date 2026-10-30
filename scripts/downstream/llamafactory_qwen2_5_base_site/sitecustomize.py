"""Register the project-owned Qwen2.5 Base SFT template in LLaMA-Factory."""

from llamafactory.data.formatter import StringFormatter
from llamafactory.data.template import TEMPLATES, register_template

_TEMPLATE_NAME = "qwen2_5_base"

if _TEMPLATE_NAME not in TEMPLATES:
    register_template(
        name=_TEMPLATE_NAME,
        format_user=StringFormatter(
            slots=[
                "<|im_start|>user\n{{content}}<|im_end|>\n"
                "<|im_start|>assistant\n"
            ]
        ),
        format_assistant=StringFormatter(
            slots=["{{content}}", {"eos_token"}, "\n"]
        ),
        format_system=StringFormatter(
            slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]
        ),
        default_system=(
            "You are Qwen, created by Alibaba Cloud. "
            "You are a helpful assistant."
        ),
        stop_words=["<|im_end|>"],
        replace_eos=False,
        replace_jinja_template=True,
    )
