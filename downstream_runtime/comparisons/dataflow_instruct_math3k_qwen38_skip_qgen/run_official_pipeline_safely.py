from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

from dataflow.serving import APILLMServing_request


official_pipeline = Path(sys.argv[1]).resolve()
fallback_path = Path(sys.argv[2]).resolve()
original_api_chat_id_retry = APILLMServing_request._api_chat_id_retry
counts = {"serving_retry_exhausted_fallback_count": 0}


def retry_then_empty_fallback(self, *args, **kwargs):
    request_id, response = original_api_chat_id_retry(self, *args, **kwargs)
    if response is None:
        counts["serving_retry_exhausted_fallback_count"] += 1
        response = ""
    return request_id, response


APILLMServing_request._api_chat_id_retry = retry_then_empty_fallback
try:
    runpy.run_path(str(official_pipeline), run_name="__main__")
finally:
    fallback_path.write_text(
        json.dumps(counts, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
