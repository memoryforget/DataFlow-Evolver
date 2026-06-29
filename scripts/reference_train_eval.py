"""参考用外部训练/评估脚本——演示并文档化 DownstreamTrainerWrapper 的 I/O 契约。

契约：
    python scripts/reference_train_eval.py --dataset <ds.jsonl> --output <out_dir>
脚本必须在 <out_dir>/results.json 回写：
    {
      "score": float,
      "metric_name": str,
      "bad_cases": [{"input","prediction","ground_truth","error_type"}, ...]
    }

本参考实现不做真实训练：它读取数据集，用一个朴素的"答案非空"判据打一个可复现的
伪分数，并把若干条不合格样例作为 bad_cases 输出。把它替换成你自己的训练+评估逻辑即可。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def evaluate(rows: list[dict]) -> dict:
    """朴素评估占位：把含非空 answer/output 的样本视为"通过"。"""
    if not rows:
        return {"score": 0.0, "metric_name": "non_empty_ratio", "bad_cases": []}

    answer_keys = ("answer", "output", "response", "completion")
    input_keys = ("instruction", "question", "input", "prompt", "text")
    passed = 0
    bad_cases: list[dict] = []
    for row in rows:
        ans = next((str(row[k]) for k in answer_keys if row.get(k)), "")
        inp = next((str(row[k]) for k in input_keys if row.get(k)), "")
        if ans.strip():
            passed += 1
        else:
            bad_cases.append({
                "input": inp,
                "prediction": ans,
                "ground_truth": "",
                "error_type": "empty_answer",
            })
    return {
        "score": passed / len(rows),
        "metric_name": "non_empty_ratio",
        "bad_cases": bad_cases[:30],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="参考训练/评估脚本（契约演示）")
    parser.add_argument("--dataset", required=True, help="输入数据集 jsonl 路径")
    parser.add_argument("--output", required=True, help="输出目录（回写 results.json）")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_jsonl(dataset) if dataset.exists() else []
    results = evaluate(rows)

    (out_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[reference_train_eval] 评估 {len(rows)} 条，score={results['score']:.4f}")


if __name__ == "__main__":
    main()
