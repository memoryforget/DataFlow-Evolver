"""参考用外部训练/评估脚本——演示并文档化 DownstreamTrainerWrapper 的 I/O 契约。

契约：
    python scripts/reference_train_eval.py --dataset <ds.jsonl> --output <out_dir> \\
        [--mode dev|test] --dev-set <dev.jsonl> | --test-set <test.jsonl>
脚本必须在 <out_dir>/results.json 回写：
    {
      "score": float,
      "metric_name": str,
      "bad_cases": [{"input","prediction","ground_truth","error_type"}, ...]
    }

这里的"训练"是一个查表模型：把 --dataset 的 (问题 → 答案) 背下来，再到题集上逐题作答。
足够朴素以至于不需要 GPU，但它演示的**形状**与真实脚本一致——在合成数据上训练，在框架
指定的那份题集上评测，产出真实的 accuracy 与错题。

两条纪律照搬 README「脚本负责什么、不负责什么」：
  1. 题集整份评完，脚本不抽样、不截断——评哪些题是框架的决策（core/dev_set.py 物化 + manifest）；
  2. 拿不到题集就当场非零退出，绝不回退到任何"默认题库"。有退路的约定迟早会被走回去，
     而每一次都不报错：评的题数与 manifest 对不上，"所有节点评在同一批题上"就悄悄失效了。

dev 题集由框架物化后经 --dev-set 下传；终评题集是脚本自己的事，把 --test-set 写进 YAML 的
trainer.test_eval_cmd 即可，框架不插手。

把 evaluate() 换成你自己的训练+评估逻辑即可。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_INPUT_KEYS = ("instruction", "question", "input", "prompt", "text")
_ANSWER_KEYS = ("answer", "output", "response", "completion")


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


def _field(row: dict, keys: tuple[str, ...]) -> str:
    return next((str(row[k]) for k in keys if row.get(k)), "").strip()


def evaluate(train_rows: list[dict], eval_rows: list[dict], max_bad_cases: int = 30) -> dict:
    """在 train_rows 上"训练"一个查表模型，在 eval_rows 上逐题评测。

    eval_rows 原样评完：条数由框架决定，这里不做任何取舍。
    """
    memory = {
        _field(r, _INPUT_KEYS): _field(r, _ANSWER_KEYS)
        for r in train_rows
        if _field(r, _INPUT_KEYS) and _field(r, _ANSWER_KEYS)
    }

    correct = 0
    bad_cases: list[dict] = []
    for row in eval_rows:
        question = _field(row, _INPUT_KEYS)
        gold = _field(row, _ANSWER_KEYS)
        prediction = memory.get(question, "")
        if prediction and prediction == gold:
            correct += 1
        elif len(bad_cases) < max_bad_cases:
            bad_cases.append({
                "input": question,
                "prediction": prediction,
                "ground_truth": gold,
                "error_type": "no_answer" if not prediction else "wrong_answer",
            })

    return {
        "score": correct / len(eval_rows) if eval_rows else 0.0,
        "metric_name": "exact_match",
        "bad_cases": bad_cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="参考训练/评估脚本（契约演示）")
    parser.add_argument("--dataset", required=True, help="流水线产出的训练数据 jsonl")
    parser.add_argument("--output", required=True, help="输出目录（回写 results.json）")
    parser.add_argument("--mode", choices=["dev", "test"], default="dev",
                        help="dev=驱动 MCTS 奖励（框架自动调用）；test=终评")
    parser.add_argument("--dev-set", default="",
                        help="框架物化的 dev 题集（core/dev_set.py），--mode dev 必需")
    parser.add_argument("--test-set", default="",
                        help="终评题集，写在 YAML 的 trainer.test_eval_cmd 里，--mode test 必需")
    args = parser.parse_args()

    eval_path = args.dev_set if args.mode == "dev" else args.test_set
    if not eval_path:
        # 不猜。评哪些题不是脚本的决策，拿不到就是配置错了，当场暴露比闷头评一份别的题集好。
        flag = "--dev-set" if args.mode == "dev" else "--test-set"
        parser.error(
            f"--mode {args.mode} 需要 {flag}：本脚本只在给定题集上评测，"
            f"不抽样也没有内置默认题库。"
            + (
                "dev 题集由框架按 trainer.dev_set 物化后下传；留空 trainer.dev_set.source "
                "则不会物化。" if args.mode == "dev" else
                "终评题集由 trainer.test_eval_cmd 自行指定。"
            )
        )

    eval_file = Path(eval_path)
    if not eval_file.exists():
        raise SystemExit(f"题集文件不存在：{eval_file}")

    dataset = Path(args.dataset)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = _load_jsonl(dataset) if dataset.exists() else []
    eval_rows = _load_jsonl(eval_file)
    results = evaluate(train_rows, eval_rows)

    (out_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[reference_train_eval] mode={args.mode} 训练 {len(train_rows)} 条，"
        f"评测 {len(eval_rows)} 条，score={results['score']:.4f}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
