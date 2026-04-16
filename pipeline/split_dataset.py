"""
将 llm_inputs JSONL 划分为训练集 / 评估集，固定随机种子以便复现（对应后续工作清单 D5）。
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="划分训练/评估 JSONL（固定 seed）")
    parser.add_argument("--input", required=True, help="输入 JSONL，如 output/llm_inputs_v4.jsonl")
    parser.add_argument("--train-out", required=True, help="训练集输出路径")
    parser.add_argument("--eval-out", required=True, help="评估集输出路径")
    parser.add_argument("--manifest", required=True, help="划分说明 JSON")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=None,
        help="评估集占比；若不指定则与 train_qwen 一致：min(200, max(25, n//5)) / n",
    )
    parser.add_argument("--eval-size", type=int, default=None, help="直接指定评估条数（覆盖 ratio）")
    args = parser.parse_args()

    inp = Path(args.input)
    samples = load_jsonl(inp)
    n = len(samples)
    if n < 4:
        raise SystemExit("样本过少，至少需要 4 条")

    rng = random.Random(args.seed)
    samples_shuffled = samples.copy()
    rng.shuffle(samples_shuffled)

    if args.eval_size is not None:
        eval_n = min(max(1, args.eval_size), n - 1)
    elif args.eval_ratio is not None:
        eval_n = max(1, min(n - 1, int(round(n * args.eval_ratio))))
    else:
        eval_cap = 200
        min_eval = 25
        eval_n = min(eval_cap, max(min_eval, n // 5))
        eval_n = min(eval_n, n - 1)

    train_size = n - eval_n
    train_samples = samples_shuffled[:train_size]
    eval_samples = samples_shuffled[train_size : train_size + eval_n]

    train_path = Path(args.train_out)
    eval_path = Path(args.eval_out)
    man_path = Path(args.manifest)
    for p in (train_path, eval_path, man_path):
        p.parent.mkdir(parents=True, exist_ok=True)

    with train_path.open("w", encoding="utf-8") as f:
        for row in train_samples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with eval_path.open("w", encoding="utf-8") as f:
        for row in eval_samples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "source": str(inp.resolve()),
        "total": n,
        "train_count": len(train_samples),
        "eval_count": len(eval_samples),
        "seed": args.seed,
        "note": "与 train_qwen2_5_7b_qlora_demo 一致：shuffle 全量后前 train_size 为训练集，紧接着 eval_n 为评估集",
    }
    with man_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"总样本: {n}，训练: {len(train_samples)}，评估: {len(eval_samples)}")
    print(f"训练集: {train_path}")
    print(f"评估集: {eval_path}")
    print(f"清单: {man_path}")


if __name__ == "__main__":
    main()
