"""
评估集上金标 root_cause_component 相对 top_candidates 的覆盖率（与训练脚本里 Top-3/Top-5 指标含义一致）。

用法：
  python pipeline/analyze_eval_candidate_coverage.py
  python pipeline/analyze_eval_candidate_coverage.py --jsonl output/eval_split.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main() -> None:
    p = argparse.ArgumentParser(description="金标根因在候选列表中的位置与覆盖率")
    p.add_argument(
        "--jsonl",
        default="output/eval_split.jsonl",
        help="SFT 格式 JSONL（含 input.top_candidates 与 output.root_cause_component）",
    )
    p.add_argument(
        "--out-json",
        default="",
        help="可选：将统计结果写入 JSON 文件",
    )
    args = p.parse_args()

    path = Path(args.jsonl)
    if not path.is_file():
        # 相对仓库根
        alt = _REPO / args.jsonl
        if alt.is_file():
            path = alt
        else:
            print(f"错误: 找不到文件 {args.jsonl}", file=sys.stderr)
            sys.exit(1)

    total = 0
    in3 = in5 = in10 = in_all = 0
    missing_gold = 0
    not_in_list = 0
    ranks: list[int] = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out = row.get("output") or {}
            inp = row.get("input") or {}
            gold = (out.get("root_cause_component") or "").strip()
            cands = inp.get("top_candidates") or []
            if not isinstance(cands, list):
                cands = []
            total += 1
            if not gold:
                missing_gold += 1
                continue
            try:
                idx = cands.index(gold)
            except ValueError:
                not_in_list += 1
                ranks.append(-1)
                continue
            ranks.append(idx)
            in_all += 1
            if idx < 3:
                in3 += 1
            if idx < 5:
                in5 += 1
            if idx < 10:
                in10 += 1

    def rate(x: int) -> float:
        return round(x / total, 6) if total else 0.0

    stats = {
        "jsonl": str(path.resolve()),
        "total_lines": total,
        "gold_missing": missing_gold,
        "gold_not_in_candidates": not_in_list,
        "gold_in_candidates_anywhere": in_all,
        "coverage_top3": rate(in3),
        "coverage_top5": rate(in5),
        "coverage_top10": rate(in10),
        "interpretation": (
            "coverage_topK 表示：金标在候选列表第 K 位之内（0-based 前 K 个）的样本占比。"
            " 若 gold_not_in_candidates 较高，应优先改进上游候选生成；否则以模型在候选内辨别为主。"
        ),
    }
    if ranks:
        valid = [r for r in ranks if r >= 0]
        stats["mean_rank_when_in_list"] = round(sum(valid) / len(valid), 4) if valid else None
        stats["median_rank_when_in_list"] = sorted(valid)[len(valid) // 2] if valid else None

    print(json.dumps(stats, ensure_ascii=False, indent=2))

    if args.out_json:
        out_p = Path(args.out_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已写入 {out_p.resolve()}", file=sys.stderr)


if __name__ == "__main__":
    main()
