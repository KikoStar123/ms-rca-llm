"""
对「模型预测 JSONL」离线计算与 train_qwen2_5_7b_qlora_demo.evaluate_model 一致的指标。
输入格式：每行 JSON，含 prediction（对象或 null）、label（对象）；可选 raw_text。
合并 gold 时需含 input，以便与训练脚本一致应用 prediction_repair（见 pipeline/prediction_repair.py）。
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from json_parse_utils import extract_json, sanitize_generation
from prediction_repair import apply_eval_repair

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_GOLD = _REPO_ROOT / "output" / "eval_split.jsonl"


def normalize_string(s: str) -> str:
    if not s:
        return s
    return re.sub(r"\s+", " ", s.strip())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_top_candidates(row: dict[str, Any]) -> list[str]:
    inp = row.get("input") or row.get("sample", {})
    if isinstance(inp, dict):
        tc = inp.get("top_candidates") or []
        if isinstance(tc, list):
            return tc
    return []


def ensure_prediction(row: dict[str, Any]) -> dict[str, Any] | None:
    if row.get("prediction") is not None:
        pred = row["prediction"]
        return pred if isinstance(pred, dict) else None
    raw = row.get("raw_text") or row.get("generation") or ""
    if raw:
        return extract_json(sanitize_generation(str(raw)))
    return None


def metrics_for(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    total = len(rows)
    parsed = 0
    root_hit = root_top3 = root_top5 = 0
    type_hit = kpi_hit = container_hit = full_match = 0

    for row in rows:
        label = row.get("label") or row.get("gold") or {}
        if not isinstance(label, dict):
            label = {}
        top_candidates = get_top_candidates(row)
        payload = ensure_prediction(row)
        if payload is None:
            continue
        payload = apply_eval_repair(row, payload)
        parsed += 1

        lr = label.get("root_cause_component", "")
        lt = normalize_string(str(label.get("fault_type", "")))
        lk = label.get("kpi", "")
        lc = label.get("related_container", "")

        pr = payload.get("root_cause_component", "")
        pt = normalize_string(str(payload.get("fault_type", "")))
        pk = payload.get("kpi", "")
        pc = payload.get("related_container", "")

        if pr == lr:
            root_hit += 1
        if lr and lr in top_candidates[:3]:
            root_top3 += 1
        if lr and lr in top_candidates[:5]:
            root_top5 += 1
        if pt == lt:
            type_hit += 1
        if pk == lk:
            kpi_hit += 1
        if pc == lc:
            container_hit += 1
        if pr == lr and pt == lt and pk == lk and pc == lc:
            full_match += 1

    def rate(x: int) -> float:
        return round(x / total, 6) if total else 0.0

    return {
        "total": total,
        "parsed": parsed,
        "parse_rate": rate(parsed),
        "root_acc_top1": rate(root_hit),
        "root_acc_top3": rate(root_top3),
        "root_acc_top5": rate(root_top5),
        "fault_type_acc": rate(type_hit),
        "kpi_acc": rate(kpi_hit),
        "related_container_acc": rate(container_hit),
        "full_match_acc": rate(full_match),
    }


def merge_gold_predictions(gold_path: Path, pred_path: Path) -> list[dict[str, Any]]:
    """按行号对齐合并：用于 pred 文件只有 raw_text 的情况。"""
    gold = load_jsonl(gold_path)
    pred = load_jsonl(pred_path)
    if len(gold) != len(pred):
        raise ValueError(f"行数不一致: gold={len(gold)} pred={len(pred)}")
    merged = []
    for g, p in zip(gold, pred):
        merged.append(
            {
                "input": g.get("input", {}),
                "label": g.get("output", {}),
                "prediction": p.get("prediction"),
                "raw_text": p.get("raw_text"),
            }
        )
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="离线评估 RCA 预测 JSONL")
    parser.add_argument("--pred", required=True, help="预测结果 JSONL（含 prediction 或 raw_text）")
    parser.add_argument(
        "--gold",
        default="",
        help="金标准 JSONL（与训练数据同格式，含 input.top_candidates）；缺省且存在 output/eval_split.jsonl 时自动使用",
    )
    parser.add_argument("--output", default="", help="可选：写出指标 JSON")
    args = parser.parse_args()

    pred_path = Path(args.pred)
    gold_path = Path(args.gold) if args.gold.strip() else None
    if gold_path is None and _DEFAULT_GOLD.is_file():
        gold_path = _DEFAULT_GOLD
        print(f"使用默认金标准: {gold_path}")

    if gold_path is not None:
        rows = merge_gold_predictions(gold_path, pred_path)
    else:
        rows = load_jsonl(pred_path)

    m = metrics_for(rows)
    for k, v in m.items():
        print(f"{k}: {v}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(m, f, ensure_ascii=False, indent=2)
        print(f"已写入: {out}")


if __name__ == "__main__":
    main()
