import argparse
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def summarize_items(items: list[dict]) -> dict:
    fault_types = Counter()
    objects = Counter()
    root_causes = Counter()
    kpis = Counter()
    evidence_rows = []

    for item in items:
        output = item.get("output", {})
        fault_types[output.get("fault_type", "unknown")] += 1
        kpis[output.get("kpi", "unknown")] += 1
        root_causes[output.get("root_cause_component", "unknown")] += 1

        fault_event = item.get("input", {}).get("fault_event", {})
        objects[fault_event.get("object", "unknown")] += 1

        evidence = item.get("input", {}).get("evidence", {})
        platform = evidence.get("platform_metrics", {})
        trace = evidence.get("trace_metrics", {})
        business = evidence.get("business_metrics", {})
        rows = 0
        for group in (platform, trace, business):
            for stats in group.values():
                rows += int(stats.get("matched_rows", 0))
        evidence_rows.append(rows)

    avg_rows = sum(evidence_rows) / len(evidence_rows) if evidence_rows else 0
    return {
        "total_samples": len(items),
        "fault_type_counts": fault_types.most_common(),
        "object_counts": objects.most_common(),
        "root_cause_counts": root_causes.most_common(),
        "kpi_counts": kpis.most_common(),
        "evidence_rows": {
            "avg": round(avg_rows, 2),
            "min": min(evidence_rows) if evidence_rows else 0,
            "max": max(evidence_rows) if evidence_rows else 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="统计SFT样本分布")
    parser.add_argument("--input", required=True, help="输入JSONL路径")
    parser.add_argument("--output", required=True, help="输出统计JSON路径")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    items = load_jsonl(input_path)
    summary = summarize_items(items)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"样本数: {summary['total_samples']}")
    print(f"输出路径: {output_path}")


if __name__ == "__main__":
    main()
