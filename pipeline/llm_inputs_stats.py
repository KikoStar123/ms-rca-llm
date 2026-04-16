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


def main() -> None:
    parser = argparse.ArgumentParser(description="统计大模型输入样本")
    parser.add_argument("--input", required=True, help="输入JSONL路径")
    parser.add_argument("--output", required=True, help="输出统计JSON路径")
    args = parser.parse_args()

    items = load_jsonl(Path(args.input))
    fault_types = Counter()
    root_causes = Counter()
    topk_sizes = Counter()

    for item in items:
        output = item.get("output", {})
        fault_types[output.get("fault_type", "unknown")] += 1
        root_causes[output.get("root_cause_component", "unknown")] += 1
        top_candidates = item.get("input", {}).get("top_candidates", [])
        topk_sizes[len(top_candidates)] += 1

    summary = {
        "total_samples": len(items),
        "fault_type_counts": fault_types.most_common(),
        "root_cause_counts": root_causes.most_common(),
        "topk_size_counts": sorted(topk_sizes.items()),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"样本数: {summary['total_samples']}")
    print(f"输出路径: {output_path}")


if __name__ == "__main__":
    main()
