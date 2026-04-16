import argparse
import json
from pathlib import Path

from sft_stats import summarize_items


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
    parser = argparse.ArgumentParser(description="合并多个JSONL样本")
    parser.add_argument("--input-dir", required=True, help="样本目录")
    parser.add_argument("--pattern", default="sft_samples_*.jsonl", help="匹配模式")
    parser.add_argument("--output", required=True, help="合并输出JSONL路径")
    parser.add_argument("--stats", required=True, help="合并统计JSON路径")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    stats_path = Path(args.stats)

    sample_files = sorted(input_dir.glob(args.pattern))
    all_items: list[dict] = []
    for file_path in sample_files:
        all_items.extend(load_jsonl(file_path))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for item in all_items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    stats = summarize_items(all_items)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)

    print(f"合并样本数: {len(all_items)}")
    print(f"输出路径: {output_path}")


if __name__ == "__main__":
    main()
