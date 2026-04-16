import argparse
import json
from pathlib import Path


def load_summary(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def avg_hit(summary: list[dict]) -> dict:
    totals = {}
    counts = {}
    for item in summary:
        hit_rate = item.get("hit_rate", {})
        for key, value in hit_rate.items():
            totals[key] = totals.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    return {key: round(totals[key] / counts[key], 4) for key in totals}


def main() -> None:
    parser = argparse.ArgumentParser(description="对比两套Top-K汇总")
    parser.add_argument("--base", required=True, help="基线汇总JSON")
    parser.add_argument("--weighted", required=True, help="加权汇总JSON")
    parser.add_argument("--output", required=True, help="对比输出JSON")
    args = parser.parse_args()

    base = load_summary(Path(args.base))
    weighted = load_summary(Path(args.weighted))

    base_avg = avg_hit(base)
    weighted_avg = avg_hit(weighted)
    delta = {k: round(weighted_avg.get(k, 0.0) - base_avg.get(k, 0.0), 4) for k in base_avg}

    result = {
        "base_avg": base_avg,
        "weighted_avg": weighted_avg,
        "delta": delta,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)

    print("对比完成:", output_path)


if __name__ == "__main__":
    main()
