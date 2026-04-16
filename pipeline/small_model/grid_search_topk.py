import argparse
import json
from pathlib import Path

from batch_rootcause_weighted_topk import main as run_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="网格搜索Top-K加权参数")
    parser.add_argument("--start-date", required=True, help="起始日期YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="结束日期YYYY-MM-DD")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_weights = [1.0, 1.5, 2.0]
    anomaly_weights = [0.5, 0.8, 1.0]

    results = []
    for tw in trace_weights:
        for aw in anomaly_weights:
            summary_path = output_dir / f"grid_summary_tw{tw}_aw{aw}.json"
            run_args = [
                "batch_rootcause_weighted_topk.py",
                "--start-date",
                args.start_date,
                "--end-date",
                args.end_date,
                "--output-dir",
                str(output_dir),
                "--summary",
                str(summary_path),
                "--platform-weight",
                "1.0",
                "--trace-weight",
                str(tw),
                "--anomaly-weight",
                str(aw),
                "--anomaly-once",
            ]
            # 复用批处理主函数
            import sys

            sys.argv = ["batch_rootcause_weighted_topk.py"] + run_args[1:]
            run_batch()

            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            avg = _avg_hit(summary)
            results.append(
                {
                    "trace_weight": tw,
                    "anomaly_weight": aw,
                    "avg": avg,
                }
            )

    best = sorted(results, key=lambda x: (x["avg"].get("hit@3", 0), x["avg"].get("hit@1", 0)), reverse=True)
    out_path = output_dir / "grid_search_results.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump({"results": results, "best": best[:3]}, handle, ensure_ascii=False, indent=2)
    print(f"网格搜索完成: {out_path}")


def _avg_hit(summary: list[dict]) -> dict:
    totals = {}
    counts = {}
    for item in summary:
        hit_rate = item.get("hit_rate", {})
        for key, value in hit_rate.items():
            totals[key] = totals.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    return {key: round(totals[key] / counts[key], 4) for key in totals}


if __name__ == "__main__":
    main()
