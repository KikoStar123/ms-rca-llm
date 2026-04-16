import argparse
import json
from datetime import datetime
from pathlib import Path

from small_model_rootcause_topk import load_events, parse_time
from small_model_rootcause_topk import aggregate_counts, evaluate_topk


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_ROOT = BASE_DIR / "data" / "aiops2020"
FAILURE_CSV = DATA_ROOT / "故障整理（预赛）.csv"


def collect_dates() -> list[datetime]:
    dates = set()
    with FAILURE_CSV.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        header = handle.readline().strip().split(",")
        for line in handle:
            values = line.strip().split(",")
            if len(values) != len(header):
                continue
            row = dict(zip(header, values))
            log_time = parse_time(row.get("start_time") or row.get("log_time") or "")
            if not log_time:
                continue
            dates.add(datetime(log_time.year, log_time.month, log_time.day))
    return sorted(dates)


def main() -> None:
    parser = argparse.ArgumentParser(description="批量评估根因Top-K基线")
    parser.add_argument("--start-date", required=True, help="起始日期YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="结束日期YYYY-MM-DD")
    parser.add_argument("--window-before", type=int, default=5, help="窗口前分钟数")
    parser.add_argument("--window-after", type=int, default=5, help="窗口后分钟数")
    parser.add_argument("--k", default="1,3,5", help="Top-K列表")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--summary", required=True, help="汇总JSON路径")
    args = parser.parse_args()

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    ks = [int(k.strip()) for k in args.k.split(",") if k.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dates = [d for d in collect_dates() if start_date <= d <= end_date]
    summary = []
    for target_date in dates:
        date_str = target_date.strftime("%Y-%m-%d")
        day_dir = DATA_ROOT / f"day_{target_date:%Y_%m_%d}" / f"{target_date:%Y_%m_%d}"
        platform_dir = day_dir / "平台指标"
        trace_dir = day_dir / "调用链指标"
        if not platform_dir.exists() or not trace_dir.exists():
            print(f"[跳过] {date_str} 缺少数据目录")
            continue

        events = load_events(target_date, args.window_before, args.window_after)
        if not events:
            print(f"[跳过] {date_str} 无故障事件")
            continue

        platform_files = sorted(platform_dir.glob("*.csv"))
        trace_files = sorted(trace_dir.glob("*.csv"))
        platform_counts = aggregate_counts(events, platform_files, "timestamp", "cmdb_id")
        trace_counts = aggregate_counts(events, trace_files, "startTime", "cmdb_id")

        combined = []
        for p_counter, t_counter in zip(platform_counts, trace_counts):
            merged = p_counter.copy()
            merged.update(t_counter)
            combined.append(merged)

        result = evaluate_topk(events, combined, ks)
        report_path = output_dir / f"rootcause_topk_{target_date:%Y_%m_%d}_report.json"
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "date": date_str,
                    "total": result["total"],
                    "hit_rate": result["hit_rate"],
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )

        summary.append(
            {
                "date": date_str,
                "total": result["total"],
                "hit_rate": result["hit_rate"],
            }
        )
        print(f"[完成] {date_str} 样本数: {result['total']} {result['hit_rate']}")

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
