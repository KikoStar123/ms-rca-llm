import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_ROOT = BASE_DIR / "data" / "aiops2020"
FAILURE_CSV = DATA_ROOT / "故障整理（预赛）.csv"


@dataclass
class EventWindow:
    index: str
    log_time: datetime
    window_start: datetime
    window_end: datetime
    root_cause: str


def parse_time(value: str) -> datetime | None:
    if not value:
        return None
    value = value.strip()
    for fmt in ("%Y/%m/%d %H:%M", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def load_events(target_date: datetime, window_before: int, window_after: int) -> list[EventWindow]:
    events: list[EventWindow] = []
    with FAILURE_CSV.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            log_time = parse_time(row.get("start_time") or row.get("log_time") or "")
            if not log_time or log_time.date() != target_date.date():
                continue
            window_start = log_time - timedelta(minutes=window_before)
            window_end = log_time + timedelta(minutes=window_after)
            root_cause = row.get("name", "")
            events.append(
                EventWindow(
                    index=row.get("index", ""),
                    log_time=log_time,
                    window_start=window_start,
                    window_end=window_end,
                    root_cause=root_cause,
                )
            )
    return events


def in_window(ts: datetime, start: datetime, end: datetime) -> bool:
    return start <= ts <= end


def aggregate_counts(events: list[EventWindow], files: list[Path], ts_field: str, cmdb_field: str) -> list[Counter]:
    counters = [Counter() for _ in events]
    for file_path in files:
        with file_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                ts_raw = row.get(ts_field, "")
                if not ts_raw:
                    continue
                try:
                    ts = datetime.fromtimestamp(int(ts_raw) / 1000)
                except ValueError:
                    continue
                cmdb_id = row.get(cmdb_field, "")
                if not cmdb_id:
                    continue
                for idx, event in enumerate(events):
                    if in_window(ts, event.window_start, event.window_end):
                        counters[idx][cmdb_id] += 1
    return counters


def evaluate_topk(events: list[EventWindow], combined: list[Counter], ks: list[int]) -> dict:
    hits = {k: 0 for k in ks}
    total = len(events)
    predictions = []
    for event, counter in zip(events, combined):
        ranked = [cmdb for cmdb, _ in counter.most_common()]
        for k in ks:
            if event.root_cause in ranked[:k]:
                hits[k] += 1
        predictions.append(
            {
                "event_index": event.index,
                "log_time": event.log_time.isoformat(sep=" "),
                "root_cause": event.root_cause,
                "top_candidates": ranked[: max(ks)],
            }
        )
    hit_rate = {f"hit@{k}": round(hits[k] / total, 4) if total else 0.0 for k in ks}
    return {"total": total, "hit": hits, "hit_rate": hit_rate, "predictions": predictions}


def main() -> None:
    parser = argparse.ArgumentParser(description="小模型基线：根因Top-K候选")
    parser.add_argument("--date", required=True, help="日期YYYY-MM-DD")
    parser.add_argument("--window-before", type=int, default=5, help="窗口前分钟数")
    parser.add_argument("--window-after", type=int, default=5, help="窗口后分钟数")
    parser.add_argument("--k", default="1,3,5", help="Top-K列表")
    parser.add_argument("--output", required=True, help="预测输出JSONL路径")
    parser.add_argument("--report", required=True, help="评估报告JSON路径")
    args = parser.parse_args()

    target_date = datetime.strptime(args.date, "%Y-%m-%d")
    ks = [int(k.strip()) for k in args.k.split(",") if k.strip()]

    day_dir = DATA_ROOT / f"day_{target_date:%Y_%m_%d}" / f"{target_date:%Y_%m_%d}"
    platform_dir = day_dir / "平台指标"
    trace_dir = day_dir / "调用链指标"

    events = load_events(target_date, args.window_before, args.window_after)
    if not events:
        raise ValueError(f"指定日期没有故障事件: {args.date}")

    platform_files = sorted(platform_dir.glob("*.csv"))
    trace_files = sorted(trace_dir.glob("*.csv"))

    platform_counts = aggregate_counts(events, platform_files, "timestamp", "cmdb_id")
    trace_counts = aggregate_counts(events, trace_files, "startTime", "cmdb_id")

    combined = []
    for p_counter, t_counter in zip(platform_counts, trace_counts):
        merged = Counter()
        merged.update(p_counter)
        merged.update(t_counter)
        combined.append(merged)

    result = evaluate_topk(events, combined, ks)

    output_path = Path(args.output)
    report_path = Path(args.report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in result["predictions"]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "date": args.date,
        "total": result["total"],
        "hit_rate": result["hit_rate"],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(f"日期: {args.date}")
    print(f"样本数: {result['total']}")
    for k, rate in result["hit_rate"].items():
        print(f"{k}: {rate}")
    print(f"输出路径: {output_path}")
    print(f"报告路径: {report_path}")


if __name__ == "__main__":
    main()
