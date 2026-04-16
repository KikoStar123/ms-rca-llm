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


def aggregate_platform_counts(
    events: list[EventWindow],
    files: list[Path],
    weight: float,
    anomaly_weight: float,
    anomaly_once: bool,
) -> list[Counter]:
    counters = [Counter() for _ in events]
    stats = [{} for _ in events]
    for file_path in files:
        with file_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                ts_raw = row.get("timestamp", "")
                if not ts_raw:
                    continue
                try:
                    ts = datetime.fromtimestamp(int(ts_raw) / 1000)
                except ValueError:
                    continue
                cmdb_id = row.get("cmdb_id", "")
                if not cmdb_id:
                    continue
                value_raw = row.get("value", "")
                try:
                    value = float(value_raw)
                except ValueError:
                    continue
                for idx, event in enumerate(events):
                    if in_window(ts, event.window_start, event.window_end):
                        counters[idx][cmdb_id] += weight
                        entry = stats[idx].setdefault(cmdb_id, {"min": value, "max": value, "sum": 0.0, "count": 0})
                        entry["min"] = min(entry["min"], value)
                        entry["max"] = max(entry["max"], value)
                        entry["sum"] += value
                        entry["count"] += 1
        if not anomaly_once:
            for idx, event in enumerate(events):
                for cmdb_id, entry in stats[idx].items():
                    if entry["count"] == 0:
                        continue
                    mean = entry["sum"] / entry["count"]
                    span = entry["max"] - entry["min"]
                    score = span / (abs(mean) + 1e-6)
                    counters[idx][cmdb_id] += anomaly_weight * min(score, 10.0)
            stats = [{} for _ in events]
    if anomaly_once:
        for idx, event in enumerate(events):
            for cmdb_id, entry in stats[idx].items():
                if entry["count"] == 0:
                    continue
                mean = entry["sum"] / entry["count"]
                span = entry["max"] - entry["min"]
                score = span / (abs(mean) + 1e-6)
                counters[idx][cmdb_id] += anomaly_weight * min(score, 10.0)
    return counters


def aggregate_trace_counts(
    events: list[EventWindow],
    files: list[Path],
    weight: float,
    elapsed_cap: float,
    failure_penalty: float,
    slow_threshold: float,
    rate_mode: bool,
) -> list[Counter]:
    counters = [Counter() for _ in events]
    stats = [dict() for _ in events]
    for file_path in files:
        with file_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                ts_raw = row.get("startTime", "")
                if not ts_raw:
                    continue
                try:
                    ts = datetime.fromtimestamp(int(ts_raw) / 1000)
                except ValueError:
                    continue
                cmdb_id = row.get("cmdb_id", "")
                if not cmdb_id:
                    continue
                elapsed_raw = row.get("elapsedTime", "0")
                success_raw = row.get("success", "")
                try:
                    elapsed = float(elapsed_raw)
                except ValueError:
                    elapsed = 0.0
                success = str(success_raw).lower() == "true"
                for idx, event in enumerate(events):
                    if not in_window(ts, event.window_start, event.window_end):
                        continue
                    if rate_mode:
                        entry = stats[idx].setdefault(cmdb_id, {"total": 0, "fail": 0, "slow": 0})
                        entry["total"] += 1
                        if not success:
                            entry["fail"] += 1
                        if elapsed / 1000.0 >= slow_threshold:
                            entry["slow"] += 1
                    else:
                        extra = min(elapsed / 1000.0, elapsed_cap)
                        if not success:
                            extra += failure_penalty
                        score = weight + extra
                        counters[idx][cmdb_id] += score
        if rate_mode:
            for idx, event in enumerate(events):
                for cmdb_id, entry in stats[idx].items():
                    if entry["total"] == 0:
                        continue
                    fail_rate = entry["fail"] / entry["total"]
                    slow_rate = entry["slow"] / entry["total"]
                    score = weight + failure_penalty * fail_rate + elapsed_cap * slow_rate
                    counters[idx][cmdb_id] += score
            stats = [dict() for _ in events]
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
    parser = argparse.ArgumentParser(description="小模型基线：加权根因Top-K候选")
    parser.add_argument("--date", required=True, help="日期YYYY-MM-DD")
    parser.add_argument("--window-before", type=int, default=5, help="窗口前分钟数")
    parser.add_argument("--window-after", type=int, default=5, help="窗口后分钟数")
    parser.add_argument("--k", default="1,3,5", help="Top-K列表")
    parser.add_argument("--output", required=True, help="预测输出JSONL路径")
    parser.add_argument("--report", required=True, help="评估报告JSON路径")
    parser.add_argument("--platform-weight", type=float, default=1.0, help="平台指标权重")
    parser.add_argument("--trace-weight", type=float, default=2.0, help="调用链权重")
    parser.add_argument("--anomaly-weight", type=float, default=0.5, help="平台异常强度权重")
    parser.add_argument("--anomaly-once", action="store_true", help="异常强度只累计一次")
    parser.add_argument("--elapsed-cap", type=float, default=5.0, help="调用链耗时上限(秒)")
    parser.add_argument("--failure-penalty", type=float, default=5.0, help="失败惩罚")
    parser.add_argument("--slow-threshold", type=float, default=1.0, help="慢调用阈值(秒)")
    parser.add_argument("--rate-mode", action="store_true", help="按失败率/慢调用率计分")
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

    platform_counts = aggregate_platform_counts(
        events, platform_files, args.platform_weight, args.anomaly_weight, args.anomaly_once
    )
    trace_counts = aggregate_trace_counts(
        events,
        trace_files,
        args.trace_weight,
        args.elapsed_cap,
        args.failure_penalty,
        args.slow_threshold,
        args.rate_mode,
    )

    combined = []
    for p_counter, t_counter in zip(platform_counts, trace_counts):
        merged = p_counter.copy()
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
        "weights": {
            "platform": args.platform_weight,
            "trace": args.trace_weight,
            "anomaly": args.anomaly_weight,
            "anomaly_once": args.anomaly_once,
            "elapsed_cap": args.elapsed_cap,
            "failure_penalty": args.failure_penalty,
            "slow_threshold": args.slow_threshold,
            "rate_mode": args.rate_mode,
        },
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
