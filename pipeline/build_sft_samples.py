import argparse
import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_ROOT = BASE_DIR / "data" / "aiops2020"
DAY_DIR = DATA_ROOT / "day_2020_04_11" / "2020_04_11"
FAILURE_CSV = DATA_ROOT / "故障整理（预赛）.csv"
OUTPUT_PATH = BASE_DIR / "output" / "sft_samples_2020_04_11.jsonl"

WINDOW_BEFORE_MIN = 5
WINDOW_AFTER_MIN = 5
MAX_SUMMARY_ITEMS = 5


@dataclass
class FailureEvent:
    index: str
    obj: str
    fault_description: str
    kpi: str
    name: str
    container: str
    log_time: datetime
    duration: str


@dataclass
class EventContext:
    event: FailureEvent
    start: datetime
    end: datetime
    cmdb_targets: set[str]
    platform_summary: dict = field(default_factory=dict)
    trace_summary: dict = field(default_factory=dict)
    business_summary: dict = field(default_factory=dict)


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


def read_failures(target_date: datetime) -> list[FailureEvent]:
    events: list[FailureEvent] = []
    if not FAILURE_CSV.exists():
        raise FileNotFoundError(f"未找到故障清单: {FAILURE_CSV}")
    with FAILURE_CSV.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            log_time = parse_time(row.get("start_time") or row.get("log_time") or "")
            if not log_time:
                continue
            if log_time.date() != target_date.date():
                continue
            events.append(
                FailureEvent(
                    index=row.get("index", ""),
                    obj=row.get("object", ""),
                    fault_description=row.get("fault_desrcibtion", ""),
                    kpi=row.get("kpi", ""),
                    name=row.get("name", ""),
                    container=row.get("container", ""),
                    log_time=log_time,
                    duration=row.get("duration", ""),
                )
            )
    return events


def in_window(ts: datetime, start: datetime, end: datetime) -> bool:
    return start <= ts <= end


def finalize_metric_summary(summary: dict, matched_rows: int, max_items: int) -> dict:
    items = []
    for name, stats in summary.items():
        count = int(stats["count"])
        avg = stats["sum"] / count if count else 0.0
        items.append(
            {
                "metric": name,
                "count": count,
                "avg": round(avg, 6),
                "min": round(float(stats["min"]), 6),
                "max": round(float(stats["max"]), 6),
            }
        )
    items.sort(key=lambda x: x["count"], reverse=True)
    return {"matched_rows": matched_rows, "metrics": items[:max_items]}


def finalize_trace_summary(summary: dict, matched_rows: int, max_items: int) -> dict:
    items = []
    for service, stats in summary.items():
        count = int(stats["count"])
        avg = stats["sum"] / count if count else 0.0
        success_rate = stats["success"] / count if count else 0.0
        items.append(
            {
                "service": service,
                "count": count,
                "avg_elapsed": round(avg, 6),
                "success_rate": round(success_rate, 6),
            }
        )
    items.sort(key=lambda x: x["count"], reverse=True)
    return {"matched_rows": matched_rows, "services": items[:max_items]}


def finalize_business_summary(summary: dict, matched_rows: int, max_items: int) -> dict:
    items = []
    for service, stats in summary.items():
        count = int(stats["count"])
        avg_time = stats["avg_time_sum"] / count if count else 0.0
        avg_rate = stats["succee_rate_sum"] / count if count else 0.0
        items.append(
            {
                "service": service,
                "count": count,
                "avg_time": round(avg_time, 6),
                "avg_succee_rate": round(avg_rate, 6),
                "total_num": int(stats["num"]),
            }
        )
    items.sort(key=lambda x: x["count"], reverse=True)
    return {"matched_rows": matched_rows, "services": items[:max_items]}


def aggregate_platform_files(
    file_paths: list[Path], events: list[EventContext], max_items: int
) -> None:
    for file_path in file_paths:
        summaries = [defaultdict(lambda: {"count": 0, "sum": 0.0, "min": 0.0, "max": 0.0}) for _ in events]
        matched_rows = [0 for _ in events]
        with file_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                ts_raw = row.get("timestamp", "")
                if not ts_raw:
                    continue
                ts = datetime.fromtimestamp(int(ts_raw) / 1000)
                cmdb_id = row.get("cmdb_id", "")
                name = row.get("name", "unknown")
                value_raw = row.get("value", "")
                try:
                    value = float(value_raw)
                except ValueError:
                    continue
                for idx, ctx in enumerate(events):
                    if not in_window(ts, ctx.start, ctx.end):
                        continue
                    if ctx.cmdb_targets and cmdb_id not in ctx.cmdb_targets:
                        continue
                    matched_rows[idx] += 1
                    item = summaries[idx][name]
                    if item["count"] == 0:
                        item["min"] = value
                        item["max"] = value
                    item["count"] += 1
                    item["sum"] += value
                    item["min"] = min(item["min"], value)
                    item["max"] = max(item["max"], value)
        for idx, ctx in enumerate(events):
            ctx.platform_summary[file_path.name] = finalize_metric_summary(
                summaries[idx], matched_rows[idx], max_items
            )


def aggregate_trace_files(
    file_paths: list[Path], events: list[EventContext], max_items: int
) -> None:
    for file_path in file_paths:
        summaries = [defaultdict(lambda: {"count": 0, "sum": 0.0, "success": 0}) for _ in events]
        matched_rows = [0 for _ in events]
        with file_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                ts_raw = row.get("startTime", "")
                if not ts_raw:
                    continue
                ts = datetime.fromtimestamp(int(ts_raw) / 1000)
                cmdb_id = row.get("cmdb_id", "")
                service = row.get("serviceName", "unknown")
                elapsed_raw = row.get("elapsedTime", "0")
                success_raw = row.get("success", "")
                try:
                    elapsed = float(elapsed_raw)
                except ValueError:
                    elapsed = 0.0
                success = str(success_raw).lower() == "true"
                for idx, ctx in enumerate(events):
                    if not in_window(ts, ctx.start, ctx.end):
                        continue
                    if ctx.cmdb_targets and cmdb_id not in ctx.cmdb_targets:
                        continue
                    matched_rows[idx] += 1
                    item = summaries[idx][service]
                    item["count"] += 1
                    item["sum"] += elapsed
                    if success:
                        item["success"] += 1
        for idx, ctx in enumerate(events):
            ctx.trace_summary[file_path.name] = finalize_trace_summary(
                summaries[idx], matched_rows[idx], max_items
            )


def aggregate_business_files(
    file_paths: list[Path], events: list[EventContext], max_items: int
) -> None:
    for file_path in file_paths:
        summaries = [defaultdict(lambda: {"count": 0, "avg_time_sum": 0.0, "succee_rate_sum": 0.0, "num": 0}) for _ in events]
        matched_rows = [0 for _ in events]
        with file_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                ts_raw = row.get("startTime", "")
                if not ts_raw:
                    continue
                ts = datetime.fromtimestamp(int(ts_raw) / 1000)
                service = row.get("serviceName", "unknown")
                avg_time = float(row.get("avg_time") or 0)
                succee_rate = float(row.get("succee_rate") or 0)
                num = int(float(row.get("num") or 0))
                for idx, ctx in enumerate(events):
                    if not in_window(ts, ctx.start, ctx.end):
                        continue
                    matched_rows[idx] += 1
                    item = summaries[idx][service]
                    item["count"] += 1
                    item["avg_time_sum"] += avg_time
                    item["succee_rate_sum"] += succee_rate
                    item["num"] += num
        for idx, ctx in enumerate(events):
            ctx.business_summary[file_path.name] = finalize_business_summary(
                summaries[idx], matched_rows[idx], max_items
            )


def build_sft_samples(
    target_date: datetime,
    window_before: int,
    window_after: int,
    max_items: int,
    limit: int | None,
) -> list[dict]:
    failures = read_failures(target_date)
    if limit:
        failures = failures[:limit]
    day_dir = DATA_ROOT / f"day_{target_date:%Y_%m_%d}" / f"{target_date:%Y_%m_%d}"
    platform_dir = day_dir / "平台指标"
    trace_dir = day_dir / "调用链指标"
    business_dir = day_dir / "业务指标"

    platform_files = sorted(platform_dir.glob("*.csv"))
    trace_files = sorted(trace_dir.glob("*.csv"))
    business_files = sorted(business_dir.glob("*.csv"))

    samples: list[dict] = []
    event_contexts: list[EventContext] = []
    for event in failures:
        start = event.log_time - timedelta(minutes=window_before)
        end = event.log_time + timedelta(minutes=window_after)
        cmdb_targets = {event.name, event.container}
        cmdb_targets = {item for item in cmdb_targets if item}
        event_contexts.append(
            EventContext(
                event=event,
                start=start,
                end=end,
                cmdb_targets=cmdb_targets,
            )
        )

    aggregate_platform_files(platform_files, event_contexts, max_items)
    aggregate_trace_files(trace_files, event_contexts, max_items)
    aggregate_business_files(business_files, event_contexts, max_items)

    for ctx in event_contexts:
        evidence = {
            "platform_metrics": ctx.platform_summary,
            "trace_metrics": ctx.trace_summary,
            "business_metrics": ctx.business_summary,
        }
        fault_event = {
            "index": ctx.event.index,
            "object": ctx.event.obj,
            "fault_description": ctx.event.fault_description,
            "kpi": ctx.event.kpi,
            "name": ctx.event.name,
            "container": ctx.event.container,
            "log_time": ctx.event.log_time.isoformat(sep=" "),
            "duration": ctx.event.duration,
            "window_start": ctx.start.isoformat(sep=" "),
            "window_end": ctx.end.isoformat(sep=" "),
        }

        samples.append(
            {
                "instruction": "基于观测数据定位故障根因并给出故障类型。",
                "input": {
                    "fault_event": fault_event,
                    "evidence": evidence,
                },
                "output": {
                    "root_cause_component": ctx.event.name,
                    "fault_type": ctx.event.fault_description,
                    "kpi": ctx.event.kpi,
                    "related_container": ctx.event.container,
                },
            }
        )
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="生成AIOps SFT样本")
    parser.add_argument("--date", default="2020-04-11", help="日期，格式YYYY-MM-DD")
    parser.add_argument("--output", default=str(OUTPUT_PATH), help="输出JSONL路径")
    parser.add_argument("--window-before", type=int, default=WINDOW_BEFORE_MIN, help="窗口前分钟数")
    parser.add_argument("--window-after", type=int, default=WINDOW_AFTER_MIN, help="窗口后分钟数")
    parser.add_argument("--max-items", type=int, default=MAX_SUMMARY_ITEMS, help="每类摘要条数上限")
    parser.add_argument("--limit", type=int, default=0, help="限制故障事件条数，0为不限制")
    args = parser.parse_args()

    target_date = datetime.strptime(args.date, "%Y-%m-%d")
    samples = build_sft_samples(
        target_date=target_date,
        window_before=args.window_before,
        window_after=args.window_after,
        max_items=args.max_items,
        limit=args.limit or None,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for item in samples:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"生成样本数: {len(samples)}")
    print(f"输出路径: {output_path}")


if __name__ == "__main__":
    main()
