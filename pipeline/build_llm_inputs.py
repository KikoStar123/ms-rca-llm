import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from small_model_rootcause_weighted_topk import (
    aggregate_platform_counts,
    aggregate_trace_counts,
    evaluate_topk,
    load_events,
)


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_ROOT = BASE_DIR / "data" / "aiops2020"


def load_jsonl(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def compute_topk(
    date: datetime, window_before: int, window_after: int, weights: dict, ks: list[int]
) -> list[dict]:
    day_dir = DATA_ROOT / f"day_{date:%Y_%m_%d}" / f"{date:%Y_%m_%d}"
    platform_dir = day_dir / "平台指标"
    trace_dir = day_dir / "调用链指标"
    events = load_events(date, window_before, window_after)
    platform_files = sorted(platform_dir.glob("*.csv"))
    trace_files = sorted(trace_dir.glob("*.csv"))

    platform_counts = aggregate_platform_counts(
        events,
        platform_files,
        weights["platform"],
        weights["anomaly"],
        weights["anomaly_once"],
    )
    trace_counts = aggregate_trace_counts(
        events,
        trace_files,
        weights["trace"],
        weights["elapsed_cap"],
        weights["failure_penalty"],
        weights["slow_threshold"],
        weights["rate_mode"],
    )

    combined = []
    for p_counter, t_counter in zip(platform_counts, trace_counts):
        merged = p_counter.copy()
        merged.update(t_counter)
        combined.append(merged)
    result = evaluate_topk(events, combined, ks)
    return result["predictions"]


def compute_one_key(payload: tuple[tuple[str, int, int], dict]) -> tuple[tuple[str, int, int], dict]:
    key, weights = payload
    date_str, w_before, w_after = key
    event_date = datetime.strptime(date_str, "%Y-%m-%d")
    predictions = compute_topk(event_date, w_before, w_after, weights, [1, 3, 5])
    pred_map = {item["event_index"]: item["top_candidates"] for item in predictions}
    return key, pred_map


def parse_iso_time(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def compute_window_minutes(
    log_time: datetime | None,
    window_start: datetime | None,
    window_end: datetime | None,
    default_before: int,
    default_after: int,
) -> tuple[int, int]:
    before = default_before
    after = default_after
    if log_time and window_start:
        delta = (log_time - window_start).total_seconds() / 60
        if delta >= 0:
            before = int(round(delta))
    if log_time and window_end:
        delta = (window_end - log_time).total_seconds() / 60
        if delta >= 0:
            after = int(round(delta))
    return before, after


def main() -> None:
    parser = argparse.ArgumentParser(description="构建大模型输入样本")
    parser.add_argument("--input-dir", required=True, help="SFT样本目录")
    parser.add_argument("--output", required=True, help="输出JSONL路径")
    parser.add_argument("--pattern", default="sft_samples_*.jsonl", help="样本匹配模式")
    parser.add_argument("--window-before", type=int, default=5, help="窗口前分钟数")
    parser.add_argument("--window-after", type=int, default=5, help="窗口后分钟数")
    parser.add_argument("--platform-weight", type=float, default=1.0, help="平台指标权重")
    parser.add_argument("--trace-weight", type=float, default=1.5, help="调用链权重")
    parser.add_argument("--anomaly-weight", type=float, default=0.8, help="平台异常强度权重")
    parser.add_argument("--anomaly-once", action="store_true", help="异常强度只累计一次")
    parser.add_argument("--elapsed-cap", type=float, default=5.0, help="调用链耗时上限(秒)")
    parser.add_argument("--failure-penalty", type=float, default=5.0, help="失败惩罚")
    parser.add_argument("--slow-threshold", type=float, default=1.0, help="慢调用阈值(秒)")
    parser.add_argument("--rate-mode", action="store_true", help="按失败率/慢调用率计分")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Top-K 计算的并行线程数，默认取 CPU 核心数",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    sample_files = sorted(
        path for path in input_dir.glob(args.pattern) if "merged" not in path.name
    )

    weights = {
        "platform": args.platform_weight,
        "trace": args.trace_weight,
        "anomaly": args.anomaly_weight,
        "anomaly_once": args.anomaly_once,
        "elapsed_cap": args.elapsed_cap,
        "failure_penalty": args.failure_penalty,
        "slow_threshold": args.slow_threshold,
        "rate_mode": args.rate_mode,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_path.with_suffix(output_path.suffix + ".lock")
    if lock_path.exists():
        print(f"已有进程在写入，请勿重复运行。若确认无其他进程，可删除锁文件后重试: {lock_path}")
        sys.exit(1)
    try:
        lock_path.write_text("", encoding="utf-8")
    except OSError:
        print(f"无法创建锁文件: {lock_path}")
        sys.exit(1)

    # 第一遍：按文件顺序收集 (sample, key, idx)，以及所有需计算的 key
    ordered_items: list[tuple[dict, tuple[str, int, int], str]] = []
    keys_to_compute: set[tuple[str, int, int]] = set()
    for file_path in sample_files:
        date = None
        date_str = file_path.stem.replace("sft_samples_", "")
        try:
            date = datetime.strptime(date_str, "%Y_%m_%d")
        except ValueError:
            date = None
        samples = load_jsonl(file_path)
        for sample in samples:
            event = sample.get("input", {}).get("fault_event", {})
            idx = event.get("index", "")
            log_time = parse_iso_time(event.get("log_time", ""))
            window_start = parse_iso_time(event.get("window_start", ""))
            window_end = parse_iso_time(event.get("window_end", ""))
            
            # 优先使用variant字段中的窗口信息（多窗口数据增强）
            variant = sample.get("variant", {})
            if variant and "window_before" in variant and "window_after" in variant:
                window_before = variant["window_before"]
                window_after = variant["window_after"]
            else:
                window_before, window_after = compute_window_minutes(
                    log_time,
                    window_start,
                    window_end,
                    args.window_before,
                    args.window_after,
                )
            event_date = date
            if log_time:
                event_date = datetime(log_time.year, log_time.month, log_time.day)
            if not event_date:
                continue
            key = (event_date.strftime("%Y-%m-%d"), window_before, window_after)
            ordered_items.append((sample, key, idx))
            keys_to_compute.add(key)

    # I/O + 统计混合任务，进程数可设为 CPU 核心数的 1-2 倍
    default_workers = (os.cpu_count() or 4) * 2
    workers = args.workers if args.workers is not None else default_workers
    workers = max(1, workers)
    print(f"使用 {workers} 个进程并行计算 Top-K（共 {len(keys_to_compute)} 个唯一 key）")

    cache: dict[tuple[str, int, int], dict] = {}
    try:
        payloads = [(key, weights) for key in keys_to_compute]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(compute_one_key, payload): payload for payload in payloads}
            for future in as_completed(futures):
                key, pred_map = future.result()
                cache[key] = pred_map

        with output_path.open("w", encoding="utf-8") as out_handle:
            for sample, key, idx in ordered_items:
                top_candidates = cache[key].get(idx, [])
                llm_input = {
                    "instruction": "给定观测证据与候选根因列表，输出根因组件与故障类型，并给出简要解释。",
                    "input": {
                        "fault_event": sample.get("input", {}).get("fault_event", {}),
                        "evidence": sample.get("input", {}).get("evidence", {}),
                        "top_candidates": top_candidates,
                    },
                    "output": sample.get("output", {}),
                }
                # 保留variant信息（如果存在）
                if sample.get("variant"):
                    llm_input["input"]["variant"] = sample["variant"]
                out_handle.write(json.dumps(llm_input, ensure_ascii=False) + "\n")

        print(f"输出路径: {output_path}，共 {len(ordered_items)} 条")
    finally:
        if lock_path.exists():
            lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
