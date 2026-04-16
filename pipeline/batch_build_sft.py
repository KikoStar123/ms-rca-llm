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

from build_sft_samples import FAILURE_CSV, build_sft_samples, parse_time
from extract_day import extract_day
from sft_stats import summarize_items


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"


def collect_dates() -> list[datetime]:
    dates = set()
    if not FAILURE_CSV.exists():
        raise FileNotFoundError(f"未找到故障清单: {FAILURE_CSV}")
    with FAILURE_CSV.open("r", encoding="utf-8", errors="replace") as handle:
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


def write_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def parse_window_presets(value: str, default_before: int, default_after: int) -> list[tuple[int, int]]:
    if not value:
        return [(default_before, default_after)]
    presets = []
    for raw in value.split(";"):
        raw = raw.strip()
        if not raw:
            continue
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) != 2:
            raise ValueError(f"窗口配置格式错误: {raw}")
        presets.append((int(parts[0]), int(parts[1])))
    return presets or [(default_before, default_after)]


def process_one_date(params: tuple) -> tuple[str, int, int]:
    """处理单个日期的SFT样本生成（用于多进程，必须在模块级别）"""
    target_date, window_presets, max_items, limit_events, overwrite, output_dir_str = params
    # 重新导入必要的模块（多进程需要）
    from datetime import datetime
    from pathlib import Path
    from build_sft_samples import build_sft_samples
    from extract_day import extract_day
    from sft_stats import summarize_items
    
    output_dir = Path(output_dir_str)
    date_key = target_date.strftime("%Y_%m_%d")
    output_samples = output_dir / f"sft_samples_{date_key}.jsonl"
    output_stats = output_dir / f"sft_stats_{date_key}.json"
    
    if output_samples.exists() and output_stats.exists() and not overwrite:
        return (date_key, 0, 1)  # (date_key, count, status: 1=skipped)
    
    try:
        extract_day(target_date.strftime("%Y-%m-%d"))
    except FileNotFoundError:
        return (date_key, 0, 2)  # status: 2=missing data
    
    all_samples: list[dict] = []
    for before, after in window_presets:
        samples = build_sft_samples(
            target_date=target_date,
            window_before=before,
            window_after=after,
            max_items=max_items,
            limit=limit_events or None,
        )
        if len(window_presets) > 1:
            for item in samples:
                item["variant"] = {"window_before": before, "window_after": after}
        all_samples.extend(samples)
    
    # write_jsonl
    output_samples.parent.mkdir(parents=True, exist_ok=True)
    with output_samples.open("w", encoding="utf-8") as handle:
        for item in all_samples:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    stats = summarize_items(all_samples)
    output_stats.parent.mkdir(parents=True, exist_ok=True)
    with output_stats.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)
    
    return (date_key, len(all_samples), 0)  # status: 0=success


def main() -> None:
    parser = argparse.ArgumentParser(description="批量生成SFT样本")
    parser.add_argument("--start-date", default="", help="起始日期YYYY-MM-DD")
    parser.add_argument("--end-date", default="", help="结束日期YYYY-MM-DD")
    parser.add_argument("--limit-events", type=int, default=0, help="每个日期限制样本数")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在输出")
    parser.add_argument("--window-before", type=int, default=5, help="窗口前分钟数")
    parser.add_argument("--window-after", type=int, default=5, help="窗口后分钟数")
    parser.add_argument("--max-items", type=int, default=5, help="每类摘要条数上限")
    parser.add_argument(
        "--window-presets",
        default="",
        help="多窗口配置，格式: before,after;before,after",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="并行进程数，默认取 CPU 核心数",
    )
    args = parser.parse_args()

    all_dates = collect_dates()
    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        all_dates = [d for d in all_dates if d >= start_date]
    if args.end_date:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        all_dates = [d for d in all_dates if d <= end_date]

    window_presets = parse_window_presets(
        args.window_presets, args.window_before, args.window_after
    )
    
    # 设置并行workers
    default_workers = os.cpu_count() or 4
    workers = args.workers if args.workers is not None else default_workers
    workers = min(workers, len(all_dates))  # 不超过日期数量
    
    # 使用多进程并行处理
    if workers > 1 and len(all_dates) > 1:
        print(f"使用 {workers} 个进程并行生成SFT样本（共 {len(all_dates)} 个日期）")
        params_list = [
            (date, window_presets, args.max_items, args.limit_events, args.overwrite, str(OUTPUT_DIR))
            for date in all_dates
        ]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_one_date, params): params[0] for params in params_list}
            for future in as_completed(futures):
                date_key, count, status = future.result()
                if status == 1:
                    print(f"[跳过] {date_key} 已存在输出")
                elif status == 2:
                    print(f"[跳过] {date_key} 缺少数据包")
                else:
                    print(f"[完成] {date_key} 样本数: {count}")
    else:
        # 单进程模式（兼容）
        for target_date in all_dates:
            date_key = target_date.strftime("%Y_%m_%d")
            output_samples = OUTPUT_DIR / f"sft_samples_{date_key}.jsonl"
            output_stats = OUTPUT_DIR / f"sft_stats_{date_key}.json"
            if output_samples.exists() and output_stats.exists() and not args.overwrite:
                print(f"[跳过] {date_key} 已存在输出")
                continue

            try:
                extract_day(target_date.strftime("%Y-%m-%d"))
            except FileNotFoundError as exc:
                print(f"[跳过] {date_key} 缺少数据包: {exc}")
                continue

            all_samples: list[dict] = []
            for before, after in window_presets:
                samples = build_sft_samples(
                    target_date=target_date,
                    window_before=before,
                    window_after=after,
                    max_items=args.max_items,
                    limit=args.limit_events or None,
                )
                if len(window_presets) > 1:
                    for item in samples:
                        item["variant"] = {"window_before": before, "window_after": after}
                all_samples.extend(samples)

            write_jsonl(output_samples, all_samples)

            stats = summarize_items(all_samples)
            output_stats.parent.mkdir(parents=True, exist_ok=True)
            with output_stats.open("w", encoding="utf-8") as handle:
                json.dump(stats, handle, ensure_ascii=False, indent=2)

            print(f"[完成] {date_key} 样本数: {len(all_samples)}")


if __name__ == "__main__":
    main()
