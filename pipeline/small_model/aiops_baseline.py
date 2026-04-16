import csv
import os
from pathlib import Path
from zipfile import ZipFile


BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "data" / "aiops2020"
ZIP_PATH = DATA_ROOT / "AIOps挑战赛数据" / "2020_04_11.zip"
EXTRACT_DIR = DATA_ROOT / "day_2020_04_11"
FAILURE_CSV = DATA_ROOT / "故障整理（预赛）.csv"


def ensure_extracted() -> None:
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    has_files = any(EXTRACT_DIR.rglob("*"))
    if has_files:
        return
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"找不到数据包: {ZIP_PATH}")
    with ZipFile(ZIP_PATH, "r") as zip_file:
        zip_file.extractall(EXTRACT_DIR)


def preview_csv(file_path: Path, max_rows: int = 3) -> list[list[str]]:
    rows: list[list[str]] = []
    with file_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        for _ in range(max_rows):
            try:
                rows.append(next(reader))
            except StopIteration:
                break
    return rows


def summarize_failures() -> None:
    if not FAILURE_CSV.exists():
        print(f"未找到故障清单: {FAILURE_CSV}")
        return
    rows = preview_csv(FAILURE_CSV, max_rows=5)
    print("故障清单预览（前5行）:")
    for row in rows:
        print("  ", row)


def summarize_day_data() -> None:
    print(f"日数据目录: {EXTRACT_DIR}")
    top_dirs = []
    for entry in EXTRACT_DIR.iterdir():
        if entry.is_dir():
            top_dirs.append(entry.name)
    print("日数据子目录:", top_dirs)

    csv_files = [path for path in EXTRACT_DIR.rglob("*.csv")]
    print("CSV文件总数:", len(csv_files))
    if not csv_files:
        return

    for sample in csv_files[:3]:
        rows = preview_csv(sample, max_rows=3)
        print(f"\n样例文件: {sample}")
        for row in rows:
            print("  ", row)


def main() -> None:
    print("== AIOps Challenge 数据集快速跑通 ==")
    print(f"数据根目录: {DATA_ROOT}")
    ensure_extracted()
    summarize_failures()
    summarize_day_data()


if __name__ == "__main__":
    main()
