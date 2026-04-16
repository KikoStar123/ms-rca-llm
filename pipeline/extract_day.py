import argparse
from pathlib import Path
from zipfile import ZipFile


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_ROOT = BASE_DIR / "data" / "aiops2020"
ZIP_DIR = DATA_ROOT / "AIOps挑战赛数据"


def extract_day(target_date: str) -> Path:
    date_key = target_date.replace("-", "_")
    zip_path = ZIP_DIR / f"{date_key}.zip"
    day_root = DATA_ROOT / f"day_{date_key}" / date_key
    day_root.parent.mkdir(parents=True, exist_ok=True)

    if day_root.exists() and any(day_root.rglob("*")):
        return day_root

    if not zip_path.exists():
        raise FileNotFoundError(f"未找到压缩包: {zip_path}")

    with ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(day_root.parent)
    return day_root


def main() -> None:
    parser = argparse.ArgumentParser(description="解压AIOps指定日期数据")
    parser.add_argument("--date", required=True, help="日期，格式YYYY-MM-DD")
    args = parser.parse_args()

    day_root = extract_day(args.date)
    print(f"解压完成: {day_root}")


if __name__ == "__main__":
    main()
