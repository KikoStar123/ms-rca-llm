import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def parse_date(value: str) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def extract_features(item: dict) -> dict:
    fault_event = item.get("input", {}).get("fault_event", {})
    obj = fault_event.get("object", "")
    kpi = fault_event.get("kpi", "")
    log_time = fault_event.get("log_time", "")
    return {
        "object": obj,
        "kpi": kpi,
        "log_time": log_time,
    }


def build_tables(train_items: list[dict]) -> dict:
    by_object_kpi = defaultdict(Counter)
    by_kpi = Counter()
    by_object = Counter()
    global_counts = Counter()

    for item in train_items:
        features = extract_features(item)
        fault_type = item.get("output", {}).get("fault_type", "unknown")
        obj = features["object"]
        kpi = features["kpi"]

        by_object_kpi[(obj, kpi)][fault_type] += 1
        if kpi:
            by_kpi[kpi] += 1
        if obj:
            by_object[obj] += 1
        global_counts[fault_type] += 1

    return {
        "by_object_kpi": by_object_kpi,
        "by_kpi": by_kpi,
        "by_object": by_object,
        "global": global_counts,
    }


def predict_fault_type(features: dict, tables: dict) -> str:
    obj = features["object"]
    kpi = features["kpi"]

    key = (obj, kpi)
    if key in tables["by_object_kpi"] and tables["by_object_kpi"][key]:
        return tables["by_object_kpi"][key].most_common(1)[0][0]
    if kpi and tables["by_kpi"]:
        return tables["by_kpi"].most_common(1)[0][0]
    if obj and tables["by_object"]:
        return tables["by_object"].most_common(1)[0][0]
    if tables["global"]:
        return tables["global"].most_common(1)[0][0]
    return "unknown"


def split_by_last_date(items: list[dict]) -> tuple[list[dict], list[dict], str]:
    dated_items = []
    for item in items:
        log_time = item.get("input", {}).get("fault_event", {}).get("log_time", "")
        dt = parse_date(log_time)
        if not dt:
            continue
        dated_items.append((dt, item))
    if not dated_items:
        return items, [], ""

    dated_items.sort(key=lambda x: x[0])
    last_date = dated_items[-1][0].date()
    train, test = [], []
    for dt, item in dated_items:
        if dt.date() == last_date:
            test.append(item)
        else:
            train.append(item)
    return train, test, last_date.isoformat()


def evaluate(items: list[dict], tables: dict) -> dict:
    correct = 0
    total = 0
    confusion = Counter()
    predictions = []

    for item in items:
        features = extract_features(item)
        true_type = item.get("output", {}).get("fault_type", "unknown")
        pred_type = predict_fault_type(features, tables)
        predictions.append(
            {
                "features": features,
                "true_fault_type": true_type,
                "pred_fault_type": pred_type,
            }
        )
        if true_type == pred_type:
            correct += 1
        total += 1
        confusion[(true_type, pred_type)] += 1

    accuracy = correct / total if total else 0.0
    return {
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "confusion": confusion,
        "predictions": predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="小模型基线：故障类型预测")
    parser.add_argument("--input", required=True, help="输入JSONL路径")
    parser.add_argument("--output", required=True, help="预测输出JSONL路径")
    parser.add_argument("--report", required=True, help="评估报告JSON路径")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report)

    items = load_jsonl(input_path)
    train_items, test_items, test_date = split_by_last_date(items)
    tables = build_tables(train_items)
    result = evaluate(test_items, tables)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in result["predictions"]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    confusion_list = [
        {"true": true, "pred": pred, "count": count}
        for (true, pred), count in result["confusion"].most_common()
    ]
    report = {
        "test_date": test_date,
        "total": result["total"],
        "correct": result["correct"],
        "accuracy": result["accuracy"],
        "confusion": confusion_list,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(f"测试日期: {test_date}")
    print(f"样本数: {result['total']}, 准确率: {result['accuracy']}")
    print(f"输出路径: {output_path}")
    print(f"报告路径: {report_path}")


if __name__ == "__main__":
    main()
