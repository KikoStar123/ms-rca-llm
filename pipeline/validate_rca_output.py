import argparse
import json
from pathlib import Path


REQUIRED_ROOT_FIELDS = {"event_id", "time", "top_candidates", "prediction"}
REQUIRED_PRED_FIELDS = {"root_cause_component", "fault_type", "kpi", "related_container"}


def load_jsonl(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def to_rca_output(sample: dict) -> dict:
    event = sample.get("input", {}).get("fault_event", {})
    top_candidates = sample.get("input", {}).get("top_candidates", [])
    pred = sample.get("output", {})
    return {
        "event_id": str(event.get("index", "")),
        "time": event.get("log_time", ""),
        "top_candidates": top_candidates,
        "prediction": {
            "root_cause_component": pred.get("root_cause_component", ""),
            "fault_type": pred.get("fault_type", ""),
            "kpi": pred.get("kpi", ""),
            "related_container": pred.get("related_container", ""),
            "explanation": pred.get("explanation", ""),
        },
    }


def validate_item(item: dict) -> list[str]:
    errors = []
    missing_root = REQUIRED_ROOT_FIELDS - set(item.keys())
    if missing_root:
        errors.append(f"missing_root_fields={sorted(missing_root)}")
    if "top_candidates" in item and not isinstance(item["top_candidates"], list):
        errors.append("top_candidates_not_list")
    pred = item.get("prediction", {})
    if not isinstance(pred, dict):
        errors.append("prediction_not_object")
        return errors
    missing_pred = REQUIRED_PRED_FIELDS - set(pred.keys())
    if missing_pred:
        errors.append(f"missing_pred_fields={sorted(missing_pred)}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="校验RCA输出格式")
    parser.add_argument("--input", required=True, help="LLM输入样本JSONL")
    parser.add_argument("--output", required=True, help="校验后的输出JSONL")
    parser.add_argument("--report", required=True, help="校验报告JSON")
    args = parser.parse_args()

    samples = load_jsonl(Path(args.input))
    outputs = [to_rca_output(sample) for sample in samples]

    errors = []
    for idx, item in enumerate(outputs):
        item_errors = validate_item(item)
        if item_errors:
            errors.append({"index": idx, "errors": item_errors})

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for item in outputs:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    report = {
        "total": len(outputs),
        "invalid": len(errors),
        "errors": errors,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(f"总数: {report['total']}, 无效: {report['invalid']}")
    print(f"输出路径: {output_path}")
    print(f"报告路径: {report_path}")


if __name__ == "__main__":
    main()
