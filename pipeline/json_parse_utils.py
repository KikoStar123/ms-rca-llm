"""
LLM 输出中的 JSON 抽取与解析失败原因分类（与 train_qwen2_5_7b_qlora_demo 逻辑一致）。
"""
from __future__ import annotations

import json
import re
from typing import Any


def sanitize_generation(text: str) -> str:
    stripped = text.strip()
    if "```" in stripped:
        parts = stripped.split("```")
        if len(parts) >= 3:
            candidate = parts[1].strip()
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].lstrip()
            return candidate
    return stripped


def extract_json(text: str) -> dict[str, Any] | None:
    """从模型输出中提取第一个 JSON 对象。"""
    cleaned = sanitize_generation(text)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = cleaned[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return None


def parse_failure_reason(text: str) -> str:
    """用于评估与日志：不可解析时的粗分类。"""
    if not text or not str(text).strip():
        return "empty_output"
    t = str(text).strip()
    if "{" not in t:
        return "no_json_brace"
    if "```" in t and extract_json(t) is None:
        return "markdown_fence_decode_error"
    if extract_json(t) is None:
        if re.search(r"}\s*{", t):
            return "multiple_json_objects"
        return "json_decode_error"
    return "ok"


REQUIRED_RCA_KEYS = frozenset(
    {"root_cause_component", "fault_type", "kpi", "related_container"}
)


def schema_ok(obj: dict[str, Any] | None) -> bool:
    if not obj or not isinstance(obj, dict):
        return False
    return REQUIRED_RCA_KEYS.issubset(obj.keys())
