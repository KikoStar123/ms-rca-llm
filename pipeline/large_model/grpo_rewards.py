"""
GRPO / 偏好优化阶段的标量奖励（与 research/GRPO训练配置模板 一致，可接入 TRL GRPOTrainer）。

reward ∈ [-1, 1] 量级，便于与 KL 系数组合；实际系数在训练脚本中再缩放。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_pipeline = Path(__file__).resolve().parent.parent
if str(_pipeline) not in sys.path:
    sys.path.insert(0, str(_pipeline))

from json_parse_utils import REQUIRED_RCA_KEYS, schema_ok


def normalize_str(s: str) -> str:
    return " ".join(str(s).split())


def reward_rca(
    pred: dict[str, Any] | None,
    label: dict[str, Any],
    *,
    parsed: bool,
) -> float:
    """
    总奖励：可解析 + 字段正确性 + 根因命中（权重最大）。
    """
    if not parsed or pred is None or not isinstance(pred, dict):
        return -1.0
    if not schema_ok(pred):
        return -0.5

    lr = label.get("root_cause_component", "")
    lt = normalize_str(str(label.get("fault_type", "")))
    lk = label.get("kpi", "")
    lc = label.get("related_container", "")

    pr = pred.get("root_cause_component", "")
    pt = normalize_str(str(pred.get("fault_type", "")))
    pk = pred.get("kpi", "")
    pc = pred.get("related_container", "")

    r = 0.0
    if pr == lr:
        r += 0.45
    if pt == lt:
        r += 0.2
    if pk == lk:
        r += 0.15
    if pc == lc:
        r += 0.2
    return min(1.0, r)


def reward_format_only(pred: dict[str, Any] | None, parsed: bool) -> float:
    """仅格式合规（用于分阶段调试）。"""
    if not parsed or not schema_ok(pred):
        return -1.0
    return 0.2


def reward_rca_trl(completions: list[str], gold_label: list[str], **kwargs: Any) -> list[float]:
    """
    TRL GRPOTrainer 用：与数据集列 `gold_label`（每条样本的 JSON 字符串）对齐，
    对 `completions` 逐条解析并调用 `reward_rca`。
    """
    from json_parse_utils import extract_json

    rewards: list[float] = []
    for comp, gl in zip(completions, gold_label, strict=True):
        try:
            label = json.loads(gl) if isinstance(gl, str) else gl
        except (json.JSONDecodeError, TypeError):
            rewards.append(-1.0)
            continue
        if not isinstance(label, dict):
            rewards.append(-1.0)
            continue
        pred = extract_json(comp)
        parsed = pred is not None
        rewards.append(reward_rca(pred, label, parsed=parsed))
    return rewards
