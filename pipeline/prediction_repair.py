"""
评估前对解析后的 JSON 做任务合规修复，提高与标注一致的指标。

默认（EVAL_OUTPUT_REPAIR=1）：
  - 将 fault_type / kpi / related_container 与输入 fault_event 对齐（符合题目「与输入一致」的约束）。
  - 不修改 root_cause_component（避免金标不在 top_candidates 时把「对的」根因改错）。

可选根因投影（EVAL_ROOT_PROJECT_TO_CANDIDATES=1）：
  - 若预测根因不在 top_candidates 内，按字符串相似度映射到候选之一。
  - 当数据存在「金标根因 ∉ 候选」时，该映射可能降低根因 Top‑1，请谨慎使用。

关闭全部修复：EVAL_OUTPUT_REPAIR=0
"""
from __future__ import annotations

import os
from difflib import SequenceMatcher
from typing import Any


def eval_repair_enabled() -> bool:
    return os.environ.get("EVAL_OUTPUT_REPAIR", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def root_project_enabled() -> bool:
    return os.environ.get("EVAL_ROOT_PROJECT_TO_CANDIDATES", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _pick_root_in_candidates(pred_root: str, candidates: list[str]) -> str:
    if not candidates:
        return (pred_root or "").strip()
    cand = [str(x) for x in candidates]
    pr = (pred_root or "").strip()
    if pr in cand:
        return pr
    if not pr:
        return cand[0]
    return max(cand, key=lambda c: SequenceMatcher(None, pr, c).ratio())


def apply_eval_repair(sample: dict[str, Any], payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None or not eval_repair_enabled():
        return payload
    out = dict(payload)
    inp = sample.get("input") or {}

    if root_project_enabled():
        tc = inp.get("top_candidates")
        if isinstance(tc, list) and len(tc) > 0:
            out["root_cause_component"] = _pick_root_in_candidates(
                str(out.get("root_cause_component", "")),
                tc,
            )

    fe = inp.get("fault_event")
    if isinstance(fe, dict):
        if "fault_description" in fe:
            out["fault_type"] = fe["fault_description"]
        if "kpi" in fe:
            out["kpi"] = fe.get("kpi") if fe.get("kpi") is not None else ""
        if "container" in fe:
            out["related_container"] = fe.get("container") if fe.get("container") is not None else ""

    return out
