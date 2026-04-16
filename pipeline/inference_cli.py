"""
演示用推理入口：mock（无需 GPU）或加载本地 Qwen 权重生成结构化 JSON。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from json_parse_utils import REQUIRED_RCA_KEYS, extract_json, sanitize_generation

# 与 train_qwen2_5_7b_qlora_demo 一致
SYSTEM_PROMPT = "你是微服务故障根因定位助手，输出必须为可解析JSON。"
OUTPUT_KEYS = ["root_cause_component", "fault_type", "kpi", "related_container"]


def build_user_prompt(sample: dict[str, Any]) -> str:
    instruction = sample.get("instruction", "")
    input_obj = sample.get("input", {})
    key_hint = ", ".join(OUTPUT_KEYS)
    return (
        f"{instruction}\n"
        "root_cause_component 必须从 top_candidates 中选择。\n"
        "fault_type 必须与输入 fault_event.fault_description 完全一致（保持原语言）。\n"
        "请仅输出JSON对象，不要输出解释或Markdown。\n"
        f"JSON只包含以下键：{key_hint}。\n"
        "只输出单行JSON，不要添加额外字符。\n"
        f"输入：{json.dumps(input_obj, ensure_ascii=False)}"
    )


def mock_predict(sample: dict[str, Any]) -> dict[str, Any]:
    """无模型时：取候选第一项为根因，其余字段来自 fault_event / 标签。"""
    inp = sample.get("input") or {}
    fe = inp.get("fault_event") or {}
    cands = inp.get("top_candidates") or []
    root = cands[0] if cands else fe.get("name", "")
    return {
        "root_cause_component": root,
        "fault_type": fe.get("fault_description", ""),
        "kpi": fe.get("kpi", ""),
        "related_container": fe.get("container", ""),
    }


def run_transformers(prompt_text: str, model_path: str, max_new_tokens: int) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if hasattr(tok, "padding_side"):
        tok.padding_side = "left"

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=bnb,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        torch_dtype=torch.float16,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text},
    ]
    if hasattr(tok, "apply_chat_template"):
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = f"### 指令：\n{prompt_text}\n### 回复：\n"
    inputs = tok(text, return_tensors="pt").to(model.device)
    eos_id = tok.eos_token_id
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            eos_token_id=eos_id,
            pad_token_id=tok.pad_token_id,
        )
    gen = out[0][inputs["input_ids"].shape[-1] :]
    return tok.decode(gen, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="RCA 推理 CLI（mock 或 Qwen）")
    parser.add_argument("--input", default="", help="单条样本 JSON 文件；缺省从 stdin 读 JSON")
    parser.add_argument("--mock", action="store_true", help="不加载模型，使用规则占位输出")
    parser.add_argument("--model", default="", help="Qwen 模型目录（4bit 加载）；与 --mock 互斥")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    if args.input:
        raw = Path(args.input).read_text(encoding="utf-8")
        sample = json.loads(raw)
    else:
        raw = sys.stdin.read()
        sample = json.loads(raw)

    if args.mock:
        out = mock_predict(sample)
        print(json.dumps(out, ensure_ascii=False))
        return

    if not args.model:
        print("请指定 --model 路径或使用 --mock", file=sys.stderr)
        sys.exit(2)

    prompt = build_user_prompt(sample)
    raw_text = run_transformers(prompt, args.model, args.max_new_tokens)
    parsed = extract_json(sanitize_generation(raw_text))
    if parsed is None:
        print(json.dumps({"error": "parse_failed", "raw_text": raw_text}, ensure_ascii=False))
        sys.exit(1)
    missing = REQUIRED_RCA_KEYS - set(parsed.keys())
    if missing:
        print(json.dumps({"error": "missing_keys", "missing": list(missing), "partial": parsed}, ensure_ascii=False))
        sys.exit(1)
    print(json.dumps(parsed, ensure_ascii=False))


if __name__ == "__main__":
    main()
