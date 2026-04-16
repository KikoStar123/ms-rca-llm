import json
import os
import random
import sys
from datetime import datetime
from dataclasses import dataclass
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

# 导入路径配置（支持本地/服务器环境自动切换）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import config
    BASE_DIR = config.BASE_DIR
    DATA_PATH = config.DATA_PATH
    RESULT_DIR = config.RESULT_DIR
    MODEL_PATH = config.MODEL_PATH
    MODEL_DIR = config.MODEL_CKPT_DIR
except ImportError:
    # 兼容旧版本：如果 config.py 不存在，使用默认路径
    BASE_DIR = r"d:\Graduation_Project" if os.path.exists(r"d:\Graduation_Project") else "/root/Graduation_Project"
    DATA_PATH = os.path.join(BASE_DIR, "output", "llm_inputs_v4.jsonl")
    RESULT_DIR = os.path.join(BASE_DIR, "v2_doc")
    MODEL_DIR = os.path.join(RESULT_DIR, "model_ckpt")
    MODEL_PATH = r"D:\hf_cache\Qwen2.5-7B-Instruct" if os.path.exists(r"D:\hf_cache\Qwen2.5-7B-Instruct") else "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"

# 0 表示使用全部样本；>0 时仅取前 N 条（用于快速试验）
MAX_SAMPLES = 0
EVAL_SIZE = 40
MIN_EVAL = 25
MAX_LENGTH = 384
SEED = 42
SKIP_BEFORE_EVAL = False
# 0 表示按 1 个 epoch 自动计算步数；>0 时固定步数
MAX_STEPS = 0
USE_4BIT = True
OUTPUT_KEYS = ["root_cause_component", "fault_type", "kpi", "related_container"]
SYSTEM_PROMPT = "你是微服务故障根因定位助手，输出必须为可解析JSON。"
RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")


def _set_submodule_fallback(module: torch.nn.Module, target: str, new_module: torch.nn.Module) -> None:
    parts = target.split(".")
    parent = module
    for name in parts[:-1]:
        if name.isdigit():
            parent = parent[int(name)]
        else:
            parent = getattr(parent, name)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)


if not hasattr(torch.nn.Module, "set_submodule"):
    torch.nn.Module.set_submodule = _set_submodule_fallback


@dataclass
class EvalResult:
    total: int
    parsed: int
    root_acc: float
    type_acc: float


def load_samples(path: str) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def ensure_model() -> None:
    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(f"模型路径不存在：{MODEL_PATH}")
    shard_files = [
        name
        for name in os.listdir(MODEL_PATH)
        if name.startswith("model-") and name.endswith(".safetensors") and "of-00004" in name
    ]
    if len(shard_files) < 4:
        raise FileNotFoundError(f"模型分片不完整（应为4个）：{MODEL_PATH}")


def build_user_content(sample: dict[str, Any]) -> tuple[str, str]:
    instruction = sample.get("instruction", "")
    input_obj = sample.get("input", {})
    output_obj = sample.get("output", {})
    key_hint = ", ".join(OUTPUT_KEYS)
    prompt = (
        f"{instruction}\n"
        "root_cause_component 必须从 top_candidates 中选择。\n"
        "fault_type 必须与输入 fault_event.fault_description 完全一致（保持原语言）。\n"
        "请仅输出JSON对象，不要输出解释或Markdown。\n"
        f"JSON只包含以下键：{key_hint}。\n"
        "只输出单行JSON，不要添加额外字符。\n"
        f"输入：{json.dumps(input_obj, ensure_ascii=False)}"
    )
    answer = json.dumps(output_obj, ensure_ascii=False)
    return prompt, answer


def build_chat_text(sample: dict[str, Any], tokenizer: AutoTokenizer, include_answer: bool) -> str:
    prompt, answer = build_user_content(sample)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    if include_answer:
        messages.append({"role": "assistant", "content": answer})
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=not include_answer,
        )
    fallback = (
        "### 指令：\n"
        f"{prompt}\n"
        "### 回复：\n"
    )
    return fallback + (answer if include_answer else "")


def build_train_item(sample: dict[str, Any], tokenizer: AutoTokenizer) -> dict[str, str]:
    return {
        "prompt_text": build_chat_text(sample, tokenizer, include_answer=False),
        "full_text": build_chat_text(sample, tokenizer, include_answer=True),
    }


def extract_json(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = text[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return None


def sanitize_generation(text: str) -> str:
    stripped = text.strip()
    if "```" in stripped:
        parts = stripped.split("```")
        if len(parts) >= 3:
            candidate = parts[1].strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].lstrip()
            return candidate
    return stripped


def get_generation_eos(tokenizer: AutoTokenizer) -> int | list[int] | None:
    eos_id = tokenizer.eos_token_id
    im_end_id = None
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_id == tokenizer.unk_token_id:
            im_end_id = None
    if im_end_id is None:
        return eos_id
    if eos_id is None:
        return im_end_id
    if eos_id == im_end_id:
        return eos_id
    return [eos_id, im_end_id]


def evaluate_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    samples: list[dict[str, Any]],
    out_path: str,
) -> EvalResult:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    parsed = 0
    root_hit = 0
    type_hit = 0
    model.eval()
    eos_token_id = get_generation_eos(tokenizer)
    with open(out_path, "w", encoding="utf-8") as handle:
        for sample in samples:
            prompt_text = build_chat_text(sample, tokenizer, include_answer=False)
            inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    num_beams=1,
                    eos_token_id=eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            gen_ids = outputs[0][inputs["input_ids"].shape[-1] :]
            raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            payload = extract_json(sanitize_generation(raw_text))
            record = {
                "prompt": prompt_text,
                "prediction": payload,
                "label": sample.get("output", {}),
            }
            if payload is None:
                record["raw_text"] = raw_text
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if payload is None:
                continue
            parsed += 1
            if payload.get("root_cause_component") == sample.get("output", {}).get(
                "root_cause_component"
            ):
                root_hit += 1
            if payload.get("fault_type") == sample.get("output", {}).get("fault_type"):
                type_hit += 1
    total = len(samples)
    root_acc = root_hit / total if total else 0.0
    type_acc = type_hit / total if total else 0.0
    return EvalResult(total=total, parsed=parsed, root_acc=root_acc, type_acc=type_acc)


def main() -> None:
    random.seed(SEED)
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    ensure_model()

    all_samples = load_samples(DATA_PATH)
    random.shuffle(all_samples)
    if MAX_SAMPLES > 0:
        all_samples = all_samples[:MAX_SAMPLES]
    total = len(all_samples)
    if total < 4:
        raise ValueError("样本数量过少，至少需要 4 条样本用于训练/评估分割。")
    eval_size = min(EVAL_SIZE, max(MIN_EVAL, total // 5))
    train_size = total - eval_size
    train_samples = all_samples[:train_size]
    eval_samples = all_samples[train_size : train_size + eval_size]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"

    if USE_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
    model.config.use_cache = False
    if USE_4BIT:
        model = prepare_model_for_kbit_training(model)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.is_parallelizable = True
    model.model_parallel = True

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable} / {total}")

    before_path = os.path.join(RESULT_DIR, f"llm_eval_before_{RUN_TAG}.jsonl")
    if SKIP_BEFORE_EVAL:
        before_metrics = EvalResult(total=len(eval_samples), parsed=0, root_acc=0.0, type_acc=0.0)
        with open(before_path, "w", encoding="utf-8") as handle:
            for sample in eval_samples:
                record = {
                    "prompt": build_chat_text(sample, tokenizer, include_answer=False),
                    "prediction": None,
                    "label": sample.get("output", {}),
                    "skipped": True,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    else:
        before_metrics = evaluate_model(model, tokenizer, eval_samples, before_path)

    train_dataset = Dataset.from_list([build_train_item(s, tokenizer) for s in train_samples])

    def tokenize_fn(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        input_ids_list: list[list[int]] = []
        attention_masks: list[list[int]] = []
        labels_list: list[list[int]] = []
        pad_id = tokenizer.pad_token_id
        for prompt_text, full_text in zip(batch["prompt_text"], batch["full_text"]):
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
            full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
            if len(full_ids) <= len(prompt_ids):
                answer_ids: list[int] = []
            else:
                answer_ids = full_ids[len(prompt_ids) :]

            if len(answer_ids) >= MAX_LENGTH:
                input_ids = answer_ids[-MAX_LENGTH:]
                labels = input_ids.copy()
            else:
                max_prompt = MAX_LENGTH - len(answer_ids)
                prompt_ids = prompt_ids[-max_prompt:] if max_prompt > 0 else []
                input_ids = prompt_ids + answer_ids
                labels = [-100] * len(prompt_ids) + answer_ids

            attention_mask = [1] * len(input_ids)
            if len(input_ids) < MAX_LENGTH:
                pad_len = MAX_LENGTH - len(input_ids)
                input_ids = input_ids + [pad_id] * pad_len
                attention_mask = attention_mask + [0] * pad_len
                labels = labels + [-100] * pad_len

            input_ids_list.append(input_ids)
            attention_masks.append(attention_mask)
            labels_list.append(labels)

        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_masks,
            "labels": labels_list,
        }

    train_dataset = train_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["prompt_text", "full_text"],
    )
    label_active = sum(1 for row in train_dataset["labels"] for token in row if token != -100)
    label_total = len(train_dataset["labels"]) * len(train_dataset["labels"][0])
    print(f"Label tokens: {label_active} / {label_total}")

    # 步数：MAX_STEPS>0 用设定值，否则按 1 epoch（batch=1*4=4）自动计算
    steps_per_epoch = max(1, (len(train_samples) + 3) // 4)
    effective_max_steps = MAX_STEPS if MAX_STEPS > 0 else steps_per_epoch
    print(f"Train samples: {len(train_samples)}, max_steps: {effective_max_steps}")

    args = TrainingArguments(
        output_dir=MODEL_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        max_steps=effective_max_steps,
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="no",
        fp16=False,
        bf16=False,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
    )
    trainer.train()

    after_path = os.path.join(RESULT_DIR, f"llm_eval_after_{RUN_TAG}.jsonl")
    after_metrics = evaluate_model(model, tokenizer, eval_samples, after_path)

    summary = {
        "model": MODEL_PATH,
        "train_size": len(train_samples),
        "eval_size": len(eval_samples),
        "before": before_metrics.__dict__,
        "after": after_metrics.__dict__,
    }
    summary_path = os.path.join(RESULT_DIR, f"训练评估结果_{RUN_TAG}.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    report_path = os.path.join(RESULT_DIR, f"训练评估报告_{RUN_TAG}.md")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("# 轻量 SFT 训练与评估报告\n\n")
        handle.write(f"- 模型：{MODEL_PATH}\n")
        handle.write(f"- 训练样本：{len(train_samples)}\n")
        handle.write(f"- 评估样本：{len(eval_samples)}\n\n")
        handle.write("## 微调前\n")
        handle.write(
            f"- 可解析率：{before_metrics.parsed}/{before_metrics.total}\n"
            f"- 根因 Top‑1：{before_metrics.root_acc:.3f}\n"
            f"- 故障类型 Acc：{before_metrics.type_acc:.3f}\n\n"
        )
        handle.write("## 微调后\n")
        handle.write(
            f"- 可解析率：{after_metrics.parsed}/{after_metrics.total}\n"
            f"- 根因 Top‑1：{after_metrics.root_acc:.3f}\n"
            f"- 故障类型 Acc：{after_metrics.type_acc:.3f}\n\n"
        )
        handle.write("## 备注\n")
        handle.write("- 该结果基于轻量样本与单轮训练，仅用于验证流程可跑通与趋势。\n")
        handle.write("- 详细预测见：llm_eval_before.jsonl / llm_eval_after.jsonl。\n")


if __name__ == "__main__":
    main()
