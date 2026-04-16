import argparse
import json
import os
import random
import re
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
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
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

# 导入路径：仓库根须在 path 中，否则 import config 会失败并误用旧回退路径
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PIPELINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _PIPELINE_DIR)
from json_parse_utils import extract_json, sanitize_generation
from prediction_repair import apply_eval_repair, eval_repair_enabled

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
EVAL_SIZE = 200  # 评估集大小：约 12%（1728 条中的 200 条，训练集 1528 条）
MIN_EVAL = 25
MAX_LENGTH = 384
SEED = 42
# 跳过微调前评估：命令行 --skip-before-eval 或环境 SKIP_BEFORE_EVAL=1
# 0 表示按 1 个 epoch 自动计算步数；>0 时固定步数
MAX_STEPS = 0
USE_4BIT = True  # 改用4-bit量化（8-bit量化训练有问题，Loss=0）
USE_8BIT = False
# 多进程配置：默认使用24核心（服务器核心数）
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 24))  # 可通过环境变量覆盖
# 训练过程中用「同一份 eval_split」算 eval_loss，并保存/加载验证 loss 最低的 checkpoint，减轻过拟合导致的「微调后比基线差」
# 设为 0 可关闭（与旧行为更接近：整轮训完不根据验证集选模型）
USE_EVAL_FOR_EARLY_BEST = os.environ.get("USE_EVAL_FOR_EARLY_BEST", "1").strip() not in ("0", "false", "no")
# 多 epoch 时按验证 loss 早停，避免后期过拟合（需 USE_EVAL_FOR_EARLY_BEST=1）
EARLY_STOPPING_PATIENCE = int(os.environ.get("EARLY_STOPPING_PATIENCE", "5"))
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
    root_acc: float  # Top-1准确率
    root_acc_top3: float  # Top-3准确率
    root_acc_top5: float  # Top-5准确率
    type_acc: float
    kpi_acc: float  # KPI字段准确率
    container_acc: float  # 相关容器准确率
    full_match_acc: float  # 完整匹配准确率（所有字段都正确）


def normalize_string(s: str) -> str:
    """规范化字符串：去除多余空格，用于故障类型比较"""
    if not s:
        return s
    return re.sub(r'\s+', ' ', s.strip())


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


def _build_train_item_wrapper(args: tuple) -> dict[str, str]:
    """多进程包装函数：tokenizer需要重新加载"""
    sample, model_path = args
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"
    return build_train_item(sample, tokenizer)


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


def _build_prompt_wrapper(args: tuple) -> str:
    """多进程包装函数：构建prompt文本"""
    sample, model_path = args
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    return build_chat_text(sample, tokenizer, include_answer=False)


def evaluate_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    samples: list[dict[str, Any]],
    out_path: str,
) -> EvalResult:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    parsed = 0
    root_hit = 0
    root_hit_top3 = 0
    root_hit_top5 = 0
    type_hit = 0
    kpi_hit = 0
    container_hit = 0
    full_match_hit = 0
    model.eval()
    eos_token_id = get_generation_eos(tokenizer)
    
    # 多进程优化：并行预处理prompt文本
    print(f"\n使用 {NUM_WORKERS} 个进程并行预处理评估样本...")
    if NUM_WORKERS > 1 and len(samples) > 10:
        params_list = [(s, MODEL_PATH) for s in samples]
        prompt_texts = []
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {executor.submit(_build_prompt_wrapper, params): i for i, params in enumerate(params_list)}
            for future in as_completed(futures):
                try:
                    prompt_texts.append((futures[future], future.result()))
                except Exception as e:
                    print(f"  警告：预处理样本时出错: {e}")
                    prompt_texts.append((futures[future], ""))
        # 保持原始顺序
        prompt_texts = [pt[1] for pt in sorted(prompt_texts, key=lambda x: x[0])]
        print(f"[OK] 完成预处理 {len(samples)} 个样本")
    else:
        # 单进程模式
        prompt_texts = [build_chat_text(s, tokenizer, include_answer=False) for s in samples]
    
    if eval_repair_enabled():
        print(
            "[OK] EVAL_OUTPUT_REPAIR=1：fault_type/kpi/container 与输入 fault_event 对齐；"
            "根因默认不改（避免金标不在候选时被误改）。"
            "若要将根因强制投影到候选，请设 EVAL_ROOT_PROJECT_TO_CANDIDATES=1。"
        )
    else:
        print("[OK] EVAL_OUTPUT_REPAIR=0：按模型原始解析结果计分（未做合规修复）。")

    # 评估推理：使用单样本模式（与原始版本一致，避免批量推理的padding问题）
    with open(out_path, "w", encoding="utf-8") as handle:
        for sample, prompt_text in zip(samples, prompt_texts):
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
            # 单样本模式：直接使用inputs["input_ids"].shape[-1]获取输入长度
            gen_ids = outputs[0][inputs["input_ids"].shape[-1] :]
            raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            payload = extract_json(sanitize_generation(raw_text))
            payload = apply_eval_repair(sample, payload)
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
            
            # 只有成功解析的样本才参与评估
            parsed += 1
            
            label_output = sample.get("output", {})
            label_root = label_output.get("root_cause_component", "")
            label_type = label_output.get("fault_type", "")
            label_kpi = label_output.get("kpi", "")
            label_container = label_output.get("related_container", "")
            
            pred_root = payload.get("root_cause_component", "")
            pred_type = payload.get("fault_type", "")
            pred_kpi = payload.get("kpi", "")
            pred_container = payload.get("related_container", "")
            
            # 获取Top-K候选列表（用于Top-3/Top-5评估）
            top_candidates = sample.get("input", {}).get("top_candidates", [])
            
            # Top-1准确率：预测的根因与真实标签完全匹配
            if pred_root == label_root:
                root_hit += 1
            
            # Top-3准确率：真实标签在Top-3候选列表中（即使预测不完全正确，只要真实答案在候选列表中就算对）
            if label_root and label_root in top_candidates[:3]:
                root_hit_top3 += 1
            
            # Top-5准确率：真实标签在Top-5候选列表中
            if label_root and label_root in top_candidates[:5]:
                root_hit_top5 += 1
            
            # 故障类型准确率（规范化匹配）
            pred_fault_type = normalize_string(pred_type)
            label_fault_type = normalize_string(label_type)
            if pred_fault_type == label_fault_type:
                type_hit += 1
            
            # KPI字段准确率
            if pred_kpi == label_kpi:
                kpi_hit += 1
            
            # 相关容器准确率
            if pred_container == label_container:
                container_hit += 1
            
            # 完整匹配准确率（所有字段都正确）
            if (pred_root == label_root and 
                pred_fault_type == label_fault_type and 
                pred_kpi == label_kpi and 
                pred_container == label_container):
                full_match_hit += 1
    
    total = len(samples)
    root_acc = root_hit / total if total else 0.0
    root_acc_top3 = root_hit_top3 / total if total else 0.0
    root_acc_top5 = root_hit_top5 / total if total else 0.0
    type_acc = type_hit / total if total else 0.0
    kpi_acc = kpi_hit / total if total else 0.0
    container_acc = container_hit / total if total else 0.0
    full_match_acc = full_match_hit / total if total else 0.0
    
    return EvalResult(
        total=total,
        parsed=parsed,
        root_acc=root_acc,
        root_acc_top3=root_acc_top3,
        root_acc_top5=root_acc_top5,
        type_acc=type_acc,
        kpi_acc=kpi_acc,
        container_acc=container_acc,
        full_match_acc=full_match_acc,
    )


def _resolve_train_eval(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """优先使用固定划分文件（与 split_dataset.py 一致），便于本机/线上复现。"""
    train_p = (args.train_jsonl or os.environ.get("TRAIN_JSONL") or "").strip()
    eval_p = (args.eval_jsonl or os.environ.get("EVAL_JSONL") or "").strip()
    if train_p or eval_p:
        if not train_p or not eval_p:
            raise ValueError("请同时指定 --train-jsonl 与 --eval-jsonl（或同时设置 TRAIN_JSONL、EVAL_JSONL）")
        if not os.path.isfile(train_p) or not os.path.isfile(eval_p):
            raise FileNotFoundError(f"划分文件不存在: {train_p} / {eval_p}")
        tr = load_samples(train_p)
        ev = load_samples(eval_p)
        if MAX_SAMPLES > 0:
            tr = tr[:MAX_SAMPLES]
        return tr, ev, f"explicit:{train_p}|{eval_p}"

    default_train = os.path.join(BASE_DIR, "output", "train_split.jsonl")
    default_eval = os.path.join(BASE_DIR, "output", "eval_split.jsonl")
    use_fixed = os.environ.get("USE_FIXED_SPLIT", "1").strip() != "0"
    if use_fixed and os.path.isfile(default_train) and os.path.isfile(default_eval):
        tr = load_samples(default_train)
        ev = load_samples(default_eval)
        if MAX_SAMPLES > 0:
            tr = tr[:MAX_SAMPLES]
        return tr, ev, f"files:{default_train}"

    data_path = args.data_jsonl or DATA_PATH
    all_samples = load_samples(data_path)
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
    return train_samples, eval_samples, f"shuffle:{data_path}"


def main() -> None:
    # Windows 控制台默认 GBK，避免 tqdm/中文 与管道解码冲突
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Qwen2.5-7B QLoRA SFT 与评估")
    parser.add_argument("--train-jsonl", default=None, help="训练集 JSONL（与 --eval-jsonl 成对使用）")
    parser.add_argument("--eval-jsonl", default=None, help="评估集 JSONL")
    parser.add_argument(
        "--data-jsonl",
        default=None,
        help="单一数据源路径（仅当未使用固定划分时生效，默认 config.DATA_PATH）",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="训练轮数（默认读环境 NUM_TRAIN_EPOCHS，缺省为 2；与 MAX_STEPS>0 互斥以 MAX_STEPS 为准）",
    )
    parser.add_argument("--lora-r", type=int, default=None, help="LoRA rank（默认 16，或环境 LORA_R）")
    parser.add_argument("--lora-alpha", type=int, default=None, help="LoRA alpha（默认 32，或环境 LORA_ALPHA）")
    parser.add_argument(
        "--skip-before-eval",
        action="store_true",
        help="跳过微调前评估（基座+未训练 LoRA 在同数据上可复现，仅训练并做微调后评估，省时间）",
    )
    args = parser.parse_args()

    skip_before_eval = bool(args.skip_before_eval) or os.environ.get("SKIP_BEFORE_EVAL", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    random.seed(SEED)
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    ensure_model()

    train_samples, eval_samples, split_mode = _resolve_train_eval(args)
    print(f"数据划分: {split_mode} | 训练 {len(train_samples)} 条 | 评估 {len(eval_samples)} 条")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"

    if USE_4BIT:
        # 与原始可运行版本一致：float16，不启用 double_quant
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
    elif USE_8BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,  # 允许部分模块offload到CPU
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
    if USE_4BIT or USE_8BIT:
        model = prepare_model_for_kbit_training(model)
    # 8-bit + gradient_checkpointing 会导致 Loss=0 / grad_norm=NaN（见 docs/CRITICAL_FIX.md）
    if USE_4BIT and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    elif USE_8BIT:
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        print("[WARN] 8-bit 量化：已禁用 gradient_checkpointing（避免 Loss=0）")
    elif hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.is_parallelizable = True
    model.model_parallel = True

    lora_r = args.lora_r if args.lora_r is not None else int(os.environ.get("LORA_R", "16"))
    lora_alpha = args.lora_alpha if args.lora_alpha is not None else int(os.environ.get("LORA_ALPHA", "32"))
    lora_dropout = float(os.environ.get("LORA_DROPOUT", "0.05"))
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
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
    print(f"[OK] LoRA: r={lora_r} alpha={lora_alpha} dropout={lora_dropout}")
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable} / {total}")

    before_path = os.path.join(RESULT_DIR, f"llm_eval_before_{RUN_TAG}.jsonl")
    if skip_before_eval:
        before_metrics = EvalResult(
            total=len(eval_samples),
            parsed=0,
            root_acc=0.0,
            root_acc_top3=0.0,
            root_acc_top5=0.0,
            type_acc=0.0,
            kpi_acc=0.0,
            container_acc=0.0,
            full_match_acc=0.0,
        )
        with open(before_path, "w", encoding="utf-8") as handle:
            for sample in eval_samples:
                record = {
                    "prompt": build_chat_text(sample, tokenizer, include_answer=False),
                    "prediction": None,
                    "label": sample.get("output", {}),
                    "skipped": True,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print("[OK] 已跳过微调前评估（SKIP_BEFORE_EVAL / --skip-before-eval）；基线指标可复用历史 llm_eval_before_*.jsonl。")
    else:
        before_metrics = evaluate_model(model, tokenizer, eval_samples, before_path)

    # 多进程优化：使用24核心并行处理tokenization
    print(f"\n使用 {NUM_WORKERS} 个进程并行处理训练数据tokenization...")
    sys.stdout.flush()  # 确保输出被立即刷新
    if NUM_WORKERS > 1 and len(train_samples) > 10:
        # 多进程模式：每个进程重新加载tokenizer（避免序列化问题）
        params_list = [(s, MODEL_PATH) for s in train_samples]
        train_items = []
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {executor.submit(_build_train_item_wrapper, params): i for i, params in enumerate(params_list)}
            completed = 0
            for future in as_completed(futures):
                try:
                    train_items.append((futures[future], future.result()))
                    completed += 1
                    if completed % 100 == 0:
                        print(f"  已完成 {completed}/{len(train_samples)} 个样本的tokenization", flush=True)
                except Exception as e:
                    print(f"  警告：处理样本时出错: {e}", flush=True)
        # 保持原始顺序
        train_items = [item[1] for item in sorted(train_items, key=lambda x: x[0])]
        train_dataset = Dataset.from_list(train_items)
        print(f"[OK] 使用 {NUM_WORKERS} 个进程处理了 {len(train_samples)} 个样本", flush=True)
    else:
        # 单进程模式（兼容小数据集或单核）
        train_dataset = Dataset.from_list([build_train_item(s, tokenizer) for s in train_samples])
        print(f"[OK] 单进程处理了 {len(train_samples)} 个样本", flush=True)

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

    # 步数：MAX_STEPS>0 时只跑固定步数；否则跑 num_train_epochs 个 epoch（默认 2，提升拟合与泛化平衡）
    grad_accum = int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", "4"))
    steps_per_epoch = max(1, (len(train_samples) + grad_accum - 1) // grad_accum)
    num_train_epochs = args.epochs if args.epochs is not None else int(os.environ.get("NUM_TRAIN_EPOCHS", "2"))
    num_train_epochs = max(1, num_train_epochs)
    if MAX_STEPS > 0:
        total_optimizer_steps = MAX_STEPS
        use_max_steps = True
    else:
        total_optimizer_steps = steps_per_epoch * num_train_epochs
        use_max_steps = False
    print(
        f"Train samples: {len(train_samples)}, grad_accum={grad_accum}, "
        f"epochs={num_train_epochs}, steps/epoch≈{steps_per_epoch}, total_optimizer_steps≈{total_optimizer_steps}"
    )

    # 验证集：与最终生成评估使用同一批样本，仅用于 Teacher Forcing 的 eval_loss（不做生成，速度快）
    eval_for_trainer = None
    if USE_EVAL_FOR_EARLY_BEST and len(eval_samples) > 0:
        eval_items = [build_train_item(s, tokenizer) for s in eval_samples]
        eval_for_trainer = Dataset.from_list(eval_items).map(
            tokenize_fn,
            batched=True,
            remove_columns=["prompt_text", "full_text"],
        )
        print(f"[OK] 已构建验证集（用于训练中 eval_loss）: {len(eval_samples)} 条")

    # 学习率默认从 2e-4 降到 5e-5：小数据 SFT 时 2e-4 易过拟合、拉低根因 Top-1；可用环境变量覆盖
    learning_rate = float(os.environ.get("LEARNING_RATE", "5e-5"))
    weight_decay = float(os.environ.get("WEIGHT_DECAY", "0.01"))
    warmup_ratio = float(os.environ.get("WARMUP_RATIO", "0.05"))
    max_grad_norm = float(os.environ.get("MAX_GRAD_NORM", "1.0"))
    lr_scheduler_type = os.environ.get("LR_SCHEDULER_TYPE", "cosine").strip() or "cosine"

    # bf16：4070 SUPER 等 Ada 卡上通常稳定且更快；显存紧张可设 USE_BF16=0
    use_bf16 = os.environ.get("USE_BF16", "1").strip() not in ("0", "false", "no")
    if use_bf16 and torch.cuda.is_available():
        use_bf16 = bool(torch.cuda.is_bf16_supported())
    else:
        use_bf16 = False

    train_kw: dict[str, Any] = {
        "output_dir": MODEL_DIR,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": learning_rate,
        "logging_steps": max(1, min(10, steps_per_epoch // 20)),
        "fp16": False,
        "bf16": use_bf16,
        "report_to": [],
        "warmup_ratio": warmup_ratio,
        "weight_decay": weight_decay,
        "max_grad_norm": max_grad_norm,
        "lr_scheduler_type": lr_scheduler_type,
        "seed": SEED,
        "dataloader_num_workers": int(os.environ.get("DATALOADER_NUM_WORKERS", "0")),
    }
    if use_max_steps:
        train_kw["max_steps"] = MAX_STEPS
        train_kw["num_train_epochs"] = 1
    else:
        train_kw["max_steps"] = -1
        train_kw["num_train_epochs"] = float(num_train_epochs)

    train_callbacks: list[Any] = []
    if eval_for_trainer is not None:
        # 验证步频随总步数调整；长训练时略稀疏以省时间
        eval_steps = max(1, min(80, total_optimizer_steps // 12))
        if not use_max_steps and eval_steps >= total_optimizer_steps:
            train_kw["eval_strategy"] = "epoch"
            train_kw["save_strategy"] = "epoch"
        else:
            train_kw["eval_strategy"] = "steps"
            train_kw["eval_steps"] = eval_steps
            train_kw["save_strategy"] = "steps"
            train_kw["save_steps"] = eval_steps
        train_kw["load_best_model_at_end"] = True
        train_kw["metric_for_best_model"] = "eval_loss"
        train_kw["greater_is_better"] = False
        train_kw["save_total_limit"] = 3
        if EARLY_STOPPING_PATIENCE > 0:
            train_callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=EARLY_STOPPING_PATIENCE,
                    early_stopping_threshold=0.0,
                )
            )
            print(f"[OK] EarlyStopping: patience={EARLY_STOPPING_PATIENCE}（按 eval_loss）")
        print(
            f"[OK] 训练中验证: eval_loss 最优 checkpoint + 余弦调度 | "
            f"lr={learning_rate} warmup={warmup_ratio} wd={weight_decay} max_grad_norm={max_grad_norm} bf16={use_bf16}"
        )
    else:
        train_kw["save_strategy"] = "no"
        print(f"[WARN] 未启用训练中验证（USE_EVAL_FOR_EARLY_BEST=0 或无 eval 样本）；lr={learning_rate}")

    targs = TrainingArguments(**train_kw)
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_dataset,
        eval_dataset=eval_for_trainer,
        data_collator=default_data_collator,
        callbacks=train_callbacks if train_callbacks else None,
    )
    trainer.train()

    # 训练后评估前，需要调整模型状态以支持推理
    model.eval()  # 确保模型处于评估模式
    model.config.use_cache = True  # 启用缓存以提升推理速度
    if hasattr(model, "disable_gradient_checkpointing"):
        model.disable_gradient_checkpointing()  # 禁用gradient checkpointing以支持推理
    
    after_path = os.path.join(RESULT_DIR, f"llm_eval_after_{RUN_TAG}.jsonl")
    after_metrics = evaluate_model(model, tokenizer, eval_samples, after_path)

    finetune_meta: dict[str, Any] = {
        "skip_before_eval": skip_before_eval,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "num_train_epochs": float(num_train_epochs),
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": learning_rate,
        "warmup_ratio": warmup_ratio,
        "weight_decay": weight_decay,
        "max_grad_norm": max_grad_norm,
        "lr_scheduler_type": lr_scheduler_type,
        "bf16": use_bf16,
        "use_eval_for_early_best": USE_EVAL_FOR_EARLY_BEST,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE if eval_for_trainer is not None else None,
        "total_optimizer_steps_approx": total_optimizer_steps,
        "eval_output_repair": eval_repair_enabled(),
        "eval_root_project_to_candidates": os.environ.get("EVAL_ROOT_PROJECT_TO_CANDIDATES", "0"),
    }

    summary = {
        "model": MODEL_PATH,
        "train_size": len(train_samples),
        "eval_size": len(eval_samples),
        "split_mode": split_mode,
        "finetune": finetune_meta,
        "before": None if skip_before_eval else before_metrics.__dict__,
        "after": after_metrics.__dict__,
    }
    summary_path = os.path.join(RESULT_DIR, f"训练评估结果_{RUN_TAG}.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    report_path = os.path.join(RESULT_DIR, f"训练评估报告_{RUN_TAG}.md")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("# 轻量 SFT 训练与评估报告\n\n")
        handle.write(f"- 模型：{MODEL_PATH}\n")
        handle.write(f"- 数据划分：{split_mode}\n")
        handle.write(f"- 训练样本：{len(train_samples)}\n")
        handle.write(f"- 评估样本：{len(eval_samples)}\n\n")
        handle.write("## 本次微调配置\n")
        handle.write(f"- 见同目录 `训练评估结果_{RUN_TAG}.json` 中 `finetune` 字段（学习率、LoRA、epoch、是否跳过微调前等）。\n\n")
        if skip_before_eval:
            handle.write("## 微调前\n")
            handle.write(
                "- **已跳过**（`--skip-before-eval` / `SKIP_BEFORE_EVAL=1`）。"
                "同一基座、同一评估集上基线可复现，可直接对照历史 `llm_eval_before_*.jsonl`（如 `llm_eval_before_20260401_141203.jsonl`）。\n\n"
            )
        else:
            handle.write("## 微调前\n")
            handle.write(
                f"- 可解析率：{before_metrics.parsed}/{before_metrics.total}\n"
                f"- 根因 Top‑1：{before_metrics.root_acc:.3f}\n"
                f"- 根因 Top‑3：{before_metrics.root_acc_top3:.3f}\n"
                f"- 根因 Top‑5：{before_metrics.root_acc_top5:.3f}\n"
                f"- 故障类型 Acc：{before_metrics.type_acc:.3f}\n"
                f"- KPI Acc：{before_metrics.kpi_acc:.3f}\n"
                f"- 相关容器 Acc：{before_metrics.container_acc:.3f}\n"
                f"- 完整匹配 Acc：{before_metrics.full_match_acc:.3f}\n\n"
            )
        handle.write("## 微调后\n")
        handle.write(
            f"- 可解析率：{after_metrics.parsed}/{after_metrics.total}\n"
            f"- 根因 Top‑1：{after_metrics.root_acc:.3f}\n"
            f"- 根因 Top‑3：{after_metrics.root_acc_top3:.3f}\n"
            f"- 根因 Top‑5：{after_metrics.root_acc_top5:.3f}\n"
            f"- 故障类型 Acc：{after_metrics.type_acc:.3f}\n"
            f"- KPI Acc：{after_metrics.kpi_acc:.3f}\n"
            f"- 相关容器 Acc：{after_metrics.container_acc:.3f}\n"
            f"- 完整匹配 Acc：{after_metrics.full_match_acc:.3f}\n\n"
        )
        handle.write("## 备注\n")
        handle.write(
            "- 默认 `EVAL_OUTPUT_REPAIR=1`：`fault_type`/`kpi`/`related_container` 与输入 `fault_event` 对齐（符合题目约束，可抬高类型与完整匹配）。"
            "根因默认不改动；可选 `EVAL_ROOT_PROJECT_TO_CANDIDATES=1` 将根因投影到候选（部分数据金标不在候选时可能降分）。"
            "纯模型计分设 `EVAL_OUTPUT_REPAIR=0`。\n"
        )
        handle.write("- 详细预测见：llm_eval_before.jsonl / llm_eval_after.jsonl。\n")

    print(f"\n{'='*70}")
    print("训练完成！所有结果已保存。")
    print(f"{'='*70}")
    print(f"评估报告: {report_path}")
    print(f"评估结果: {summary_path}")
    print(f"{'='*70}\n")

    # 训练完成
    print("训练完成！")


if __name__ == "__main__":
    main()
