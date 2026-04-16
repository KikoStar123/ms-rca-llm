"""
在 SFT（QLoRA）checkpoint 上继续 GRPO，奖励为 `grpo_rewards.reward_rca`（与 SFT 任务一致）。

依赖：pip install "trl>=0.15"（与 transformers 5.x 配套）。
Windows：若 `import trl` 报 GBK 解码错误，请先设置环境变量 PYTHONUTF8=1，或在本文件最开头已 `setdefault`。

用法示例（单卡，与 train_qwen 相同数据划分）::

    set PYTHONUTF8=1
    set ADAPTER_PATH=v2_doc\\model_ckpt\\checkpoint-XXX
    python pipeline/large_model/train_grpo_rca.py --train-jsonl output/train_split.jsonl --output-dir v2_doc/model_ckpt_grpo_run1

训练结束后用现有 evaluate_predictions_jsonl / train 脚本中的评估流程在 eval_split 上对比。
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTHONUTF8", "1")

import argparse
import json
import sys
from datetime import datetime

import torch
from datasets import Dataset
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.trainer_utils import get_last_checkpoint

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PIPELINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LARGE_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _PIPELINE_DIR)
sys.path.insert(0, _LARGE_MODEL_DIR)

try:
    import config

    DEFAULT_BASE = config.MODEL_PATH
    DEFAULT_OUTPUT_ROOT = config.RESULT_DIR
    DEFAULT_TRAIN = os.path.join(config.OUTPUT_DIR, "train_split.jsonl")
except ImportError:
    DEFAULT_BASE = os.environ.get("QWEN_MODEL_PATH", "")
    DEFAULT_OUTPUT_ROOT = os.path.join(_REPO_ROOT, "v2_doc")
    DEFAULT_TRAIN = os.path.join(_REPO_ROOT, "output", "train_split.jsonl")

from grpo_rewards import reward_rca_trl  # noqa: E402

from train_qwen2_5_7b_qlora_demo import build_chat_text, load_samples  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RCA GRPO（SFT LoRA 之后第二阶段）")
    p.add_argument("--base-model", default=None, help="基座模型目录（默认 config.MODEL_PATH / QWEN_MODEL_PATH）")
    p.add_argument(
        "--adapter-path",
        default=os.environ.get("ADAPTER_PATH", "").strip() or None,
        help="SFT 产出的 LoRA 目录（adapter_config.json + 权重），或环境 ADAPTER_PATH",
    )
    p.add_argument("--train-jsonl", default=DEFAULT_TRAIN, help="训练 JSONL（与 SFT 同格式）")
    p.add_argument(
        "--output-dir",
        default=None,
        help="GRPO 输出目录（默认 v2_doc/model_ckpt_grpo_<时间戳>）",
    )
    p.add_argument("--max-samples", type=int, default=0, help=">0 时只用前 N 条（试跑）")
    p.add_argument("--seed", type=int, default=42)
    # GRPO / 训练
    p.add_argument("--num-generations", type=int, default=4, help="每组采样数（须整除 generation_batch_size，默认与 grad_accum 配合为 4）")
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--max-completion-length", type=int, default=320)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--num-train-epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1, help=">0 时优先于 num_train_epochs")
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument(
        "--fp16",
        action="store_true",
        help="强制 AMP fp16（默认在支持的 GPU 上用 bf16，避免 fp16 GradScaler 与 BF16 权重组合报错）",
    )
    return p.parse_args()


def _pick_amp_flags(args: argparse.Namespace) -> tuple[bool, bool]:
    """返回 (fp16, bf16)。Ada/40 系默认 bf16；显式 --fp16 时用 fp16。"""
    if args.fp16:
        return True, False
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return False, True
    return True, False


def _load_model_and_tokenizer(base_model: str, adapter_path: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"
    return model, tokenizer


def _build_grpo_dataset(tokenizer, samples: list) -> Dataset:
    rows = []
    for s in samples:
        prompt = build_chat_text(s, tokenizer, include_answer=False)
        out = s.get("output", {})
        rows.append(
            {
                "prompt": prompt,
                "gold_label": json.dumps(out, ensure_ascii=False),
            }
        )
    return Dataset.from_list(rows)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    args = _parse_args()
    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as e:
        raise SystemExit(
            "未安装 trl 或导入失败。请执行: pip install \"trl>=0.15\"\n"
            "Windows 若仍报错，请设置环境变量 PYTHONUTF8=1 后再运行。\n"
            f"原始错误: {e}"
        ) from e

    base_model = (args.base_model or DEFAULT_BASE or "").strip()
    if not base_model or not os.path.isdir(base_model):
        raise SystemExit(f"基座模型路径无效：{base_model!r}，请设置 --base-model 或 QWEN_MODEL_PATH。")
    adapter_path = (args.adapter_path or "").strip()
    if not adapter_path or not os.path.isdir(adapter_path):
        raise SystemExit(
            "请指定 SFT LoRA 目录：--adapter-path 或环境变量 ADAPTER_PATH（需含 adapter_config.json）。"
        )

    train_path = args.train_jsonl
    if not os.path.isfile(train_path):
        raise SystemExit(f"训练集不存在：{train_path}")

    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (args.output_dir or "").strip() or os.path.join(DEFAULT_OUTPUT_ROOT, f"model_ckpt_grpo_{tag}")
    os.makedirs(output_dir, exist_ok=True)

    samples = load_samples(train_path)
    if args.max_samples and args.max_samples > 0:
        samples = samples[: args.max_samples]

    print(f"[GRPO] base_model={base_model}\n[GRPO] adapter_path={adapter_path}\n[GRPO] samples={len(samples)}\n[GRPO] output_dir={output_dir}")

    model, tokenizer = _load_model_and_tokenizer(base_model, adapter_path)
    train_ds = _build_grpo_dataset(tokenizer, samples)

    # generation_batch_size = per_device * world * steps_per_generation；默认 steps_per_generation = grad_accum
    # 须满足 generation_batch_size % num_generations == 0
    num_gen = max(2, int(args.num_generations))
    per_device = int(args.per_device_train_batch_size)
    grad_accum = int(args.gradient_accumulation_steps)
    gen_bs = per_device * 1 * grad_accum  # world_size 1
    if gen_bs % num_gen != 0:
        raise SystemExit(
            f"请调整参数使 (per_device_train_batch_size * gradient_accumulation_steps) % num_generations == 0；"
            f"当前 generation_batch_size={gen_bs}, num_generations={num_gen}。"
        )

    use_fp16, use_bf16 = _pick_amp_flags(args)
    print(f"[GRPO] amp: fp16={use_fp16} bf16={use_bf16}")
    gc_kwargs: dict = dict(
        output_dir=output_dir,
        seed=args.seed,
        remove_unused_columns=False,
        num_generations=num_gen,
        max_completion_length=int(args.max_completion_length),
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=float(args.num_train_epochs),
        logging_steps=int(args.logging_steps),
        save_steps=int(args.save_steps),
        save_total_limit=3,
        fp16=use_fp16,
        bf16=use_bf16,
        beta=0.0,
        report_to="none",
    )
    if args.max_steps and args.max_steps > 0:
        gc_kwargs["max_steps"] = int(args.max_steps)
    gc = GRPOConfig(**gc_kwargs)

    trainer = GRPOTrainer(
        model=model,
        args=gc,
        train_dataset=train_ds,
        processing_class=tokenizer,
        reward_funcs=reward_rca_trl,
    )

    last_ckpt = get_last_checkpoint(output_dir)
    train_result = trainer.train(resume_from_checkpoint=last_ckpt)
    print("[GRPO] train_result:", train_result)

    # 合并保存便于推理：adapter 在 output_dir
    try:
        trainer.save_model()
    except Exception as e:
        print("[GRPO] save_model 警告:", e)

    cfg_dump = gc.to_dict() if hasattr(gc, "to_dict") else {}
    with open(os.path.join(output_dir, "grpo_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "base_model": base_model,
                "sft_adapter_path": adapter_path,
                "train_jsonl": train_path,
                "num_samples": len(samples),
                "grpo_config": cfg_dump,
                "train_runtime": getattr(train_result, "metrics", None),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[GRPO] 完成。checkpoint 目录: {output_dir}")


if __name__ == "__main__":
    main()
