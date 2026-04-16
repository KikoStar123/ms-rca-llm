#!/usr/bin/env bash
# 对照实验：epoch=2，与 20260412 配置对齐。用法：bash scripts/train_opt_followup.sh
set -euo pipefail
ROOT="${GRADUATION_PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"

export NUM_TRAIN_EPOCHS=2
export LEARNING_RATE=3e-5
export LORA_R=24
export LORA_ALPHA=48
export LORA_DROPOUT=0.08
export GRADIENT_ACCUMULATION_STEPS=4
export WARMUP_RATIO=0.08
export WEIGHT_DECAY=0.01
export MAX_GRAD_NORM=1.0
export LR_SCHEDULER_TYPE=cosine
export USE_BF16=1
export USE_EVAL_FOR_EARLY_BEST=1
export EARLY_STOPPING_PATIENCE=4
export SKIP_BEFORE_EVAL=1
export NUM_WORKERS="${NUM_WORKERS:-8}"

echo "ROOT=$ROOT"
echo "NUM_TRAIN_EPOCHS=$NUM_TRAIN_EPOCHS LEARNING_RATE=$LEARNING_RATE LORA ${LORA_R}/${LORA_ALPHA}"

python pipeline/large_model/train_qwen2_5_7b_qlora_demo.py --skip-before-eval --epochs 2
