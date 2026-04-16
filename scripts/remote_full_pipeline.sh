#!/usr/bin/env bash
# 仅在远程（Linux/AutoDL）执行：拉代码 → 构建 LLM 输入与划分 → 训练
# 使用前请已准备好：GPU、Python、torch、Qwen 权重、data/aiops2020、output 下 SFT 样本 jsonl
#
# 环境变量（可选）：
#   GRADUATION_PROJECT_ROOT  仓库根，未设则为本脚本上级目录
#   QWEN_MODEL_PATH          基座模型目录（建议 export）
#   SKIP_DATA_BUILD=1        跳过 build_llm_inputs + split_dataset（直接用已有 output/*.jsonl）
#   SKIP_TRAIN=1             只重建数据不训练
#   GIT_PULL=0               跳过 git pull（离线镜像时）
#   TRAIN_EXTRA_ARGS           附加到训练命令，如 "--skip-before-eval --epochs 2"
#
# 示例：
#   export GRADUATION_PROJECT_ROOT=/root/autodl-tmp/ms-rca-llm
#   export QWEN_MODEL_PATH=/root/autodl-tmp/hf_cache/Qwen2.5-7B-Instruct
#   export NUM_WORKERS=8
#   bash scripts/remote_full_pipeline.sh

set -euo pipefail

ROOT="${GRADUATION_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"

GIT_PULL="${GIT_PULL:-1}"
SKIP_DATA_BUILD="${SKIP_DATA_BUILD:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"

echo "========== 远程全流程 ROOT=$ROOT =========="

if [[ "$GIT_PULL" == "1" ]]; then
  echo "[1/4] git pull"
  git pull --ff-only origin main
else
  echo "[1/4] 跳过 git pull (GIT_PULL=0)"
fi

if [[ -z "${QWEN_MODEL_PATH:-}" ]]; then
  echo "警告: 未设置 QWEN_MODEL_PATH，训练阶段可能找不到基座模型。请 export QWEN_MODEL_PATH=..." >&2
fi
export GRADUATION_PROJECT_ROOT="$ROOT"

if [[ "$SKIP_DATA_BUILD" != "1" ]]; then
  echo "[2/4] build_llm_inputs（需 data/aiops2020 与 output/sft_samples_*.jsonl）"
  if [[ ! -d "$ROOT/data/aiops2020" ]]; then
    echo "错误: 缺少 $ROOT/data/aiops2020 ，无法构建候选。请上传 data 或设 SKIP_DATA_BUILD=1 使用已有 llm_inputs。" >&2
    exit 1
  fi
  python pipeline/build_llm_inputs.py --input-dir output --output output/llm_inputs_v4.jsonl
  echo "[3/4] split_dataset"
  python pipeline/split_dataset.py \
    --input output/llm_inputs_v4.jsonl \
    --train-out output/train_split.jsonl \
    --eval-out output/eval_split.jsonl \
    --manifest output/split_manifest.json
else
  echo "[2-3/4] 跳过数据构建 (SKIP_DATA_BUILD=1)"
fi

if [[ "$SKIP_TRAIN" == "1" ]]; then
  echo "[4/4] 跳过训练 (SKIP_TRAIN=1)"
  echo "完成（仅数据）。"
  exit 0
fi

echo "[4/4] 训练"
# shellcheck disable=SC2086
python pipeline/large_model/train_qwen2_5_7b_qlora_demo.py $TRAIN_EXTRA_ARGS

echo "========== 远程全流程结束 =========="
