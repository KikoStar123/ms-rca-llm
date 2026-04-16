#!/usr/bin/env bash
# AutoDL：从 GitHub 拉代码后的推荐步骤（仓库名可改）
# 用法：bash scripts/autodl_from_git.sh [目标目录]
set -e
TARGET="${1:-/root/autodl-tmp/ms-rca-llm}"
REPO_URL="${MS_RCA_REPO_URL:-https://github.com/KikoStar123/ms-rca-llm.git}"

echo "目标目录: $TARGET"
mkdir -p "$(dirname "$TARGET")"
if [ -d "$TARGET/.git" ]; then
  echo "已存在仓库，执行 git pull..."
  git -C "$TARGET" pull --ff-only
else
  echo "克隆 $REPO_URL ..."
  git clone "$REPO_URL" "$TARGET"
fi

echo ""
echo "下一步（在实例内）："
echo "  cd $TARGET"
echo "  # 建议：export GRADUATION_PROJECT_ROOT=$TARGET"
echo "  pip install -r requirements-train.txt"
echo "  # 按 README 安装与你的 CUDA 版本匹配的 torch"
echo "  # 模型放 HF 缓存或设 QWEN_MODEL_PATH，数据见 README 说明"
echo "  python pipeline/large_model/train_qwen2_5_7b_qlora_demo.py"
