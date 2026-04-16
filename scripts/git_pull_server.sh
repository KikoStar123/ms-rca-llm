#!/usr/bin/env bash
# AutoDL / Linux：进入仓库根目录后拉取最新代码
# 用法：
#   export GRADUATION_PROJECT_ROOT=/root/autodl-tmp/ms-rca-llm   # 推荐，与 config.py 一致
#   bash scripts/git_pull_server.sh
#   bash scripts/git_pull_server.sh main
set -euo pipefail

BRANCH="${1:-main}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${GRADUATION_PROJECT_ROOT:-}" ]]; then
  REPO_ROOT="${GRADUATION_PROJECT_ROOT}"
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

cd "$REPO_ROOT"
if [[ ! -d .git ]]; then
  echo "错误: 不是 git 仓库: $REPO_ROOT" >&2
  echo "请设置 GRADUATION_PROJECT_ROOT 为克隆目录，或在本仓库根目录执行。" >&2
  exit 1
fi

echo "仓库: $REPO_ROOT"
git fetch origin
git pull --ff-only origin "$BRANCH"
echo ""
echo "拉取完成。"
echo "---"
echo "大文件（未进 Git）请放到与 config.py 一致的路径："
echo "  data/   -> ${REPO_ROOT}/data/aiops2020/   （或 ${REPO_ROOT}/data/ 下你的子目录，与本地一致）"
echo "  models/ -> 任选其一："
echo "           A) ${REPO_ROOT}/models/Qwen2.5-7B-Instruct/"
echo "           B) /root/autodl-tmp/hf_cache/Qwen2.5-7B-Instruct/  且 export QWEN_MODEL_PATH=该路径"
echo "若仓库不在默认 /root/autodl-tmp/Graduation_Project，请务必: export GRADUATION_PROJECT_ROOT=$REPO_ROOT"
