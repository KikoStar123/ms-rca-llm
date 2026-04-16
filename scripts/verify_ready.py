#!/usr/bin/env python3
"""
环境就绪验证：--full 检查 GPU 训练依赖与模型；--local 仅检查仓库、数据与脚本（本机无显卡也可用）。
"""
from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def check_config() -> bool:
    print("[1] 配置文件...")
    try:
        sys.path.insert(0, _REPO_ROOT)
        import config

        print(f"  BASE_DIR: {config.BASE_DIR}")
        print(f"  MODEL_PATH: {config.MODEL_PATH}")
        print(f"  DATA_PATH: {config.DATA_PATH}")
        return True
    except Exception as e:
        print(f"  失败: {e}")
        return False


def check_packages_full() -> bool:
    print("[2] Python 包（训练必需）...")
    required = {
        "torch": "PyTorch",
        "transformers": "Transformers",
        "peft": "PEFT",
        "bitsandbytes": "BitsAndBytes",
        "accelerate": "Accelerate",
        "datasets": "Datasets",
    }
    ok = True
    for pkg, name in required.items():
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "unknown")
            print(f"  OK {name}: {ver}")
        except ImportError:
            print(f"  缺失: {name}")
            ok = False
    return ok


def check_model_dir(model_path: str) -> bool:
    print("[3] 模型目录...")
    if not os.path.exists(model_path):
        print(f"  不存在: {model_path}")
        return False
    print(f"  存在: {model_path}")
    key_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    for f in key_files:
        p = os.path.join(model_path, f)
        if os.path.exists(p):
            print(f"  OK {f}")
        else:
            print(f"  缺少: {f}")
    st = [f for f in os.listdir(model_path) if f.endswith(".safetensors")]
    if st:
        total = sum(os.path.getsize(os.path.join(model_path, f)) for f in st)
        print(f"  权重分片: {len(st)} 个, ~{total / 1024**3:.2f} GB")
    else:
        print("  未找到 .safetensors（线上请确认已下载完整 Qwen）")
        return False
    return True


def check_data(data_path: str) -> bool:
    print("[4] 训练数据 llm_inputs...")
    if not os.path.exists(data_path):
        print(f"  不存在: {data_path}")
        return False
    with open(data_path, encoding="utf-8") as f:
        n = sum(1 for _ in f)
    print(f"  OK {data_path}（约 {n} 行）")
    return True


def check_scripts() -> bool:
    print("[5] 训练脚本路径...")
    train_rel = os.path.join("pipeline", "large_model", "train_qwen2_5_7b_qlora_demo.py")
    p = os.path.join(_REPO_ROOT, train_rel)
    if os.path.exists(p):
        print(f"  OK {train_rel}")
        return True
    print(f"  不存在: {p}")
    return False


def check_cuda_optional() -> None:
    print("[额外] CUDA...")
    try:
        import torch

        if torch.cuda.is_available():
            print(f"  OK GPU 数量: {torch.cuda.device_count()}")
        else:
            print("  不可用（本机可继续跑数据/划分；训练请用线上 GPU）")
    except ImportError:
        print("  未安装 torch（--local 模式可忽略）")


def main() -> None:
    parser = argparse.ArgumentParser(description="环境就绪检查")
    parser.add_argument(
        "--local",
        action="store_true",
        help="仅检查仓库、数据与脚本，不要求 torch/模型（适合本机准备数据）",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="检查训练依赖与模型（默认：若未指定 --local 则等价于完整检查）",
    )
    args = parser.parse_args()
    local_only = args.local
    full = args.full or not local_only

    print("=" * 60)
    print("环境就绪验证" + ("（本地轻量）" if local_only else "（完整/训练）"))
    print("=" * 60)

    if not check_config():
        sys.exit(1)

    import config

    if local_only:
        check_data(config.DATA_PATH)
        check_scripts()
        dirs = [config.OUTPUT_DIR, config.RESULT_DIR]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
            print(f"[目录] OK {d}")
        print("\n本机可执行示例 (PowerShell):")
        print(f'  $env:GRADUATION_PROJECT_ROOT = "{_REPO_ROOT}"')
        print("  . .\\scripts\\local_env.ps1")
        print("  python pipeline/run_rca_pipeline.py split")
        print("  python pipeline/inference_cli.py --mock --input research/demo_inference_sft_sample.json")
        print("\n线上训练（AutoDL）保持原流程；上传前可设置 GRADUATION_PROJECT_ROOT 与线上一致。")
        check_cuda_optional()
        print("=" * 60)
        return

    if not check_packages_full():
        sys.exit(1)
    if not check_model_dir(config.MODEL_PATH):
        sys.exit(1)
    check_data(config.DATA_PATH)
    if not check_scripts():
        sys.exit(1)

    for d in (config.OUTPUT_DIR, config.RESULT_DIR):
        os.makedirs(d, exist_ok=True)
    if hasattr(config, "MODEL_CKPT_DIR"):
        os.makedirs(config.MODEL_CKPT_DIR, exist_ok=True)

    check_cuda_optional()
    print("=" * 60)
    print("完整检查结束。可运行: python pipeline/large_model/train_qwen2_5_7b_qlora_demo.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
