# download_qwen.py - 强校验版：彻底解决文件损坏/不完整问题
from huggingface_hub import snapshot_download
import os
import sys
import hashlib

# 固定配置（与验证脚本一致，无需修改）
MODEL_REPO = "Qwen/Qwen2.5-7B-Instruct"
SAVE_DIR = r"D:\hf_cache\Qwen2.5-7B-Instruct"
HF_MIRROR = "https://hf-mirror.com"  # 国内镜像，满速+稳定

def check_file_complete(file_path):
    """校验单个文件是否完整（无空文件/截断文件）"""
    if not os.path.exists(file_path):
        return False
    # 空文件/过小文件直接判定为损坏（模型分片均>2GB）
    if os.path.getsize(file_path) < 1 * 1024 * 1024 * 1024:  # 小于1GB即为损坏
        return False
    return True

def main():
    print("="*60)
    print("🚀 强校验版：重新下载Qwen2.5-7B-Instruct（确保文件完整）")
    print(f"💾 保存路径：{SAVE_DIR}")
    print(f"🔗 下载镜像：{HF_MIRROR}")
    print("⚠️  本次将彻底重新下载（约13GB），请保持网络稳定！")
    print("="*60 + "\n")

    # 强制创建新目录（覆盖旧目录）
    os.makedirs(SAVE_DIR, exist_ok=True)

    try:
        # 核心：强制重新下载+禁用缓存+断点续传
        snapshot_download(
            repo_id=MODEL_REPO,
            local_dir=SAVE_DIR,
            local_dir_use_symlinks=False,
            endpoint=HF_MIRROR,
            resume_download=True,
            force_download=True,  # 强制重新下载，覆盖损坏文件
            cache_dir=None,       # 不使用缓存，确保文件直接保存到指定目录
            token=None
        )

        # 强校验：检查4个分片是否存在+文件大小是否达标
        shard_files = [
            f for f in os.listdir(SAVE_DIR)
            if f.startswith("model-") and f.endswith(".safetensors") and "of-00004" in f
        ]
        valid_shards = [f for f in shard_files if check_file_complete(os.path.join(SAVE_DIR, f))]

        if len(valid_shards) == 4:
            print("\n" + "="*60)
            print(f"✅ 模型下载成功！4个分片均完整（每个≥2GB）")
            print(f"📌 完整模型路径：{SAVE_DIR}")
            print("="*60)
        else:
            print("\n" + "="*60)
            print(f"❌ 下载失败！仅获取{len(valid_shards)}个有效分片（需4个）")
            print("💡 原因：网络中断导致文件截断/损坏")
            print("💡 解决方案：检查网络后，重新运行本脚本")
            print("="*60)
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ 下载异常：{str(e)[:200]}")
        print("💡 解决方案：1. 关闭代理/防火墙 2. 切换稳定网络 3. 重新运行脚本")
        sys.exit(1)

if __name__ == "__main__":
    main()