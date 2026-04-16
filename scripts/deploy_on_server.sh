#!/bin/bash
# AutoDL 服务器端部署脚本
# 使用方法: bash deploy_on_server.sh

set -e

echo "========================================"
echo "AutoDL 服务器部署脚本"
echo "========================================"

# 1. 解压上传的文件
if [ -f "/root/Graduation_Project_upload.zip" ]; then
    echo "[1/4] 解压上传文件..."
    cd /root
    unzip -q -o Graduation_Project_upload.zip
    
    # 移动文件到正确位置
    if [ -d "Graduation_Project" ]; then
        echo "[2/4] 整理文件结构..."
        
        # 创建数据盘目录
        mkdir -p /root/autodl-tmp/datasets
        mkdir -p /root/autodl-tmp/output
        mkdir -p /root/autodl-tmp/models
        mkdir -p /root/autodl-tmp/hf_cache
        
        # 移动数据集到数据盘
        if [ -d "data" ]; then
            mv data /root/autodl-tmp/datasets/aiops2020_raw
            echo "  ✓ 数据集已移动到 /root/autodl-tmp/datasets/aiops2020_raw"
        fi
        
        # 移动输出文件到数据盘
        if [ -d "output" ]; then
            mv output /root/autodl-tmp/output
            echo "  ✓ 输出文件已移动到 /root/autodl-tmp/output"
        fi
        
        # 移动模型到数据盘
        if [ -d "models" ]; then
            mv models/* /root/autodl-tmp/models/
            rm -rf models
            echo "  ✓ 模型文件已移动到 /root/autodl-tmp/models"
        fi
        
        # 代码保留在系统盘
        if [ ! -d "/root/Graduation_Project" ]; then
            mv Graduation_Project /root/
        else
            # 如果已存在，合并文件
            cp -r Graduation_Project/* /root/Graduation_Project/
            rm -rf Graduation_Project
        fi
        echo "  ✓ 代码文件已整理到 /root/Graduation_Project"
    fi
    
    # 清理压缩包
    rm -f /root/Graduation_Project_upload.zip
    echo "[3/4] 清理临时文件完成"
else
    echo "警告: 未找到 /root/Graduation_Project_upload.zip"
    echo "请先上传文件到服务器"
    exit 1
fi

# 2. 设置 HuggingFace 缓存路径
echo "[4/4] 配置环境变量..."
cat >> ~/.bashrc << 'EOF'
# Graduation Project 环境变量
export HF_HOME=/root/autodl-tmp/hf_cache
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache
EOF

source ~/.bashrc

echo ""
echo "========================================"
echo "部署完成！"
echo "========================================"
echo "代码目录: /root/Graduation_Project"
echo "数据集: /root/autodl-tmp/datasets/aiops2020_raw"
echo "输出目录: /root/autodl-tmp/output"
echo "模型目录: /root/autodl-tmp/models/Qwen2.5-7B-Instruct"
echo ""
echo "下一步:"
echo "1. cd /root/Graduation_Project"
echo "2. 创建虚拟环境: conda create -n gp-rca python=3.10 -y"
echo "3. conda activate gp-rca"
echo "4. pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128"
echo "5. pip install transformers peft accelerate bitsandbytes datasets einops sentencepiece"
echo "6. python pipeline/train_qwen2_5_7b_qlora_demo.py"
echo "========================================"
