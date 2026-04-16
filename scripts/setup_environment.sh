#!/bin/bash
# 服务器环境配置脚本
# 用于在 RTX 5090 服务器上配置训练环境

set -e  # 遇到错误立即退出

echo "=========================================="
echo "开始配置训练环境"
echo "=========================================="
echo ""

PROJECT_ROOT="/root/autodl-tmp/Graduation_Project"
cd "$PROJECT_ROOT"

# 1. 检查 Python 和 PyTorch
echo "[1/6] 检查基础环境..."
python3 --version
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA version: {torch.version.cuda}')"
echo "✅ 基础环境检查完成"
echo ""

# 2. 安装必要的 Python 包
echo "[2/6] 安装必要的 Python 包..."
REQUIRED_PACKAGES=(
    "transformers"
    "peft"
    "bitsandbytes"
    "accelerate"
    "datasets"
    "sentencepiece"  # Qwen tokenizer 需要
    "protobuf"       # transformers 可能需要
)

for pkg in "${REQUIRED_PACKAGES[@]}"; do
    echo "检查/安装 $pkg..."
    pip3 install -q "$pkg" || {
        echo "⚠️  $pkg 安装失败，继续..."
    }
done

echo "验证包安装..."
python3 -c "
import transformers
import peft
import bitsandbytes
import accelerate
import datasets
print('✅ 所有核心包已安装')
print(f'  transformers: {transformers.__version__}')
print(f'  peft: {peft.__version__}')
print(f'  accelerate: {accelerate.__version__}')
print(f'  datasets: {datasets.__version__}')
"
echo ""

# 3. 检查配置文件
echo "[3/6] 检查配置文件..."
if [ -f "$PROJECT_ROOT/config.py" ]; then
    echo "✅ config.py 存在"
    # 尝试导入配置
    python3 -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')
try:
    import config
    print(f'  数据路径: {config.DATA_ROOT}')
    print(f'  模型路径: {config.MODEL_ROOT}')
    print(f'  输出路径: {config.OUTPUT_ROOT}')
    print('✅ 配置文件导入成功')
except Exception as e:
    print(f'⚠️  配置文件导入失败: {e}')
    " || echo "⚠️  配置文件检查失败"
else
    echo "❌ config.py 不存在，需要创建"
fi
echo ""

# 4. 检查项目文件结构
echo "[4/6] 检查项目文件结构..."
REQUIRED_FILES=(
    "pipeline/train_qwen2_5_7b_qlora_demo.py"
    "pipeline/build_llm_inputs.py"
    "pipeline/batch_build_sft.py"
    "pipeline/small_model_rootcause_weighted_topk.py"
)

MISSING_FILES=()
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$PROJECT_ROOT/$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file (缺失)"
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -eq 0 ]; then
    echo "✅ 所有核心文件都存在"
else
    echo "⚠️  缺失 ${#MISSING_FILES[@]} 个文件，请检查上传是否完成"
fi
echo ""

# 5. 检查数据文件
echo "[5/6] 检查数据文件..."
if [ -f "$PROJECT_ROOT/output/llm_inputs_v4.jsonl" ]; then
    LINE_COUNT=$(wc -l < "$PROJECT_ROOT/output/llm_inputs_v4.jsonl")
    FILE_SIZE=$(du -h "$PROJECT_ROOT/output/llm_inputs_v4.jsonl" | cut -f1)
    echo "✅ llm_inputs_v4.jsonl 存在"
    echo "  行数: $LINE_COUNT"
    echo "  大小: $FILE_SIZE"
else
    echo "⚠️  llm_inputs_v4.jsonl 不存在"
    echo "  提示: 如果数据未上传，需要运行数据生成流程"
fi

if [ -d "$PROJECT_ROOT/data/aiops2020" ]; then
    echo "✅ 原始数据目录存在"
else
    echo "⚠️  原始数据目录不存在"
fi
echo ""

# 6. 检查模型路径
echo "[6/6] 检查模型路径..."
MODEL_PATH="/root/autodl-tmp/hf_cache/Qwen2.5-7B-Instruct"
if [ -d "$MODEL_PATH" ]; then
    echo "✅ 模型目录存在: $MODEL_PATH"
    # 检查关键文件
    KEY_FILES=("config.json" "tokenizer_config.json" "tokenizer.json" "generation_config.json")
    for key_file in "${KEY_FILES[@]}"; do
        if [ -f "$MODEL_PATH/$key_file" ]; then
            echo "  ✅ $key_file"
        else
            echo "  ⚠️  $key_file (缺失)"
        fi
    done
    
    # 检查模型文件（通常是 .safetensors 或 .bin）
    MODEL_FILES=$(find "$MODEL_PATH" -name "*.safetensors" -o -name "*.bin" 2>/dev/null | wc -l)
    if [ "$MODEL_FILES" -gt 0 ]; then
        echo "  ✅ 找到 $MODEL_FILES 个模型文件"
    else
        echo "  ⚠️  未找到模型权重文件"
    fi
else
    echo "⚠️  模型目录不存在: $MODEL_PATH"
    echo "  提示: 如果模型未上传，需要运行: python download_qwen.py"
fi
echo ""

echo "=========================================="
echo "环境配置检查完成"
echo "=========================================="
echo ""
echo "下一步操作："
echo "1. 如果缺少文件，请等待上传完成"
echo "2. 如果模型未下载，运行: python download_qwen.py"
echo "3. 如果数据文件缺失，运行数据生成流程:"
echo "   - python pipeline/batch_build_sft.py"
echo "   - python pipeline/build_llm_inputs.py"
echo "4. 分配 GPU 实例后，开始训练:"
echo "   - python pipeline/train_qwen2_5_7b_qlora_demo.py"
echo ""
