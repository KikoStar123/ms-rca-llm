# AutoDL 服务器完整上传指南

## 📋 概述

本指南帮助你将本地 27GB 的完整项目（包括代码、数据集、模型）上传到 AutoDL 服务器。

## 🗂️ 服务器目录结构

- **系统盘 `/`** (30GB，会随镜像保存)
  - `/root/Graduation_Project/` - 代码、脚本、文档

- **数据盘 `/root/autodl-tmp/`** (50GB，速度快)
  - `/root/autodl-tmp/datasets/aiops2020_raw/` - 原始数据集
  - `/root/autodl-tmp/output/` - 训练输出、中间结果
  - `/root/autodl-tmp/models/Qwen2.5-7B-Instruct/` - 模型文件 (~13GB)
  - `/root/autodl-tmp/hf_cache/` - HuggingFace 缓存

## 📤 方法一：使用 PowerShell 脚本（推荐）

### 步骤 1: 准备上传文件

在本地 PowerShell 中执行：

```powershell
cd d:\Graduation_Project
.\upload_to_server.ps1 -ServerIP "你的服务器IP" -ServerUser "root"
```

脚本会：
- 自动打包所有文件（代码、数据、模型）
- 生成压缩包 `d:\Graduation_Project_upload.zip`

### 步骤 2: 上传到服务器

**选项 A: 使用 AUTODL 网页上传（最简单）**
1. 登录 AUTODL 控制台
2. 进入文件管理
3. 上传 `d:\Graduation_Project_upload.zip` 到 `/root/` 目录

**选项 B: 使用 SCP 命令**
```powershell
scp -P 22 d:\Graduation_Project_upload.zip root@你的服务器IP:/root/
```

### 步骤 3: 在服务器上部署

SSH 登录服务器后执行：

```bash
cd /root
bash deploy_on_server.sh
```

脚本会自动：
- 解压文件
- 整理目录结构（代码放系统盘，数据/模型放数据盘）
- 配置环境变量

## 📤 方法二：手动上传（如果脚本有问题）

### 步骤 1: 手动打包

在本地 PowerShell：

```powershell
cd d:\Graduation_Project

# 创建临时目录
mkdir d:\upload_temp
cd d:\upload_temp

# 复制代码
xcopy /E /I d:\Graduation_Project\pipeline pipeline
xcopy /E /I d:\Graduation_Project\research research
xcopy /E /I d:\Graduation_Project\v2_doc v2_doc
copy d:\Graduation_Project\*.py .
copy d:\Graduation_Project\*.md .
copy d:\Graduation_Project\*.pdf .

# 复制数据
xcopy /E /I d:\Graduation_Project\data data
xcopy /E /I d:\Graduation_Project\output output

# 复制模型
xcopy /E /I D:\hf_cache\Qwen2.5-7B-Instruct models\Qwen2.5-7B-Instruct

# 压缩
cd d:\
Compress-Archive -Path upload_temp\* -DestinationPath Graduation_Project_upload.zip
```

### 步骤 2: 上传并解压

1. 上传 `Graduation_Project_upload.zip` 到服务器 `/root/`
2. SSH 登录后执行：

```bash
cd /root
unzip -q Graduation_Project_upload.zip

# 整理目录
mkdir -p /root/autodl-tmp/datasets /root/autodl-tmp/output /root/autodl-tmp/models
mv data /root/autodl-tmp/datasets/aiops2020_raw
mv output /root/autodl-tmp/output
mv models/* /root/autodl-tmp/models/
mv Graduation_Project/* /root/Graduation_Project/
```

## ✅ 验证部署

在服务器上检查：

```bash
# 检查代码
ls -lh /root/Graduation_Project/pipeline/

# 检查数据集
ls -lh /root/autodl-tmp/datasets/aiops2020_raw/

# 检查模型
ls -lh /root/autodl-tmp/models/Qwen2.5-7B-Instruct/

# 检查输出
ls -lh /root/autodl-tmp/output/
```

## 🔧 配置环境

```bash
cd /root/Graduation_Project

# 创建环境
conda create -n gp-rca python=3.10 -y
conda activate gp-rca

# 安装 PyTorch 2.8.0 + CUDA 12.8
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# 安装依赖
pip install transformers peft accelerate bitsandbytes datasets einops sentencepiece

# 设置 HuggingFace 缓存
export HF_HOME=/root/autodl-tmp/hf_cache
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache
```

## 🚀 开始训练

```bash
cd /root/Graduation_Project
python pipeline/train_qwen2_5_7b_qlora_demo.py
```

## 📝 注意事项

1. **路径自动切换**: 代码已配置为自动识别本地/服务器环境，无需手动修改路径
2. **磁盘空间**: 确保数据盘有足够空间（模型 ~13GB + 数据集 ~几GB + 输出 ~几GB）
3. **SSH 密钥**: 建议配置 SSH 密钥认证，避免每次输入密码
4. **断点续传**: 如果上传中断，可以使用 `rsync` 或分块上传

## 🐛 常见问题

**Q: 上传速度慢怎么办？**  
A: 可以使用 AUTODL 的网页上传功能，通常比 SCP 更快。

**Q: 压缩包太大无法上传？**  
A: 可以分块压缩，或使用 `rsync` 直接同步目录。

**Q: 模型文件找不到？**  
A: 检查 `D:\hf_cache\Qwen2.5-7B-Instruct` 是否存在，如果不在，需要先下载模型。
