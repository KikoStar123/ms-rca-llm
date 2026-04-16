# 项目结构说明

## 目录结构

```
Graduation_Project/
├── data/                          # 原始数据文件
│   └── aiops2020/                 # AIOps2020原始数据集（CSV文件）
├── output/                         # 处理后的数据文件
│   └── llm_inputs_v4.jsonl        # 训练数据（1728条样本）
├── pipeline/                       # 数据处理和训练脚本
│   ├── small_model/                # 小模型相关脚本
│   │   ├── small_model_baseline.py  # 小模型基线方法
│   │   ├── small_model_rootcause_topk.py  # 小模型Top-K根因定位
│   │   ├── small_model_rootcause_weighted_topk.py  # 小模型加权Top-K
│   │   ├── batch_rootcause_topk.py  # 批量计算Top-K根因候选
│   │   ├── batch_rootcause_weighted_topk.py  # 批量计算加权Top-K
│   │   ├── grid_search_topk.py      # Top-K参数网格搜索
│   │   ├── compare_topk_summaries.py  # 对比Top-K结果摘要
│   │   └── aiops_baseline.py        # AIOps基线方法（从scripts移动）
│   ├── large_model/                # 大模型相关脚本
│   │   ├── train_qwen2_5_7b_qlora_demo.py  # Qwen2.5-7B QLoRA训练脚本（当前使用）
│   │   ├── test_qwen_gpu.py        # 测试Qwen GPU环境（从scripts移动）
│   │   └── download_qwen.py         # 下载Qwen模型（从scripts移动）
│   ├── batch_build_sft.py          # 批量生成SFT样本（多进程，通用）
│   ├── build_sft_samples.py        # 单个日期SFT样本生成（通用）
│   ├── build_llm_inputs.py         # 构建LLM训练输入（集成Top-K候选，通用）
│   ├── extract_day.py              # 解压指定日期的AIOps数据（通用）
│   ├── merge_jsonl.py              # 合并多个JSONL文件（通用）
│   ├── sft_stats.py                # SFT样本统计（通用）
│   ├── llm_inputs_stats.py         # LLM输入数据统计（通用）
│   ├── validate_rca_output.py      # 验证根因定位输出格式（通用）
│   └── run_end_to_end.ps1          # 端到端流程脚本（Windows，通用）
├── research/                       # 研究文档（29个Markdown文件）
├── v2_doc/                         # 训练评估结果
│   ├── archive/                    # 归档的旧评估结果（30个文件）
│   ├── 训练评估结果_*.json         # 最新3次评估结果
│   ├── 训练评估报告_*.md           # 最新3次评估报告
│   ├── llm_eval_before_*.jsonl     # 微调前评估输出
│   ├── llm_eval_after_*.jsonl      # 微调后评估输出
│   └── 实验环境与方案设计.md       # 实验设计文档
├── logs/                           # 所有日志文件（20个）
│   ├── training_*.log              # 训练日志
│   ├── sft_generation*.log         # 数据生成日志
│   └── build_llm_inputs*.log       # 数据构建日志
├── docs/                           # 项目文档
│   ├── CRITICAL_FIX.md             # 关键问题修复记录（loss=0问题）
│   ├── CRITICAL_FIX_V2.md          # 关键问题修复记录V2
│   ├── TRAINING_FIX_SUMMARY.md     # 训练问题修复总结
│   ├── MONITORING_GUIDE.md         # 训练监控指南
│   ├── OPTIMIZATION_NOTES.md        # 优化说明（多进程、服务器管理）
│   ├── PROJECT_HISTORY.md          # 项目历史记录
│   ├── UPLOAD_GUIDE.md             # 文件上传指南
│   ├── document.md                 # 项目文档
│   ├── Paper1-5.pdf                # 相关论文（5篇）
│   └── 基于大小模型协同的故障根因定位系统设计与实现--前期调研报告(1).pdf
├── scripts/                         # 工具脚本
│   ├── aiops_baseline.py            # AIOps基线方法实现
│   ├── download_qwen.py            # 下载Qwen模型脚本
│   ├── test_qwen_gpu.py            # 测试Qwen GPU环境
│   ├── verify_ready.py             # 验证环境就绪状态
│   ├── deploy_on_server.sh         # 服务器部署脚本
│   ├── setup_environment.sh        # 环境设置脚本
│   ├── check_upload_progress.ps1   # 检查上传进度（Windows）
│   ├── pack_for_upload.ps1         # 打包上传文件（Windows）
│   └── upload_to_server.ps1        # 上传到服务器（Windows）
├── config/                         # 配置文件
│   └── window_presets_*.txt        # 时间窗口预设配置（10个文件）
│       ├── window_presets_3.txt   # 3分钟窗口
│       ├── window_presets_5.txt    # 5分钟窗口
│       ├── window_presets_8.txt    # 8分钟窗口
│       ├── window_presets_15.txt   # 15分钟窗口
│       ├── window_presets_20.txt   # 20分钟窗口
│       ├── window_presets_22_reasonable.txt  # 22分钟合理窗口
│       ├── window_presets_30.txt   # 30分钟窗口
│       ├── window_presets_50.txt  # 50分钟窗口
│       ├── window_presets_50_balanced.txt  # 50分钟平衡窗口
│       └── window_presets_55.txt  # 55分钟窗口
├── models/                         # 模型检查点目录
├── config.py                       # 项目配置文件（路径配置，支持本地/服务器环境）
├── train_qwen2_5_7b_qlora_old.py  # 原始工作版本（参考）
├── README.md                       # 项目说明
└── PROJECT_STRUCTURE.md           # 本文件
```

## 重要文件详细说明

### 大模型训练脚本

#### `pipeline/large_model/train_qwen2_5_7b_qlora_demo.py` - 主训练脚本（当前使用）
- **功能**: Qwen2.5-7B-Instruct模型的QLoRA微调训练
- **特性**:
  - 4-bit量化（NF4）
  - LoRA配置：r=8, alpha=16
  - 多进程tokenization（24核心）
  - 自动评估（微调前后对比）
  - 评估指标：根因Top-1/3/5、故障类型、KPI、相关容器、完整匹配
- **配置**: 1 epoch, lr=2e-4, batch_size=1, gradient_accumulation=4

#### `train_qwen2_5_7b_qlora_old.py` - 原始工作版本
- **功能**: 用户上传的原始工作版本（参考）
- **用途**: 作为配置参考，确保训练配置正确

### 数据处理脚本

#### `pipeline/extract_day.py`
- **功能**: 解压AIOps2020数据集中指定日期的数据
- **用法**: `python extract_day.py --date 2020-04-11`

#### `pipeline/build_sft_samples.py`
- **功能**: 为单个日期生成SFT（Supervised Fine-Tuning）样本
- **参数**: 时间窗口（before/after）、最大样本数、事件限制

#### `pipeline/batch_build_sft.py`
- **功能**: 批量生成SFT样本（多进程，支持多日期）
- **特性**: 使用24核心并行处理，大幅提升数据生成速度

#### `pipeline/build_llm_inputs.py`
- **功能**: 构建LLM训练输入，集成Top-K根因候选列表
- **特性**: 
  - 支持加权Top-K计算
  - 可配置权重（平台指标、调用链、异常强度）
  - 多线程并行计算

#### `pipeline/merge_jsonl.py`
- **功能**: 合并多个JSONL文件为一个文件
- **用途**: 合并多日期的SFT样本

#### `pipeline/sft_stats.py`
- **功能**: 统计SFT样本的分布情况
- **输出**: JSON格式的统计报告

#### `pipeline/llm_inputs_stats.py`
- **功能**: 统计LLM输入数据的分布情况

#### `pipeline/validate_rca_output.py`
- **功能**: 验证根因定位输出的格式和有效性
- **输出**: 验证报告JSON

### 小模型根因定位脚本

#### `pipeline/small_model/small_model_baseline.py`
- **功能**: 小模型基线方法实现

#### `pipeline/small_model/small_model_rootcause_topk.py`
- **功能**: 小模型Top-K根因定位

#### `pipeline/small_model/small_model_rootcause_weighted_topk.py`
- **功能**: 小模型加权Top-K根因定位

#### `pipeline/small_model/batch_rootcause_topk.py`
- **功能**: 批量计算Top-K根因候选（标准方法）

#### `pipeline/small_model/batch_rootcause_weighted_topk.py`
- **功能**: 批量计算加权Top-K根因候选（加权方法）

#### `pipeline/small_model/grid_search_topk.py`
- **功能**: Top-K参数网格搜索

#### `pipeline/small_model/compare_topk_summaries.py`
- **功能**: 对比不同Top-K方法的结果摘要

#### `pipeline/small_model/aiops_baseline.py`
- **功能**: AIOps基线方法实现（从scripts移动）

### 大模型工具脚本

#### `pipeline/large_model/download_qwen.py`
- **功能**: 下载Qwen模型文件（从scripts移动）

#### `pipeline/large_model/test_qwen_gpu.py`
- **功能**: 测试Qwen模型在GPU上的运行情况（从scripts移动）

#### `scripts/verify_ready.py`
- **功能**: 验证环境是否就绪（模型、数据、依赖等）

#### `scripts/deploy_on_server.sh`
- **功能**: 服务器部署脚本

#### `scripts/setup_environment.sh`
- **功能**: 环境设置脚本（安装依赖、配置环境）

#### `scripts/check_upload_progress.ps1`
- **功能**: 检查文件上传进度（Windows PowerShell）

#### `scripts/pack_for_upload.ps1`
- **功能**: 打包文件用于上传（Windows PowerShell）

#### `scripts/upload_to_server.ps1`
- **功能**: 上传文件到服务器（Windows PowerShell）

**注意**: 以下文件已移动到对应目录：
- `aiops_baseline.py` → `pipeline/small_model/`
- `download_qwen.py` → `pipeline/large_model/`
- `test_qwen_gpu.py` → `pipeline/large_model/`

### 配置文件

#### `config.py`
- **功能**: 项目路径配置
- **特性**: 自动检测本地/服务器环境，切换路径

#### `config/window_presets_*.txt`
- **功能**: 时间窗口预设配置
- **格式**: 每行一个配置，格式为 `before,after`（分钟数）
- **用途**: 用于SFT样本生成时指定时间窗口

### 文档文件

#### `docs/CRITICAL_FIX.md` / `docs/CRITICAL_FIX_V2.md`
- **内容**: 记录loss=0问题的诊断和修复过程

#### `docs/TRAINING_FIX_SUMMARY.md`
- **内容**: 训练问题修复总结

#### `docs/MONITORING_GUIDE.md`
- **内容**: 训练监控指南（如何查看loss、GPU使用率等）

#### `docs/OPTIMIZATION_NOTES.md`
- **内容**: 优化说明（多进程、服务器自动关闭等）

#### `docs/PROJECT_HISTORY.md`
- **内容**: 项目历史记录和开发过程

#### `docs/UPLOAD_GUIDE.md`
- **内容**: 文件上传到服务器的指南

#### `docs/document.md`
- **内容**: 项目主要文档

#### `docs/Paper1-5.pdf`
- **内容**: 相关研究论文（5篇）

### 数据文件

#### `output/llm_inputs_v4.jsonl`
- **功能**: LLM训练数据
- **格式**: JSONL，每行一个训练样本
- **内容**: 
  - `input`: 包含故障事件、观测证据、Top-K候选列表
  - `output`: 包含根因组件、故障类型、KPI、相关容器
- **数量**: 1728条样本（训练集1528条，评估集200条）

### 评估结果

#### `v2_doc/训练评估结果_*.json`
- **格式**: JSON格式的评估结果
- **内容**: 
  - `before`: 微调前的评估指标
  - `after`: 微调后的评估指标
  - 指标包括：可解析率、根因Top-1/3/5、故障类型、KPI、相关容器、完整匹配

#### `v2_doc/训练评估报告_*.md`
- **格式**: Markdown格式的评估报告
- **内容**: 人类可读的评估结果摘要

#### `v2_doc/llm_eval_before_*.jsonl` / `llm_eval_after_*.jsonl`
- **格式**: JSONL格式的详细评估输出
- **内容**: 每个评估样本的prompt、prediction、label

## 文件整理规则

1. **日志文件**: 统一存放在 `logs/` 目录
2. **评估结果**: 只保留最新3次在主目录，其余归档到 `v2_doc/archive/`
3. **Python缓存**: 已清理 `__pycache__` 和 `*.pyc` 文件
4. **重复文件**: 已删除无时间戳的重复文件
5. **文档文件**: 统一存放在 `docs/` 目录
6. **工具脚本**: 统一存放在 `scripts/` 目录
7. **配置文件**: 统一存放在 `config/` 目录

## 快速命令

```bash
# 查看最新训练结果
ls -lt v2_doc/训练评估结果_*.json | head -1

# 查看最新训练日志
ls -lt logs/training_*.log | head -1

# 查看项目结构
tree -L 2 -I '__pycache__|*.pyc' --dirsfirst

# 启动大模型训练
python3 pipeline/large_model/train_qwen2_5_7b_qlora_demo.py

# 运行小模型基线方法
python3 pipeline/small_model/small_model_baseline.py

# 批量生成SFT样本
python3 pipeline/batch_build_sft.py --preset config/window_presets_22_reasonable.txt

# 构建LLM训练输入
python3 pipeline/build_llm_inputs.py --input-dir output --output output/llm_inputs_v4.jsonl
```

## 数据流程

1. **数据提取**: `extract_day.py` → 解压原始数据
2. **SFT样本生成**: `build_sft_samples.py` / `batch_build_sft.py` → 生成SFT样本
3. **LLM输入构建**: `build_llm_inputs.py` → 构建训练输入（集成Top-K）
4. **训练**: `train_qwen2_5_7b_qlora_demo.py` → 微调模型
5. **评估**: 自动在训练前后进行评估，生成评估报告
