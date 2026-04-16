# 项目完整历史总结

## 项目概述

**项目名称**：基于大小模型协同的微服务故障根因定位系统设计与实现

**核心目标**：
1. 构建适用于大模型训练的数据准备流程，完成 SFT 微调
2. 采用 GRPO 等方式进行优化训练（未来阶段）
3. 完成系统工程实现，包括数据接入、分析流程编排、结果输出与展示

**核心架构**：小模型（加权统计 Top-K）负责候选召回，大模型（Qwen2.5-7B-Instruct）负责精炼判断和故障类型分类

---

## 一、环境搭建阶段

### 1.1 硬件配置
- **CPU**: Intel i7-14650HX（14 核 20 线程）
- **内存**: 32GB DDR5
- **GPU**: NVIDIA RTX 4070 Laptop（8GB 显存）
- **CUDA**: 12.1（文档中固定版本，实际驱动支持更高版本）

### 1.2 软件环境
- **Python**: 3.12
- **PyTorch**: 2.10.0+cu130（本地环境，适配 CUDA 13.0）
- **核心库**: transformers, peft, bitsandbytes, accelerate, datasets

### 1.3 环境问题解决
- ✅ 解决 PyTorch 与 CUDA 版本匹配问题
- ✅ 解决 bitsandbytes 量化库适配（最终采用 4-bit 量化）
- ✅ 解决 Qwen2.5-7B-Instruct 模型文件下载和完整性校验
- ✅ 解决 PyTorch 废弃 API 问题（`torch.cuda.memory_used()` → `torch.cuda.max_memory_allocated()`）

### 1.4 模型部署
- **模型**: Qwen2.5-7B-Instruct（13GB，4 个分片）
- **加载方式**: `device_map="auto"` + `torch.bfloat16` + 4-bit 量化
- **量化配置**: `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")`
- **显存占用**: 峰值 6.4GB/8.0GB，部分参数卸载至 CPU（正常策略）

---

## 二、数据处理流程

### 2.1 数据集
- **来源**: AIOps2020 挑战赛数据集
- **内容**: 故障事件列表 + 多源观测数据（平台指标、链路追踪、业务指标）
- **时间窗口**: 围绕故障 `log_time` 的 `window_before` 和 `window_after`

### 2.2 数据预处理流程

#### 阶段 1: 原始 SFT 样本生成（`batch_build_sft.py`）
- **功能**: 批量生成 SFT 样本，支持多时间窗口预设
- **窗口预设**: 从 22 个增加到 30 个（对称和非对称窗口）
- **输出**: `output/sft_samples_YYYY-MM-DD.jsonl`（每日一个文件）
- **数据量**: 从 ~622 条增加到 1728 条

#### 阶段 2: LLM 输入构建（`build_llm_inputs.py`）
- **功能**: 从 SFT 样本构建最终 LLM 训练输入，集成 Top-K 候选
- **关键改进**:
  - 从顺序执行改为多进程并行（`ProcessPoolExecutor`）
  - 实现文件锁机制（`.jsonl.lock`）防止并发写入冲突
  - 修复缩进错误导致的文件处理问题
- **输出**: `output/llm_inputs_v4.jsonl`（1728 条训练样本）

#### 阶段 3: Top-K 候选生成（`small_model_rootcause_weighted_topk.py`）
- **功能**: 小模型生成根因候选列表（Top-1, Top-3, Top-5）
- **方法**: 加权统计模型（频率 + 异常分数）
- **输出**: 每个故障事件的 Top-K 候选组件列表

### 2.3 数据增强策略
- **多窗口数据增强**: 使用 30 个不同的时间窗口预设，从同一故障事件生成多个训练样本
- **证据聚合**: 将原始数据汇总为统计摘要（count, avg, min, max 等），减少 LLM 输入大小

---

## 三、模型训练阶段

### 3.1 训练配置（`train_qwen2_5_7b_qlora_demo.py`）

#### 核心参数
- **微调方法**: QLoRA（4-bit 量化 + LoRA）
- **LoRA 配置**: `r=8, lora_alpha=16`
- **目标模块**: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
- **训练样本**: 1728 条（`MAX_SAMPLES = 0` 表示使用全部）
- **评估样本**: 40 条（`EVAL_SIZE = 40`）
- **最大长度**: 384 tokens
- **训练步数**: 自动计算 1 epoch（`MAX_STEPS = 0`，实际为 `train_size // 4`）

#### SFT 实现细节
- **损失计算**: 仅对答案部分计算损失（prompt 部分 labels 设为 -100）
- **数据格式**: 
  - `prompt_text`: 仅包含问题
  - `full_text`: 问题 + 答案
  - `labels = [-100] * len(prompt_ids) + answer_ids`

#### 输出管理
- **时间戳标记**: 所有输出文件带时间戳（`RUN_TAG = YYYYMMDD_HHMMSS`）
- **输出文件**:
  - `v2_doc/llm_eval_before_{RUN_TAG}.jsonl`
  - `v2_doc/llm_eval_after_{RUN_TAG}.jsonl`
  - `v2_doc/训练评估结果_{RUN_TAG}.json`
  - `v2_doc/训练评估报告_{RUN_TAG}.md`

### 3.2 训练过程演进

#### 第一轮训练（小数据集）
- **训练样本**: 614 条
- **评估样本**: 40 条
- **结果**:
  - 微调前: root_acc 0.65, type_acc 1.0
  - 微调后: root_acc 0.85, type_acc 0.925

#### 数据扩充
- **问题**: 训练数据太少导致性能不稳定
- **解决方案**: 增加窗口预设数量（22 → 30），数据量从 ~622 → 1728
- **性能优化**: `build_llm_inputs.py` 从单线程改为多进程并行

#### 最新训练（1728 条数据）
- **状态**: 已完成模型加载，训练进行中
- **预期**: 使用更大数据集应能提升模型性能

---

## 四、评估指标

### 4.1 核心指标
1. **Parse Rate**: JSON 解析成功率
2. **Root Cause Top-1 Accuracy**: 根因组件 Top-1 准确率
3. **Fault Type Accuracy**: 故障类型分类准确率

### 4.2 评估流程
- **微调前评估**: 使用原始 Qwen2.5-7B-Instruct 模型
- **微调后评估**: 使用 SFT 后的模型
- **对比分析**: 生成详细的评估报告和 JSON 结果

---

## 五、文档编写

### 5.1 主要文档
1. **`document.md`**: 项目主文档，包含完整的技术方案和实验结果
2. **`README.md`**: 项目概览和快速开始指南
3. **`v2_doc/实验环境与方案设计.md`**: 
   - 实验环境构建和配置
   - 实验数据说明
   - 方案设计详解（大小模型协同架构）
   - 关键代码位置与片段
4. **`v2_doc/训练评估报告_{TIMESTAMP}.md`**: 每次训练的详细评估报告

### 5.2 文档更新历史
- 更新硬件配置（CUDA 版本从 13.0 修正为 12.1）
- 更新训练数据扩充说明（30 个窗口预设）
- 更新性能指标（最新训练结果）
- 添加代码片段说明（便于理解实现细节）

---

## 六、代码结构

### 6.1 核心脚本
```
pipeline/
├── batch_build_sft.py              # 批量生成 SFT 样本
├── build_llm_inputs.py            # 构建 LLM 训练输入（集成 Top-K）
├── train_qwen2_5_7b_qlora_demo.py # QLoRA 微调主脚本
└── small_model_rootcause_weighted_topk.py  # 小模型 Top-K 生成

根目录/
├── config.py                       # 路径配置（支持本地/服务器切换）
├── download_qwen.py                # 模型下载脚本
├── test_qwen_gpu.py                # 模型推理测试脚本
└── ...
```

### 6.2 数据目录
```
data/
└── aiops2020/                      # AIOps2020 原始数据集

output/
├── llm_inputs_v4.jsonl            # 最终训练数据（1728 条）
├── sft_samples_*.jsonl            # 原始 SFT 样本
└── ...

v2_doc/
├── 实验环境与方案设计.md
├── 训练评估报告_*.md
└── 训练评估结果_*.json
```

---

## 七、关键问题与解决方案

### 7.1 数据生成性能问题
- **问题**: `build_llm_inputs.py` 运行缓慢，CPU 利用率低
- **解决**: 从 `ThreadPoolExecutor` 改为 `ProcessPoolExecutor`，充分利用多核 CPU
- **结果**: CPU 8 个大核几乎占满，生成速度显著提升

### 7.2 文件并发写入问题
- **问题**: 多个进程同时写入 `llm_inputs_v4.jsonl` 导致数据损坏
- **解决**: 实现文件锁机制（`.jsonl.lock`），确保单进程写入

### 7.3 训练数据不足
- **问题**: 初始数据量太小（~622 条），模型性能不稳定
- **解决**: 增加时间窗口预设数量，数据扩充至 1728 条

### 7.4 CUDA 版本兼容性
- **问题**: 文档中 CUDA 版本不一致（13.0 vs 12.1）
- **解决**: 统一修正为 CUDA 12.1（基于 RTX 4070 常见配置）

---

## 八、服务器迁移准备（最新阶段）

### 8.1 迁移目标
- **目标平台**: AutoDL RTX 5090 服务器
- **环境要求**: PyTorch 2.8.0 + Python 3.12 + CUDA 12.8

### 8.2 迁移方案
- **路径配置**: 创建 `config.py` 自动识别本地/服务器环境
- **上传脚本**: `upload_to_server.ps1`（PowerShell 打包脚本）
- **部署脚本**: `deploy_on_server.sh`（服务器端自动部署）
- **目录结构**:
  - 代码 → `/root/Graduation_Project`（系统盘）
  - 数据/模型 → `/root/autodl-tmp/`（数据盘）

### 8.3 当前状态
- ✅ 配置文件已创建
- ✅ 上传和部署脚本已准备
- ✅ 训练脚本已适配新配置
- ✅ 文件打包已完成
- ⏳ 压缩和上传进行中

---

## 九、技术栈总结

### 9.1 模型
- **大模型**: Qwen2.5-7B-Instruct（通义千问）
- **微调方法**: QLoRA（4-bit 量化 + LoRA）
- **小模型**: 加权统计 Top-K 模型（Python 实现）

### 9.2 框架与库
- **深度学习**: PyTorch 2.10.0+cu130（本地）/ 2.8.0+cu128（服务器）
- **Transformers**: HuggingFace Transformers（最新版）
- **微调**: PEFT（Parameter-Efficient Fine-Tuning）
- **量化**: bitsandbytes（4-bit NF4 量化）
- **数据处理**: HuggingFace Datasets

### 9.3 数据格式
- **输入**: JSON Lines（`.jsonl`），包含任务指令、故障信息、证据摘要、Top-K 候选
- **输出**: 结构化 JSON，包含根因组件、故障类型、KPI、相关容器等字段

---

## 十、项目里程碑

1. ✅ **环境搭建完成** - 本地 RTX 4070 环境配置成功
2. ✅ **数据流程打通** - 从原始数据到训练样本的完整流程
3. ✅ **首次训练成功** - 614 条数据完成 SFT，性能提升明显
4. ✅ **数据扩充完成** - 数据量扩充至 1728 条
5. ✅ **性能优化完成** - 多进程并行加速数据生成
6. ✅ **文档体系完善** - 完整的技术文档和实验报告
7. ⏳ **服务器迁移进行中** - 准备在 RTX 5090 上继续训练

---

## 十一、下一步计划

### 短期（当前阶段）
1. 完成服务器迁移和部署
2. 在 RTX 5090 上使用 1728 条数据重新训练
3. 评估更大数据集的效果

### 中期（未来阶段）
1. 实施 GRPO 优化训练（提升输出稳定性）
2. 进一步扩充训练数据（如需要）
3. 完善系统工程实现（数据接入、流程编排、结果展示）

### 长期（毕业设计完成）
1. 系统集成和前端展示
2. 完整实验报告和论文撰写
3. 答辩准备

---

## 十二、重要文件清单

### 代码文件
- `pipeline/train_qwen2_5_7b_qlora_demo.py` - 训练主脚本
- `pipeline/build_llm_inputs.py` - 数据构建脚本
- `pipeline/batch_build_sft.py` - SFT 样本生成
- `pipeline/small_model_rootcause_weighted_topk.py` - 小模型实现
- `config.py` - 路径配置（新增）

### 数据文件
- `output/llm_inputs_v4.jsonl` - 训练数据（1728 条）
- `data/aiops2020/` - 原始数据集

### 文档文件
- `document.md` - 主文档
- `README.md` - 项目概览
- `v2_doc/实验环境与方案设计.md` - 详细技术说明
- `v2_doc/训练评估报告_*.md` - 训练结果报告

### 部署文件
- `upload_to_server.ps1` - 上传脚本
- `deploy_on_server.sh` - 部署脚本
- `UPLOAD_GUIDE.md` - 上传指南

---

## 十三、关键配置参数

### 训练参数（当前）
```python
MAX_SAMPLES = 0          # 使用全部样本
EVAL_SIZE = 40          # 评估集大小
MAX_LENGTH = 384        # 最大序列长度
MAX_STEPS = 0           # 自动计算 1 epoch
USE_4BIT = True         # 使用 4-bit 量化
SEED = 42               # 随机种子
```

### LoRA 参数
```python
r = 8                   # LoRA rank
lora_alpha = 16        # LoRA alpha
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                  "gate_proj", "up_proj", "down_proj"]
```

### 量化参数
```python
load_in_4bit = True
bnb_4bit_quant_type = "nf4"
bnb_4bit_compute_dtype = torch.float16
```

---

## 十四、注意事项

1. **路径配置**: 代码已支持自动识别本地/服务器环境，无需手动修改路径
2. **数据生成**: 如果重新生成数据，需要先运行 `batch_build_sft.py`，再运行 `build_llm_inputs.py`
3. **训练输出**: 所有训练结果带时间戳，不会覆盖之前的运行
4. **文件锁**: `build_llm_inputs.py` 使用文件锁防止并发冲突，如果卡住检查 `.lock` 文件
5. **显存管理**: 8GB 显存下使用 4-bit 量化是必需的，模型会部分卸载到 CPU（正常现象）

---

## 十五、联系方式与资源

- **项目路径**: `d:\Graduation_Project`（本地）/ `/root/Graduation_Project`（服务器）
- **模型路径**: `D:\hf_cache\Qwen2.5-7B-Instruct`（本地）/ `/root/autodl-tmp/models/Qwen2.5-7B-Instruct`（服务器）
- **服务器**: AutoDL RTX 5090（`ssh -p 12115 root@connect.westc.gpuhub.com`）

---

**文档生成时间**: 2026-01-23  
**项目状态**: 服务器迁移准备中  
**最新训练数据量**: 1728 条  
**最新训练结果**: 进行中（待完成）
