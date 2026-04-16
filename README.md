# 微服务故障根因定位 - Qwen2.5-7B QLoRA微调项目

## 项目结构

```
Graduation_Project/
├── data/                    # 原始数据文件
├── output/                  # 处理后的数据文件
│   └── llm_inputs_v4.jsonl # 训练数据（1728条样本）
├── pipeline/                # 数据处理和训练脚本
│   ├── small_model/         # 小模型相关脚本
│   ├── large_model/         # 大模型相关脚本
│   │   └── train_qwen2_5_7b_qlora_demo.py  # 主训练脚本
│   └── [通用数据处理脚本]
├── research/                # 研究文档
├── v2_doc/                  # 训练评估结果
│   ├── archive/             # 归档的旧评估结果
│   ├── 训练评估结果_*.json  # 最新3次评估结果
│   └── 训练评估报告_*.md    # 最新3次评估报告
├── logs/                    # 所有日志文件
├── docs/                    # 项目文档（Markdown、PDF）
├── scripts/                 # 工具脚本（Python、Shell、PowerShell）
├── config/                  # 配置文件
├── train_qwen2_5_7b_qlora_old.py  # 原始工作版本
└── README.md               # 项目说明（本文件）
```

## 快速开始

### 训练大模型
```bash
cd /root/autodl-tmp/Graduation_Project
python3 pipeline/large_model/train_qwen2_5_7b_qlora_demo.py
```

### 查看最新训练结果
```bash
ls -lt v2_doc/训练评估结果_*.json | head -1
```

### 查看训练日志
```bash
ls -lt logs/training_*.log | head -1
```

## 路径与配置

- **本机**：优先设置环境变量 `GRADUATION_PROJECT_ROOT` 指向仓库根目录（避免本机同时存在 `D:\Graduation_Project` 与 `E:\Graduation_Project_online` 时用错路径）。PowerShell 可执行：`. .\scripts\local_env.ps1`
- **线上（AutoDL）**：不设变量亦可；服务器若项目不在默认路径，同样可设 `GRADUATION_PROJECT_ROOT=/root/autodl-tmp/Graduation_Project`。
- 基座模型路径：`QWEN_MODEL_PATH` 或 `HF_MODEL_PATH`（本地默认尝试 `D:\hf_cache\Qwen2.5-7B-Instruct`）。
- 实验默认参数见 `config/experiment_defaults.yaml`。
- 打包上传：`scripts\pack_for_upload.ps1`、`upload_to_server.ps1` 已改为自动使用当前仓库根（或通过上述环境变量）。

## 流水线与工具脚本

| 说明 | 命令 |
|------|------|
| 打印推荐步骤 | `python pipeline/run_rca_pipeline.py list` |
| 划分 train/eval（与训练脚本 shuffle 逻辑一致） | `python pipeline/run_rca_pipeline.py split` |
| 离线评估推理 JSONL | `python pipeline/evaluate_predictions_jsonl.py --pred <v2_doc/llm_eval_after_*.jsonl>`（若存在 `output/eval_split.jsonl` 会自动作 `--gold`） |
| 无 GPU 演示推理 | `python pipeline/inference_cli.py --mock --input research/demo_inference_sft_sample.json` |
| GRPO 奖励函数 / 骨架 | `pipeline/large_model/grpo_rewards.py`、`grpo_train_skeleton.py` |

训练脚本在存在 `output/train_split.jsonl` 与 `output/eval_split.jsonl` 时**默认使用固定划分**（与 `split_dataset.py` 一致，便于本机/线上对齐）。需要改回单次 shuffle 时设置环境变量 `USE_FIXED_SPLIT=0`。

## 配置说明

- **模型**: Qwen2.5-7B-Instruct
- **量化**: 4-bit QLoRA (NF4)
- **LoRA配置**: r=8, alpha=16
- **训练轮数**: 1 epoch
- **学习率**: 2e-4
- **训练样本**: 1528条
- **评估样本**: 200条

## 最新训练结果

查看 `v2_doc/训练评估结果_*.json` 获取最新评估结果。

## Git 与 AutoDL 工作流

可以：在**本机**调试代码 → `git push` 到 GitHub → 在 **AutoDL** 实例里 `git clone` / `git pull` 后训练。

1. **本机**：安装依赖、改代码、小步提交；不要提交大文件（`data/`、`models/`、检查点已在 `.gitignore` 中排除）。
2. **GitHub**：`git remote add origin https://github.com/KikoStar123/ms-rca-llm.git`（若已添加可跳过），`git push -u origin main`。
3. **AutoDL**：数据盘建议 `/root/autodl-tmp`，示例：
   ```bash
   bash scripts/autodl_from_git.sh /root/autodl-tmp/ms-rca-llm
   cd /root/autodl-tmp/ms-rca-llm
   export GRADUATION_PROJECT_ROOT=/root/autodl-tmp/ms-rca-llm
   pip install -r requirements-train.txt
   # 按显卡 CUDA 版本安装 PyTorch（见 PyTorch 官网），再配置 Qwen 模型路径与数据
   python pipeline/large_model/train_qwen2_5_7b_qlora_demo.py
   ```
4. **模型与数据**：基座模型用 HuggingFace 缓存或 `huggingface-cli download`；原始大数据集在本机或网盘传到 AutoDL，与仅同步代码的 Git 流程分开。

### 快捷脚本

| 环境 | 脚本 | 作用 |
|------|------|------|
| Windows 本机 | `.\scripts\git_push.ps1 -Message "说明"` | `git add` → `commit` → `push`（默认分支 `main`） |
| AutoDL / Linux | `bash scripts/git_pull_server.sh` | `git pull --ff-only`（请先 `export GRADUATION_PROJECT_ROOT=你的克隆路径`） |
| AutoDL / Linux | `bash scripts/remote_full_pipeline.sh` | 远程一条龙：`git pull` → `build_llm_inputs` → `split_dataset` → 训练（见指南） |

**只在服务器上跑全流程**（本机不训练）：见 **`docs/SERVER_TRAINING_AGENT_GUIDE.md`** 中「全流程只在远程执行」；交给 Agent 时仍用同文档的检查清单与命令。

### 服务器上 `data/`、`models/` 放到哪里

先设定仓库根（与 `config.py` 一致），例如：

`export GRADUATION_PROJECT_ROOT=/root/autodl-tmp/ms-rca-llm`（按你的实际克隆路径修改）。

| 本机目录 | 上传到服务器 | 说明 |
|----------|----------------|------|
| `data/`（含 `aiops2020` 等） | `$GRADUATION_PROJECT_ROOT/data/` | 保持与本机相同的子目录结构；`config` 会优先使用 `data/aiops2020`（若存在） |
| `models/Qwen2.5-7B-Instruct/` | **方案 A**：`$GRADUATION_PROJECT_ROOT/models/Qwen2.5-7B-Instruct/`<br>**方案 B**：`/root/autodl-tmp/hf_cache/Qwen2.5-7B-Instruct/`，并执行 `export QWEN_MODEL_PATH=/root/autodl-tmp/hf_cache/Qwen2.5-7B-Instruct` | 方案 B 可与其它项目共用 HuggingFace 缓存目录 |

若使用 AutoDL 默认目录名 `Graduation_Project` 且路径为 `/root/autodl-tmp/Graduation_Project`，`config.py` 会自动指向该目录下的 `data` 与 `/root/autodl-tmp/hf_cache/` 中的基座模型。**若仓库目录名不是该默认路径，必须设置 `GRADUATION_PROJECT_ROOT`**，否则会误用其它分支里的路径逻辑。

## 注意事项

- 训练日志保存在 `logs/` 目录
- 旧的评估结果已归档到 `v2_doc/archive/`
- 只保留最新3次训练评估结果在主目录
