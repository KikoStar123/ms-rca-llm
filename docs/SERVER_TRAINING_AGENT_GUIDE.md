# 服务器训练执行指南（交给 Agent 按序执行）

本文档假设运行环境为 **Linux + NVIDIA GPU**（如 AutoDL），目标为在已上传依赖的前提下，**成功跑通** `pipeline/large_model/train_qwen2_5_7b_qlora_demo.py` 的 QLoRA 微调与评估。

---

## 全流程只在远程执行（本机可不跑）

若你希望**不在本机执行任何训练或数据构建**，只在 AutoDL 等服务器上完成一切，请按下面准备与执行。

### 一次性准备（传到服务器或在该机下载）

| 内容 | 说明 |
|------|------|
| 代码 | `git clone` 本仓库后，之后只用 `git pull` |
| `data/aiops2020/` | 原始赛题数据目录（**重建 LLM 输入 / 小模型 Top‑K 时必需**） |
| `output/sft_samples_*.jsonl` | SFT 样本（**运行 `build_llm_inputs.py` 前必需**；若你从未在远端生成过，需先在本机或其它环境生成后上传，或在服务器上按 `python pipeline/run_rca_pipeline.py list` 从原始数据生成） |
| 基座模型 | `Qwen2.5-7B-Instruct` 整包，或在该机 `huggingface-cli download` |

本机**不需要**再跑训练；仅需保证上述文件在服务器上的路径与 `GRADUATION_PROJECT_ROOT`、`QWEN_MODEL_PATH` 一致。

### 一键脚本（推荐）

在已 `export GRADUATION_PROJECT_ROOT` 与 `QWEN_MODEL_PATH` 后：

```bash
cd "$GRADUATION_PROJECT_ROOT"
bash scripts/remote_full_pipeline.sh
```

默认顺序：**git pull** → **`build_llm_inputs.py`** → **`split_dataset.py`** → **训练**。  
若已有现成的 `output/llm_inputs_v4.jsonl` 与划分文件、只想训练：

```bash
export SKIP_DATA_BUILD=1
bash scripts/remote_full_pipeline.sh
```

只重建数据、不训练：

```bash
export SKIP_TRAIN=1
bash scripts/remote_full_pipeline.sh
```

训练附加参数（示例）：

```bash
export TRAIN_EXTRA_ARGS="--skip-before-eval --epochs 2"
bash scripts/remote_full_pipeline.sh
```

离线无 Git 时：

```bash
export GIT_PULL=0
bash scripts/remote_full_pipeline.sh
```

---

## 0. 执行前必读（Agent 检查清单）

在运行任何训练命令前，请逐项确认：

| 序号 | 检查项 | 通过条件 |
|------|--------|----------|
| 1 | 仓库已克隆且路径已知 | 存在 `pipeline/large_model/train_qwen2_5_7b_qlora_demo.py` 与根目录 `config.py` |
| 2 | 基座模型已就绪 | 目录存在且含 **4 个** `model-*-of-00004.safetensors` 分片（脚本会校验） |
| 3 | 训练数据 JSONL 存在 | 默认使用 `output/llm_inputs_v4.jsonl`；若使用固定划分，还需 `output/train_split.jsonl` 与 `output/eval_split.jsonl`（仓库中通常已包含） |
| 4 | GPU 可用 | `nvidia-smi` 正常；`python -c "import torch; print(torch.cuda.is_available())"` 输出 `True` |
| 5 | 环境变量 | **强烈建议**设置 `GRADUATION_PROJECT_ROOT` 指向本仓库根目录（见下文） |

若任一项不满足，先补齐再训练，避免长时间报错空跑。

---

## 1. 仓库与代码更新

默认远程（若未改）：

```text
https://github.com/KikoStar123/ms-rca-llm.git
```

在服务器上（将 `REPO` 换成实际路径，例如 `/root/autodl-tmp/ms-rca-llm`）：

```bash
export REPO=/root/autodl-tmp/ms-rca-llm   # 按实际修改
cd "$REPO"
git pull --ff-only origin main
```

或使用仓库内脚本（需已设置 `GRADUATION_PROJECT_ROOT`）：

```bash
export GRADUATION_PROJECT_ROOT=/root/autodl-tmp/ms-rca-llm
bash scripts/git_pull_server.sh main
```

---

## 2. 环境变量（必须理解）

在**同一会话**中执行训练前，建议写入 `~/.bashrc` 或当前 shell：

```bash
# 仓库根（与 config.py 一致；克隆目录名不是 Graduation_Project 时必须设置）
export GRADUATION_PROJECT_ROOT=/root/autodl-tmp/ms-rca-llm

# 基座模型（二选一布局时指定其一即可）
# 方案 A：放在仓库内
export QWEN_MODEL_PATH=/root/autodl-tmp/ms-rca-llm/models/Qwen2.5-7B-Instruct
# 方案 B：共用 HuggingFace 缓存目录（示例）
# export QWEN_MODEL_PATH=/root/autodl-tmp/hf_cache/Qwen2.5-7B-Instruct

# 可选：限制 DataLoader 进程数（默认脚本内为 24；CPU 较少时请改小，例如 4 或 8）
# export NUM_WORKERS=8
```

说明：`config.py` 会优先读取 `GRADUATION_PROJECT_ROOT` 与 `QWEN_MODEL_PATH` / `HF_MODEL_PATH`，避免误用其它盘符或旧路径。

---

## 3. 基座模型与数据（不在 Git 中的部分）

以下内容 **未** 纳入 Git（见仓库根目录 `.gitignore`），需在人或上游流程中已放到服务器：

- **基座模型**：`Qwen2.5-7B-Instruct`，完整权重（含 tokenizer 等），且满足脚本内 `ensure_model()` 对 **4 个 safetensors 分片** 的检查。
- **原始大数据 `data/`**：仅当需要在服务器上**重新构建** SFT 数据时才必须；若仅复现训练，一般只需仓库内已有 `output/*.jsonl`。
- **检查点 `v2_doc/model_ckpt/`**：仅**断点续训**时需要；全新训练可不传。

训练默认读取（由 `config.DATA_PATH` 等解析）：

- `output/llm_inputs_v4.jsonl`
- 固定划分时：`output/train_split.jsonl`、`output/eval_split.jsonl`

---

## 4. Python 依赖

### 4.1 CUDA 版 PyTorch

请根据 **服务器 CUDA 版本** 从 [PyTorch 官网](https://pytorch.org/get-started/locally/) 选择安装命令，**先安装 torch**，再安装其余依赖。

示例（**仅作占位，勿照搬**；以官网生成命令为准）：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 4.2 项目依赖

在仓库根目录：

```bash
cd "$GRADUATION_PROJECT_ROOT"
pip install -r requirements-train.txt
```

### 4.3 快速自检

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())"
python -c "import transformers, peft, bitsandbytes; print('ok')"
```

---

## 5. 启动训练（主命令）

在仓库根目录执行：

```bash
cd "$GRADUATION_PROJECT_ROOT"
python pipeline/large_model/train_qwen2_5_7b_qlora_demo.py
```

### 5.1 常用可选参数（按需）

```text
--skip-before-eval     # 跳过微调前评估，节省时间
--epochs N             # 轮数（另有环境变量 NUM_TRAIN_EPOCHS 等，见脚本内说明）
--train-jsonl PATH --eval-jsonl PATH   # 显式指定训练/评估 JSONL
```

### 5.2 与数据划分相关的环境变量（与 README 一致）

- 使用仓库内固定划分：`output/train_split.jsonl` / `output/eval_split.jsonl` 时，一般**无需**改环境变量。
- 若需改回单次随机划分：可设置 `USE_FIXED_SPLIT=0`（详见项目 `README.md`）。

---

## 6. 输出位置（Agent 汇报用）

成功跑完后，通常在仓库的 `v2_doc/` 下生成带时间戳的文件，例如：

- `llm_eval_before_*.jsonl`、`llm_eval_after_*.jsonl`
- `训练评估结果_*.json`
- `训练评估报告_*.md`

LoRA 检查点目录由 `config` 中的 `MODEL_CKPT_DIR` 决定，一般为 `v2_doc/model_ckpt/` 下以 `checkpoint-*` 命名的子目录。

---

## 7. 故障排查（简要）

| 现象 | 可能原因 | 处理方向 |
|------|----------|----------|
| `模型路径不存在` / 分片不完整 | `QWEN_MODEL_PATH` 错误或权重未下全 | 核对路径、`ls` 分片文件数量与命名 |
| `import config` 后路径不对 | 未设置 `GRADUATION_PROJECT_ROOT` 或指向错误目录 | `export` 后从仓库根重跑 |
| CUDA OOM | batch 或序列长度过大 | 查阅脚本内 `TrainingArguments` / 环境变量是否有覆盖；或换更大显存实例 |
| CPU 占满、卡顿 | `NUM_WORKERS` 过大 | 设置 `export NUM_WORKERS=4`（或 8）后重试 |
| `bitsandbytes` 报错 | CUDA/PyTorch 版本不匹配 | 对齐 PyTorch 与 CUDA，重装 `bitsandbytes` |

---

## 8. 给 Agent 的执行序列

### 8A. 仅训练（仓库里已有 `output/*.jsonl` 划分）

将 `REPO` 与 `QWEN_MODEL_PATH` 改为服务器真实路径：

```bash
export REPO=/root/autodl-tmp/ms-rca-llm
export GRADUATION_PROJECT_ROOT="$REPO"
export QWEN_MODEL_PATH="$REPO/models/Qwen2.5-7B-Instruct"
# export NUM_WORKERS=8

cd "$REPO"
git pull --ff-only origin main

nvidia-smi
python -c "import torch; assert torch.cuda.is_available()"

test -d "$QWEN_MODEL_PATH" && ls "$QWEN_MODEL_PATH"/model-*-of-00004.safetensors 2>/dev/null | wc -l
test -f "$REPO/output/llm_inputs_v4.jsonl" && echo "data jsonl ok"

python pipeline/large_model/train_qwen2_5_7b_qlora_demo.py
```

### 8B. 全流程远程（数据构建 + 划分 + 训练）

见上文 **「全流程只在远程执行」** 与 `bash scripts/remote_full_pipeline.sh`。

---

**文档版本**：与仓库 `main` 分支同步；若命令与脚本不一致，以仓库内 `README.md` 与 `train_qwen2_5_7b_qlora_demo.py` 为准。
