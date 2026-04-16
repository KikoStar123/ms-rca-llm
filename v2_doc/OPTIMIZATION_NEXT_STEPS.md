# 优化下一步（已执行第 1 步：候选覆盖率）

## 第 1 步：评估集金标 vs `top_candidates`（已完成）

运行：

```bash
python pipeline/analyze_eval_candidate_coverage.py --jsonl output/eval_split.jsonl --out-json v2_doc/optimization_candidate_coverage_eval.json
```

### 结果摘要（`output/eval_split.jsonl`，N=200）

| 指标 | 值 | 含义 |
|------|-----|------|
| 金标在候选列表**任意位置** | 73 / 200（36.5%） | 仅这些样本在「严格从候选中选根因」时是**可解**的 |
| 金标在 **Top-3** 内 | 40 / 200（**20%**） | 与训练报告里「根因 Top‑3：0.200」一致 |
| 金标在 **Top-5** 内 | 73 / 200（**36.5%**） | 与「根因 Top‑5：0.365」一致（本数据候选长度多为 5） |
| 金标**不在**候选列表中 | **127 / 200（63.5%）** | 若强制只能选候选，则这些题**不存在**「选对金标」的路径 |

详细 JSON：`v2_doc/optimization_candidate_coverage_eval.json`。

### 与根因 Top‑1（约 0.645）的关系

训练评估里的 **Top‑1** 是「预测字符串是否等于金标」，**不要求**预测必须在 `top_candidates` 内。模型有可能输出与金标一致的组件名，即使该名称未出现在候选列表中（与「必须从候选选」的指令存在张力）。因此 **Top‑1 可以高于「金标在候选内」的比例**。

**结论（优先级）：**

1. **上游候选质量**：约 **2/3** 评估样本的金标根因未进入 Top‑5 候选，长期提升应改进 **小模型 Top‑K / 加权、窗口、候选数**（`build_llm_inputs` 等链路）。
2. **继续 SFT**：在现有数据上可跑 **第 2 步对照训练**（见下方脚本），观察验证集与根因 Top‑1 是否仍提升；若过拟合则减小 epoch 或略降学习率。

---

## 已在代码中修正：候选列表（`build_llm_inputs.py`）

针对「小模型 Top‑5 常为 docker，而金标为 db/os 等」导致的漏召，已做两点默认改动（**不使用金标**，推理阶段同样适用）：

1. **将小模型排序候选从 Top‑5 扩展到 Top‑10**（`--topk-max`，默认 10）。
2. **将 `fault_event.name`（告警对象组件 ID）插入候选最前并去重**，与加权 Top‑K 合并后截断到 `topk_max` 条。

关闭注入（仅对比实验）：`python pipeline/build_llm_inputs.py ... --no-inject-fault-name`

**你需要重新生成数据并划分**，再重新训练，指标才与旧版 JSONL 不可直接横比：

```bash
# 仓库根目录
python pipeline/build_llm_inputs.py --input-dir output --output output/llm_inputs_v4.jsonl
python pipeline/split_dataset.py --input output/llm_inputs_v4.jsonl ^
  --train-out output/train_split.jsonl --eval-out output/eval_split.jsonl --manifest output/split_manifest.json
```

（Linux 将 `^` 换为行末 `\`。）然后可再跑：

`python pipeline/analyze_eval_candidate_coverage.py --jsonl output/eval_split.jsonl`

查看 `gold_not_in_candidates` 是否下降。

---

## 第 2 步：对照训练（待你在 GPU 上执行）

与 `训练评估结果_20260412_011825.json` 中配置对齐，仅将 **epoch 从 1 提到 2**（仍可用验证集 early best）。本机可用：

```powershell
.\scripts\train_opt_followup.ps1
```

Linux / AutoDL：

```bash
bash scripts/train_opt_followup.sh
```

完成后对比新生成的 `v2_doc/训练评估结果_*.json` 与上一份报告。

---

## 第 3 步（可选）：纯模型字段指标

需要对比「repair 关」时的 type/kpi/container 时，在运行评估前设置 `EVAL_OUTPUT_REPAIR=0`（见 `prediction_repair.py` / 训练脚本环境变量说明）。
