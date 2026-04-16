# SFT 训练配置模板（仅配置说明）

更新时间：2026-01-23

## 1. 数据路径
- 训练数据：`d:\Graduation_Project\output\llm_inputs_v4.jsonl`
- 格式：JSONL，每行包含 `instruction/input/output`

## 2. 推荐训练框架（可选其一）
以下是**配置要点**，不涉及实际训练执行：

### 方案 A：通用 LoRA SFT（示意）
- 基础模型：`<你的基础模型路径>`
- 训练类型：SFT + LoRA
- 关键超参建议：
  - 学习率：`1e-4`
  - batch_size：`1-2`（视显存）
  - epoch：`3-5`
  - max_seq_len：`2048-4096`
  - LoRA rank：`8`
  - LoRA alpha：`32`
  - LoRA dropout：`0.05`

### 方案 B：按字段拼接的指令微调（示意）
- prompt 模板：
  - 输入：`instruction` + `input`（序列化 JSON）
  - 输出：`output`（序列化 JSON）
- 训练目标：生成结构化 JSON 输出

## 3. 输出规范建议
要求模型输出字段：
- `root_cause_component`
- `fault_type`
- `kpi`
- `related_container`
- `explanation`（可选简短解释，用于可解释性）

## 4. 下一步（GRPO）
- 在 SFT 基础上引入奖励：
  - 输出稳定性（同类样本一致）
  - 证据一致性（解释引用 evidence）
  - 格式合规性（字段齐全、JSON 合法）
