"""
GRPO 第二阶段入口已迁至 `train_grpo_rca.py`（TRL GRPOTrainer + grpo_rewards.reward_rca_trl）。

依赖：pip install "trl>=0.15"（Windows 建议设置 PYTHONUTF8=1 再 import trl）。
文档：https://huggingface.co/docs/trl/grpo_trainer
"""
from __future__ import annotations

SFT_ADAPTER_HINT = "将 ADAPTER_PATH 指向 train_qwen2_5_7b_qlora_demo 产出的 LoRA 权重目录"


def describe_setup() -> str:
    return (
        "可执行训练：python pipeline/large_model/train_grpo_rca.py "
        "--adapter-path <SFT_LoRA目录> --train-jsonl output/train_split.jsonl\n"
        "奖励与字段权重见 grpo_rewards.py；评估仍用 evaluate_predictions_jsonl 与 eval_split。\n"
        f"{SFT_ADAPTER_HINT}"
    )


if __name__ == "__main__":
    print(describe_setup())
