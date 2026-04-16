"""
一键列出或执行 RCA 流水线步骤（数据 → SFT 样本 → LLM 输入 → 划分 → 训练/评估）。

用法（在仓库根目录）:
  python pipeline/run_rca_pipeline.py list
  python pipeline/run_rca_pipeline.py split --input output/llm_inputs_v4.jsonl
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def run(cmd: list[str], *, check: bool = True) -> int:
    print("$ " + " ".join(cmd))
    p = subprocess.run(cmd, cwd=ROOT)
    if check and p.returncode != 0:
        sys.exit(p.returncode)
    return p.returncode


def cmd_list() -> None:
    print(
        """
## 推荐顺序（与 docs/后续工作清单.md 一致）

1. 解压原始日数据（如需要）  
   python pipeline/extract_day.py --date 2020-04-11

2. 生成单日 SFT 样本  
   python pipeline/build_sft_samples.py --date 2020-04-11

3. 批量多日样本 + 合并（见 pipeline/batch_build_sft.py、merge_jsonl.py）

4. 构建大模型输入（含小模型 Top-K）  
   python pipeline/build_llm_inputs.py --input-dir output --output output/llm_inputs_v4.jsonl

5. 固定划分训练/评估集（与 train 脚本一致）  
   python pipeline/split_dataset.py --input output/llm_inputs_v4.jsonl
        --train-out output/train_split.jsonl --eval-out output/eval_split.jsonl
        --manifest output/split_manifest.json

6. 统计 SFT 分布（可选）  
   python pipeline/sft_stats.py --input output/llm_inputs_v4.jsonl --output output/sft_stats_merged.json

7. QLoRA 训练 + 评估（GPU，需 Qwen 权重）  
   python pipeline/large_model/train_qwen2_5_7b_qlora_demo.py  
   若存在 output/train_split.jsonl 与 eval_split.jsonl，将自动使用固定划分（与 split_dataset 一致）。  
   若要恢复旧行为（从 llm_inputs_v4 现 shuffle 切分），请设环境变量 USE_FIXED_SPLIT=0

8. 离线评估某次推理结果 JSONL（需与金标准对齐 Top-K 指标时加 --gold）  
   python pipeline/evaluate_predictions_jsonl.py --pred v2_doc/llm_eval_after_xxx.jsonl
        --gold output/eval_split.jsonl --output v2_doc/metrics_offline.json

9. 演示推理（无 GPU）  
   python pipeline/inference_cli.py --mock --input research/demo_inference_sft_sample.json

10. GRPO 第二阶段：见 pipeline/large_model/grpo_train_skeleton.py 与 grpo_rewards.py

配置与默认参数见 config/experiment_defaults.yaml；路径见根目录 config.py（支持 Graduation_Project_online）。
"""
    )


def cmd_split(ns: argparse.Namespace) -> None:
    inp = ns.input or "output/llm_inputs_v4.jsonl"
    run(
        [
            PY,
            str(ROOT / "pipeline" / "split_dataset.py"),
            "--input",
            inp,
            "--train-out",
            ns.train_out,
            "--eval-out",
            ns.eval_out,
            "--manifest",
            ns.manifest,
            "--seed",
            str(ns.seed),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="RCA 流水线编排")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="打印推荐步骤与命令")

    p_split = sub.add_parser("split", help="调用 split_dataset 划分数据")
    p_split.add_argument("--input", default="output/llm_inputs_v4.jsonl")
    p_split.add_argument("--train-out", default="output/train_split.jsonl")
    p_split.add_argument("--eval-out", default="output/eval_split.jsonl")
    p_split.add_argument("--manifest", default="output/split_manifest.json")
    p_split.add_argument("--seed", type=int, default=42)
    p_split.set_defaults(func=cmd_split)

    args = parser.parse_args()
    if args.command == "list":
        cmd_list()
    elif args.command == "split":
        cmd_split(args)


if __name__ == "__main__":
    main()
