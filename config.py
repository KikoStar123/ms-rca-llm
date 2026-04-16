"""
项目路径配置 - 支持本地与 AutoDL 线上环境。

优先级（从高到低）：
1. 环境变量 GRADUATION_PROJECT_ROOT — 指向仓库根目录（本地/服务器均可，避免多盘符时误用旧路径）
2. 服务器路径探测（/root/autodl-tmp/Graduation_Project 等）
3. 旧版本地 D:\\Graduation_Project
4. 当前 config.py 所在目录作为仓库根（如 Graduation_Project_online）

模型路径：QWEN_MODEL_PATH 或 HF_MODEL_PATH 优先；否则按环境默认。
"""
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent

_ENV_ROOT = os.environ.get("GRADUATION_PROJECT_ROOT", "").strip()


def _apply_repo_base(base_dir: str) -> None:
    """统一设置 data/output/v2_doc/model 等路径（相对仓库根）。"""
    global BASE_DIR, DATA_DIR, OUTPUT_DIR, RESULT_DIR, MODEL_CKPT_DIR, MODEL_PATH
    BASE_DIR = os.path.abspath(base_dir)
    _data_aiops = os.path.join(BASE_DIR, "data", "aiops2020")
    _data_plain = os.path.join(BASE_DIR, "data")
    DATA_DIR = _data_aiops if os.path.isdir(_data_aiops) else _data_plain
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    RESULT_DIR = os.path.join(BASE_DIR, "v2_doc")
    MODEL_CKPT_DIR = os.path.join(RESULT_DIR, "model_ckpt")
    _mp = os.environ.get("QWEN_MODEL_PATH") or os.environ.get("HF_MODEL_PATH")
    if _mp:
        MODEL_PATH = _mp.strip()
    else:
        MODEL_PATH = (
            r"D:\hf_cache\Qwen2.5-7B-Instruct"
            if os.path.isdir(r"D:\hf_cache\Qwen2.5-7B-Instruct")
            else os.path.join(BASE_DIR, "models", "Qwen2.5-7B-Instruct")
        )


# 1) 显式仓库根（推荐：本机多路径、或服务器自定义挂载）
if _ENV_ROOT and os.path.isdir(_ENV_ROOT) and Path(_ENV_ROOT, "pipeline").is_dir():
    _apply_repo_base(_ENV_ROOT)
# 2) AutoDL / Linux 服务器
elif os.path.exists("/root/autodl-tmp/Graduation_Project") or os.path.exists("/root/Graduation_Project"):
    if os.path.exists("/root/autodl-tmp/Graduation_Project"):
        BASE_DIR = "/root/autodl-tmp/Graduation_Project"
        MODEL_PATH = "/root/autodl-tmp/hf_cache/Qwen2.5-7B-Instruct"
        DATA_DIR = os.path.join(BASE_DIR, "data", "aiops2020")
        OUTPUT_DIR = os.path.join(BASE_DIR, "output")
        RESULT_DIR = os.path.join(BASE_DIR, "v2_doc")
        MODEL_CKPT_DIR = "/root/autodl-tmp/model_ckpt"
    else:
        BASE_DIR = "/root/Graduation_Project"
        MODEL_PATH = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"
        DATA_DIR = "/root/autodl-tmp/datasets/aiops2020_raw"
        OUTPUT_DIR = "/root/autodl-tmp/output"
        RESULT_DIR = os.path.join(BASE_DIR, "v2_doc")
        MODEL_CKPT_DIR = "/root/autodl-tmp/model_ckpt"
# 3) 旧版 Windows 本地盘符
elif os.path.exists(r"d:\Graduation_Project"):
    BASE_DIR = r"d:\Graduation_Project"
    MODEL_PATH = r"D:\hf_cache\Qwen2.5-7B-Instruct"
    DATA_DIR = os.path.join(BASE_DIR, "data")
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    RESULT_DIR = os.path.join(BASE_DIR, "v2_doc")
    MODEL_CKPT_DIR = os.path.join(RESULT_DIR, "model_ckpt")
# 4) 任意目录名的本仓库（如 Graduation_Project_online）
elif _REPO_ROOT.joinpath("pipeline").is_dir():
    _apply_repo_base(str(_REPO_ROOT))
else:
    raise RuntimeError(
        "无法识别运行环境。请设置环境变量 GRADUATION_PROJECT_ROOT 指向仓库根目录，"
        "或设置 QWEN_MODEL_PATH 指向 Qwen 权重目录。"
    )

# 训练数据路径（与 pipeline 产出一致）
DATA_PATH = os.path.join(OUTPUT_DIR, "llm_inputs_v4.jsonl")

# 所有分支赋值后统一 strip，避免环境变量尾随空格导致 os.path 找不到模型目录
if isinstance(MODEL_PATH, str):
    MODEL_PATH = MODEL_PATH.strip()
