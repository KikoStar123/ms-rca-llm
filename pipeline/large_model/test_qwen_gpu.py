# test_qwen_gpu.py - 原生GPU加载版（无bitsandbytes依赖，适配RTX4070 8GB）
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import warnings
warnings.filterwarnings("ignore")

# 模型路径（与下载脚本一致，无需修改）
MODEL_PATH = r"D:\hf_cache\Qwen2.5-7B-Instruct"

# 1. 加载分词器（Qwen专属配置，保持不变）
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    use_fast=False,
    padding_side="right"
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. 原生GPU加载模型（关键：移除量化，添加显存优化配置）
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    device_map="auto",  # 自动分配模型到GPU
    torch_dtype=torch.bfloat16,  # 低精度加载，节省显存（8GB足够）
    low_cpu_mem_usage=True,  # 减少CPU内存占用
    attn_implementation="eager"  # 兼容模式，避免高版本接口问题
).eval()  # 推理模式，禁用梯度计算，节省显存

# 3. 测试微服务故障根因定位（贴合毕设场景）
prompt = """
任务：微服务故障根因定位
输入：故障现象为订单服务响应超时，多源证据显示docker组件CPU使用率98%（超阈值80%），小模型候选根因为docker、order-service、mysql；
要求：输出根因组件及简洁分析，格式清晰。
"""

# 构建输入并推送到GPU
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# 生成推理结果（低温度保证输出稳定）
outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.1,
    do_sample=False,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id
)

# 解码并打印结果
result = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 打印运行信息+推理结果
print("="*70)
print("✅ Python3.12+PyTorch2.10.0+cu130+RTX4070 环境配置成功！")
print(f"📌 模型运行设备：{model.device}（cuda:0 = NVIDIA GeForce RTX 4070 Laptop GPU）")
print(f"💾 模型显存占用：{torch.cuda.max_memory_allocated()/1024/1024/1024:.1f} GB / 8.0 GB")
print("="*70)
print("📝 微服务故障根因定位推理结果：\n")
print(result)
print("="*70)