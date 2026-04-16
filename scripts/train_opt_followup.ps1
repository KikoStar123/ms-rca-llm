# 对照实验：在 20260412 成功配置基础上增加 epoch=2，其余尽量保持一致
# 用法：在仓库根目录  .\scripts\train_opt_followup.ps1

$ErrorActionPreference = "Stop"
$Root = if ($env:GRADUATION_PROJECT_ROOT) { $env:GRADUATION_PROJECT_ROOT } else { (Resolve-Path (Join-Path $PSScriptRoot "..")).Path }
Set-Location $Root

$env:NUM_TRAIN_EPOCHS = "2"
$env:LEARNING_RATE = "3e-5"
$env:LORA_R = "24"
$env:LORA_ALPHA = "48"
$env:LORA_DROPOUT = "0.08"
$env:GRADIENT_ACCUMULATION_STEPS = "4"
$env:WARMUP_RATIO = "0.08"
$env:WEIGHT_DECAY = "0.01"
$env:MAX_GRAD_NORM = "1.0"
$env:LR_SCHEDULER_TYPE = "cosine"
$env:USE_BF16 = "1"
$env:USE_EVAL_FOR_EARLY_BEST = "1"
$env:EARLY_STOPPING_PATIENCE = "4"
$env:SKIP_BEFORE_EVAL = "1"

# CPU 核心不多时可改小
if (-not $env:NUM_WORKERS) { $env:NUM_WORKERS = "8" }

Write-Host "GRADUATION_PROJECT_ROOT=$Root" -ForegroundColor Cyan
Write-Host "NUM_TRAIN_EPOCHS=$($env:NUM_TRAIN_EPOCHS) LEARNING_RATE=$($env:LEARNING_RATE) LORA $($env:LORA_R)/$($env:LORA_ALPHA)" -ForegroundColor Cyan

python pipeline/large_model/train_qwen2_5_7b_qlora_demo.py --skip-before-eval --epochs 2
