# 仅微调：跳过微调前评估 + 「根因优先」保守超参（减轻过拟合、略增 LoRA 容量）
# 用法: .\scripts\train_finetune_only.ps1
# 日志: logs/train_finetune_only.log

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Repo

$env:GRADUATION_PROJECT_ROOT = $Repo
$env:QWEN_MODEL_PATH = Join-Path $Repo "models\Qwen2.5-7B-Instruct"
if (-not $env:NUM_WORKERS) { $env:NUM_WORKERS = "6" }

$env:SKIP_BEFORE_EVAL = "1"
$env:NUM_TRAIN_EPOCHS = "1"
$env:LEARNING_RATE = "3e-5"
$env:LORA_R = "24"
$env:LORA_ALPHA = "48"
$env:LORA_DROPOUT = "0.08"
$env:EARLY_STOPPING_PATIENCE = "4"
$env:WARMUP_RATIO = "0.08"
# 可选: $env:EVAL_OUTPUT_REPAIR = "1"  # 默认已是 1

New-Item -ItemType Directory -Path (Join-Path $Repo "logs") -Force | Out-Null
$Log = Join-Path $Repo "logs\train_finetune_only.log"
$ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content -Path $Log -Value "`n===== FINETUNE_ONLY START $ts =====`n" -Encoding utf8

$py = (Get-Command python).Source
$cmd = @"
chcp 65001>nul && cd /d "$Repo" && set GRADUATION_PROJECT_ROOT=$Repo&& set QWEN_MODEL_PATH=$($env:QWEN_MODEL_PATH)&& set NUM_WORKERS=$($env:NUM_WORKERS)&& set SKIP_BEFORE_EVAL=1&& set NUM_TRAIN_EPOCHS=1&& set LEARNING_RATE=3e-5&& set LORA_R=24&& set LORA_ALPHA=48&& set LORA_DROPOUT=0.08&& set EARLY_STOPPING_PATIENCE=4&& set WARMUP_RATIO=0.08&& "$py" -u pipeline\large_model\train_qwen2_5_7b_qlora_demo.py --skip-before-eval >> "$Log" 2>&1
"@
cmd.exe /c $cmd
$exit = $LASTEXITCODE
$ts2 = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content -Path $Log -Value "`n===== END $ts2 exit=$exit =====`n" -Encoding utf8
exit $exit
