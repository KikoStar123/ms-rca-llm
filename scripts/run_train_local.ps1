# 本机 GPU 训练（4070 SUPER 等）：避免 PowerShell 管道编码问题，日志 UTF-8 追加
# 用法: .\scripts\run_train_local.ps1
# 日志: logs/train_local_4070super.log

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Repo

$env:GRADUATION_PROJECT_ROOT = $Repo
$env:QWEN_MODEL_PATH = Join-Path $Repo "models\Qwen2.5-7B-Instruct"
if (-not $env:NUM_WORKERS) { $env:NUM_WORKERS = "6" }
# 可选环境变量示例：
#   $env:LEARNING_RATE = "5e-5"
#   $env:NUM_TRAIN_EPOCHS = "2"
#   $env:LORA_R = "16"; $env:LORA_ALPHA = "32"
#   $env:USE_BF16 = "0"   # 若训练数值异常可关 bf16
#   $env:USE_EVAL_FOR_EARLY_BEST = "0"
#   $env:EARLY_STOPPING_PATIENCE = "5"   # 0 表示不用早停

New-Item -ItemType Directory -Path (Join-Path $Repo "logs") -Force | Out-Null
$Log = Join-Path $Repo "logs\train_local_4070super.log"
$ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content -Path $Log -Value "`n===== START $ts =====`n" -Encoding utf8

# cmd 重定向避免 Tee 的 GBK/UTF-8 混用问题
$py = (Get-Command python).Source
$cmd = @"
chcp 65001>nul && cd /d "$Repo" && set GRADUATION_PROJECT_ROOT=$Repo && set QWEN_MODEL_PATH=$($env:QWEN_MODEL_PATH) && set NUM_WORKERS=$($env:NUM_WORKERS) && set LEARNING_RATE=$($env:LEARNING_RATE) && set USE_EVAL_FOR_EARLY_BEST=$($env:USE_EVAL_FOR_EARLY_BEST) && "$py" -u pipeline\large_model\train_qwen2_5_7b_qlora_demo.py >> "$Log" 2>&1
"@
cmd.exe /c $cmd
$exit = $LASTEXITCODE
$ts2 = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content -Path $Log -Value "`n===== END $ts2 exit=$exit =====`n" -Encoding utf8
exit $exit
