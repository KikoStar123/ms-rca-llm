# GRPO 第二阶段：在 SFT LoRA 上继续训练（需已安装 trl>=0.15）
# 用法：先设置 $env:ADAPTER_PATH 指向 v2_doc\model_ckpt\checkpoint-XXX，再执行：
#   .\scripts\train_grpo.ps1
# Windows 建议 UTF-8，避免 trl 读模板失败：
#   $env:PYTHONUTF8 = "1"

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Repo

$env:GRADUATION_PROJECT_ROOT = $Repo
if (-not $env:PYTHONUTF8) { $env:PYTHONUTF8 = "1" }

if (-not $env:ADAPTER_PATH -or -not (Test-Path $env:ADAPTER_PATH)) {
    Write-Error "请先设置环境变量 ADAPTER_PATH 为 SFT 产出的 LoRA 目录（含 adapter_config.json）。"
}

New-Item -ItemType Directory -Path (Join-Path $Repo "logs") -Force | Out-Null
$Log = Join-Path $Repo "logs\train_grpo.log"
$ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content -Path $Log -Value "`n===== GRPO START $ts =====`n" -Encoding utf8

$py = (Get-Command python).Source
$ap = $env:ADAPTER_PATH
$cmd = "chcp 65001>nul && cd /d `"$Repo`" && set GRADUATION_PROJECT_ROOT=$Repo&& set PYTHONUTF8=1&& set ADAPTER_PATH=$ap&& `"$py`" -u pipeline\large_model\train_grpo_rca.py --adapter-path `"$ap`" --train-jsonl output\train_split.jsonl >> `"$Log`" 2>&1"
cmd.exe /c $cmd
$exit = $LASTEXITCODE
$ts2 = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content -Path $Log -Value "`n===== END $ts2 exit=$exit =====`n" -Encoding utf8
exit $exit
