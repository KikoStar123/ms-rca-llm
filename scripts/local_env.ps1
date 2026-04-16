# 本机一键设置仓库根目录（与 config.py、打包脚本一致）
# 使用: 在 PowerShell 中执行  . .\scripts\local_env.ps1
# 之后当前会话内 python 会优先使用本仓库路径；线上无需执行。

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:GRADUATION_PROJECT_ROOT = $RepoRoot

if (Test-Path "D:\hf_cache\Qwen2.5-7B-Instruct") {
    $env:QWEN_MODEL_PATH = "D:\hf_cache\Qwen2.5-7B-Instruct"
}
Write-Host "已设置 GRADUATION_PROJECT_ROOT=$RepoRoot" -ForegroundColor Green
if ($env:QWEN_MODEL_PATH) {
    Write-Host "已设置 QWEN_MODEL_PATH=$($env:QWEN_MODEL_PATH)" -ForegroundColor Green
}
