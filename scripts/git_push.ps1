# 本机一键：git add ->（有改动则）commit -> push
# 用法：
#   .\scripts\git_push.ps1 -Message "说明本次改动"
#   .\scripts\git_push.ps1                    # 使用默认提交说明（含时间戳）
#   .\scripts\git_push.ps1 -DryRun            # 仅查看 status，不提交不推送

param(
    [string]$Message = "",
    [string]$Branch = "main",
    [string]$Remote = "origin",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
if ($env:GRADUATION_PROJECT_ROOT) {
    $RepoRoot = $env:GRADUATION_PROJECT_ROOT
} else {
    $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}
Set-Location $RepoRoot

if (-not (Test-Path (Join-Path $RepoRoot ".git"))) {
    Write-Host "错误: 不是 git 仓库: $RepoRoot" -ForegroundColor Red
    exit 1
}

Write-Host "仓库: $RepoRoot" -ForegroundColor Cyan
git status -sb

if ($DryRun) {
    Write-Host "`n[DryRun] 未执行 add/commit/push" -ForegroundColor Yellow
    exit 0
}

git add -A
git diff --cached --quiet 2>$null
$hasStaged = ($LASTEXITCODE -ne 0)

if ($hasStaged) {
    if (-not $Message) {
        $ts = Get-Date -Format "yyyy-MM-dd HH:mm"
        $Message = "chore: sync $ts"
    }
    git commit -m $Message
    if ($LASTEXITCODE -ne 0) {
        Write-Host "commit 失败" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "`n暂存区无新改动（跳过 commit）。" -ForegroundColor Gray
}

git push $Remote $Branch
if ($LASTEXITCODE -ne 0) {
    Write-Host "提示: 若尚未设置上游分支，可执行: git push -u $Remote $Branch" -ForegroundColor Yellow
    exit $LASTEXITCODE
}
Write-Host "`n完成 push -> ${Remote}/${Branch}" -ForegroundColor Green
