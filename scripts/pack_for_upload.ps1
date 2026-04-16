# 快速打包脚本 - 只打包，不上传
# 使用方法: .\pack_for_upload.ps1
# 可选: -ProjectRoot "E:\Graduation_Project_online"
# 或先: . .\scripts\local_env.ps1

param(
    [string]$ProjectRoot = "",
    [string]$ModelPath = ""
)

$ErrorActionPreference = "Stop"

if (-not $ProjectRoot) {
    if ($env:GRADUATION_PROJECT_ROOT) {
        $ProjectRoot = $env:GRADUATION_PROJECT_ROOT
    } else {
        $ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
    }
}
if (-not $ModelPath) {
    if ($env:QWEN_MODEL_PATH) {
        $ModelPath = $env:QWEN_MODEL_PATH
    } elseif (Test-Path "D:\hf_cache\Qwen2.5-7B-Instruct") {
        $ModelPath = "D:\hf_cache\Qwen2.5-7B-Instruct"
    } else {
        $ModelPath = Join-Path $ProjectRoot "models\Qwen2.5-7B-Instruct"
    }
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "项目打包脚本" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "ProjectRoot: $ProjectRoot" -ForegroundColor Gray
Write-Host "ModelPath:   $ModelPath" -ForegroundColor Gray

$TempDir = Join-Path $ProjectRoot "_upload_temp"
$ZipFile = Join-Path $ProjectRoot "Graduation_Project_upload.zip"

# 清理临时目录和旧压缩包
if (Test-Path $TempDir) {
    Write-Host "清理旧临时目录..." -ForegroundColor Yellow
    Remove-Item -Path $TempDir -Recurse -Force
}
if (Test-Path $ZipFile) {
    Write-Host "删除旧压缩包..." -ForegroundColor Yellow
    Remove-Item -Path $ZipFile -Force
}

New-Item -ItemType Directory -Path $TempDir | Out-Null

Write-Host "`n[1/5] 复制代码文件..." -ForegroundColor Green

# 1. 复制代码目录
$CodeDirs = @("pipeline", "research", "v2_doc")
foreach ($dir in $CodeDirs) {
    $src = Join-Path $ProjectRoot $dir
    $dst = Join-Path $TempDir $dir
    if (Test-Path $src) {
        Write-Host "  - $dir" -ForegroundColor Gray
        Copy-Item -Path $src -Destination $dst -Recurse -Force
    }
}

# 2. 复制单个代码文件
Write-Host "  - 单个文件 (*.py, *.md, *.pdf, *.txt, *.json, *.csv, *.html, *.sh)" -ForegroundColor Gray
$CodeFiles = @("*.py", "*.md", "*.pdf", "*.txt", "*.json", "*.csv", "*.html", "*.sh")
foreach ($pattern in $CodeFiles) {
    Get-ChildItem -Path $ProjectRoot -Filter $pattern -File -ErrorAction SilentlyContinue | ForEach-Object {
        Copy-Item -Path $_.FullName -Destination $TempDir -Force
    }
}

Write-Host "`n[2/5] 复制数据集文件..." -ForegroundColor Green
$DataSrc = Join-Path $ProjectRoot "data"
$DataDst = Join-Path $TempDir "data"
if (Test-Path $DataSrc) {
    Write-Host "  - 复制 data 目录（这可能需要几分钟）..." -ForegroundColor Yellow
    Copy-Item -Path $DataSrc -Destination $DataDst -Recurse -Force
    $DataSize = (Get-ChildItem -Path $DataDst -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1GB
    Write-Host "  ✓ 数据集大小: $([math]::Round($DataSize, 2)) GB" -ForegroundColor Cyan
} else {
    Write-Host "  警告: 未找到 data 目录" -ForegroundColor Yellow
}

Write-Host "`n[3/5] 复制输出文件..." -ForegroundColor Green
$OutputSrc = Join-Path $ProjectRoot "output"
$OutputDst = Join-Path $TempDir "output"
if (Test-Path $OutputSrc) {
    Write-Host "  - 复制 output 目录..." -ForegroundColor Yellow
    Copy-Item -Path $OutputSrc -Destination $OutputDst -Recurse -Force
    $OutputSize = (Get-ChildItem -Path $OutputDst -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1GB
    Write-Host "  ✓ 输出文件大小: $([math]::Round($OutputSize, 2)) GB" -ForegroundColor Cyan
} else {
    Write-Host "  警告: 未找到 output 目录" -ForegroundColor Yellow
}

Write-Host "`n[4/5] 复制模型文件（这可能需要较长时间）..." -ForegroundColor Green
$ModelDst = Join-Path $TempDir "models\Qwen2.5-7B-Instruct"
if (Test-Path $ModelPath) {
    Write-Host "  - 复制模型文件: $ModelPath" -ForegroundColor Yellow
    Write-Host "    目标: $ModelDst" -ForegroundColor Gray
    New-Item -ItemType Directory -Path (Split-Path $ModelDst) -Force | Out-Null
    
    # 显示进度
    $ModelSize = (Get-ChildItem -Path $ModelPath -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1GB
    Write-Host "    模型大小: $([math]::Round($ModelSize, 2)) GB" -ForegroundColor Gray
    Write-Host "    开始复制（请耐心等待）..." -ForegroundColor Yellow
    
    Copy-Item -Path $ModelPath -Destination $ModelDst -Recurse -Force
    
    Write-Host "  ✓ 模型复制完成" -ForegroundColor Green
} else {
    Write-Host "  警告: 未找到模型文件 $ModelPath" -ForegroundColor Yellow
    Write-Host "  请确认模型路径是否正确" -ForegroundColor Yellow
}

Write-Host "`n[5/5] 压缩文件..." -ForegroundColor Green
Write-Host "  正在压缩（这可能需要10-30分钟，取决于文件大小）..." -ForegroundColor Yellow

$StartTime = Get-Date
Compress-Archive -Path $TempDir\* -DestinationPath $ZipFile -CompressionLevel Optimal
$EndTime = Get-Date
$Duration = ($EndTime - $StartTime).TotalMinutes

$ZipSize = (Get-Item $ZipFile).Length / 1GB
Write-Host "  ✓ 压缩完成！" -ForegroundColor Green
Write-Host "    耗时: $([math]::Round($Duration, 1)) 分钟" -ForegroundColor Cyan
Write-Host "    文件: $ZipFile" -ForegroundColor Cyan
Write-Host "    大小: $([math]::Round($ZipSize, 2)) GB" -ForegroundColor Cyan

Write-Host "`n清理临时文件..." -ForegroundColor Green
Remove-Item -Path $TempDir -Recurse -Force

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "打包完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "压缩包位置: $ZipFile" -ForegroundColor White
Write-Host "大小: $([math]::Round($ZipSize, 2)) GB" -ForegroundColor White
Write-Host "`n下一步:" -ForegroundColor Yellow
Write-Host "1. 使用 AUTODL 网页上传功能上传此文件到 /root/" -ForegroundColor White
Write-Host "2. 或使用 SCP: scp `"$ZipFile`" root@服务器IP:/root/" -ForegroundColor White
Write-Host "3. 在服务器上执行: bash deploy_on_server.sh" -ForegroundColor White
Write-Host "========================================" -ForegroundColor Cyan
