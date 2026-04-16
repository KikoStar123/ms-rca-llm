# 检查上传打包进度
$TempDir = "d:\Graduation_Project_upload_temp"
$ZipFile = "d:\Graduation_Project_upload.zip"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "上传打包进度检查" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

if (Test-Path $TempDir) {
    Write-Host "`n临时目录: $TempDir" -ForegroundColor Green
    
    # 检查各目录
    $dirs = @("pipeline", "research", "v2_doc", "data", "output", "models")
    foreach ($dir in $dirs) {
        $path = Join-Path $TempDir $dir
        if (Test-Path $path) {
            $size = (Get-ChildItem $path -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum / 1GB
            Write-Host "  ✓ $dir : $([math]::Round($size, 2)) GB" -ForegroundColor Gray
        } else {
            Write-Host "  ⚠ $dir : 未找到" -ForegroundColor Yellow
        }
    }
    
    $totalSize = (Get-ChildItem $TempDir -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum / 1GB
    Write-Host "`n总大小: $([math]::Round($totalSize, 2)) GB" -ForegroundColor Cyan
} else {
    Write-Host "临时目录不存在，打包可能已完成或未开始" -ForegroundColor Yellow
}

if (Test-Path $ZipFile) {
    $zipSize = (Get-Item $ZipFile).Length / 1GB
    Write-Host "`n压缩包: $ZipFile" -ForegroundColor Green
    Write-Host "  大小: $([math]::Round($zipSize, 2)) GB" -ForegroundColor Cyan
    Write-Host "  状态: ✓ 已生成，可以上传" -ForegroundColor Green
} else {
    Write-Host "`n压缩包: 尚未生成" -ForegroundColor Yellow
    Write-Host "  状态: 文件复制可能仍在进行中..." -ForegroundColor Yellow
}

Write-Host "`n========================================" -ForegroundColor Cyan
