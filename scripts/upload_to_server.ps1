# AutoDL 服务器完整上传脚本
# 使用方法: .\upload_to_server.ps1 -ServerIP "xxx" -ServerUser "root" -ServerPassword "xxx"

param(
    [Parameter(Mandatory=$true)]
    [string]$ServerIP,
    
    [Parameter(Mandatory=$false)]
    [string]$ServerUser = "root",
    
    [Parameter(Mandatory=$false)]
    [string]$ServerPassword = "",
    
    [Parameter(Mandatory=$false)]
    [string]$ServerPort = "22",

    [Parameter(Mandatory=$false)]
    [string]$ProjectRoot = "",

    [Parameter(Mandatory=$false)]
    [string]$ModelPath = ""
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "AutoDL 服务器完整上传脚本" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# 检查必要工具
if (-not (Get-Command scp -ErrorAction SilentlyContinue)) {
    Write-Host "错误: 未找到 scp 命令，请安装 OpenSSH 客户端" -ForegroundColor Red
    Write-Host "Windows 10/11: 设置 -> 应用 -> 可选功能 -> OpenSSH 客户端" -ForegroundColor Yellow
    exit 1
}

if (-not $ProjectRoot) {
    if ($env:GRADUATION_PROJECT_ROOT) { $ProjectRoot = $env:GRADUATION_PROJECT_ROOT }
    else { $ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path }
}
if (-not $ModelPath) {
    if ($env:QWEN_MODEL_PATH) { $ModelPath = $env:QWEN_MODEL_PATH }
    elseif (Test-Path "D:\hf_cache\Qwen2.5-7B-Instruct") { $ModelPath = "D:\hf_cache\Qwen2.5-7B-Instruct" }
    else { $ModelPath = Join-Path $ProjectRoot "models\Qwen2.5-7B-Instruct" }
}
$TempDir = Join-Path $ProjectRoot "_upload_temp"
Write-Host "ProjectRoot: $ProjectRoot" -ForegroundColor Gray

# 清理临时目录
if (Test-Path $TempDir) {
    Remove-Item -Path $TempDir -Recurse -Force
}
New-Item -ItemType Directory -Path $TempDir | Out-Null

Write-Host "`n[1/5] 准备上传文件..." -ForegroundColor Green

# 1. 复制项目代码（到系统盘）
Write-Host "  - 复制代码文件..." -ForegroundColor Yellow
$CodeDirs = @("pipeline", "research", "v2_doc")
foreach ($dir in $CodeDirs) {
    $src = Join-Path $ProjectRoot $dir
    $dst = Join-Path $TempDir $dir
    if (Test-Path $src) {
        Copy-Item -Path $src -Destination $dst -Recurse -Force
    }
}

# 复制单个文件
$CodeFiles = @("*.py", "*.md", "*.pdf", "*.txt", "*.json", "*.csv", "*.html")
foreach ($pattern in $CodeFiles) {
    Get-ChildItem -Path $ProjectRoot -Filter $pattern -File | ForEach-Object {
        Copy-Item -Path $_.FullName -Destination $TempDir -Force
    }
}

# 2. 复制数据集（到数据盘）
Write-Host "  - 复制数据集文件..." -ForegroundColor Yellow
$DataSrc = Join-Path $ProjectRoot "data"
$DataDst = Join-Path $TempDir "data"
if (Test-Path $DataSrc) {
    Copy-Item -Path $DataSrc -Destination $DataDst -Recurse -Force
}

# 3. 复制输出文件（到数据盘）
Write-Host "  - 复制输出文件..." -ForegroundColor Yellow
$OutputSrc = Join-Path $ProjectRoot "output"
$OutputDst = Join-Path $TempDir "output"
if (Test-Path $OutputSrc) {
    Copy-Item -Path $OutputSrc -Destination $OutputDst -Recurse -Force
}

# 4. 复制模型文件（到数据盘）
Write-Host "  - 复制模型文件（这可能需要几分钟）..." -ForegroundColor Yellow
$ModelDst = Join-Path $TempDir "models\Qwen2.5-7B-Instruct"
if (Test-Path $ModelPath) {
    New-Item -ItemType Directory -Path (Split-Path $ModelDst) -Force | Out-Null
    Copy-Item -Path $ModelPath -Destination $ModelDst -Recurse -Force
} else {
    Write-Host "  警告: 未找到模型文件 $ModelPath" -ForegroundColor Yellow
}

Write-Host "`n[2/5] 压缩文件..." -ForegroundColor Green
$ZipFile = Join-Path $ProjectRoot "Graduation_Project_upload.zip"
if (Test-Path $ZipFile) {
    Remove-Item -Path $ZipFile -Force
}

Compress-Archive -Path $TempDir\* -DestinationPath $ZipFile -CompressionLevel Optimal

$ZipSize = (Get-Item $ZipFile).Length / 1GB
Write-Host "  压缩完成: $ZipFile ($([math]::Round($ZipSize, 2)) GB)" -ForegroundColor Cyan

Write-Host "`n[3/5] 上传到服务器..." -ForegroundColor Green

# 构建 SCP 命令
$RemoteCodePath = "/root/Graduation_Project"
$RemoteDataPath = "/root/autodl-tmp"

# 如果使用密码，需要先设置 SSH_ASKPASS 或使用 sshpass（Windows 需要额外安装）
# 这里假设你已经配置了 SSH 密钥认证，或者手动输入密码

Write-Host "  上传代码到系统盘: $RemoteCodePath" -ForegroundColor Yellow
$CodeFilesOnly = Join-Path $TempDir "*"
$ScpCommand1 = "scp -r -P $ServerPort `"$CodeFilesOnly`" ${ServerUser}@${ServerIP}:$RemoteCodePath/"

Write-Host "  执行命令: $ScpCommand1" -ForegroundColor Gray
Write-Host "  注意: 如果提示输入密码，请输入服务器密码" -ForegroundColor Yellow

# 实际上传（需要用户手动执行或配置密钥）
Write-Host "`n[4/5] 手动上传步骤:" -ForegroundColor Cyan
Write-Host "  1. 使用 AUTODL 网页的文件上传功能上传: $ZipFile" -ForegroundColor White
Write-Host "  2. 或者使用以下命令（需要配置 SSH 密钥）:" -ForegroundColor White
Write-Host "     scp -P $ServerPort `"$ZipFile`" ${ServerUser}@${ServerIP}:/root/" -ForegroundColor Gray
Write-Host "`n  3. 在服务器上解压:" -ForegroundColor White
Write-Host "     ssh ${ServerUser}@${ServerIP} -p $ServerPort" -ForegroundColor Gray
Write-Host "     cd /root" -ForegroundColor Gray
Write-Host "     unzip -q Graduation_Project_upload.zip" -ForegroundColor Gray
Write-Host "     mv Graduation_Project/* /root/Graduation_Project/" -ForegroundColor Gray
Write-Host "     mkdir -p /root/autodl-tmp/datasets /root/autodl-tmp/output /root/autodl-tmp/models" -ForegroundColor Gray
Write-Host "     mv data /root/autodl-tmp/datasets/aiops2020_raw" -ForegroundColor Gray
Write-Host "     mv output /root/autodl-tmp/output" -ForegroundColor Gray
Write-Host "     mv models /root/autodl-tmp/models" -ForegroundColor Gray

Write-Host "`n[5/5] 清理临时文件..." -ForegroundColor Green
Remove-Item -Path $TempDir -Recurse -Force
Write-Host "  完成!" -ForegroundColor Green

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "上传准备完成！" -ForegroundColor Cyan
Write-Host "压缩包位置: $ZipFile" -ForegroundColor Cyan
Write-Host "大小: $([math]::Round($ZipSize, 2)) GB" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
