# 训练监控指南

## 训练已启动！

训练进程已在后台运行，日志文件会自动生成。

## 监控方法

### 1. 实时查看训练日志（推荐）

```bash
# 找到最新的日志文件
cd /root/autodl-tmp/Graduation_Project
LOG_FILE=$(ls -t training_*.log | head -1)

# 实时查看日志（类似tail -f）
tail -f $LOG_FILE
```

**或者直接使用：**
```bash
tail -f /root/autodl-tmp/Graduation_Project/training_*.log
```

### 2. 查看训练Loss和进度

在日志中查找以下关键信息：

#### 训练Loss（每5步记录一次）
```bash
# 查看最近的Loss
tail -100 training_*.log | grep -E "loss|grad_norm|learning_rate"
```

**输出示例：**
```
{'loss': '2.345', 'grad_norm': '0.123', 'learning_rate': '0.0002', 'epoch': '0.1'}
{'loss': '2.123', 'grad_norm': '0.115', 'learning_rate': '0.000198', 'epoch': '0.2'}
```

#### 训练进度
```bash
# 查看训练进度条
tail -50 training_*.log | grep -E "[0-9]+/[0-9]+|%|step"
```

**输出示例：**
```
10%|█         | 38/382 [01:23<08:42, 1.37s/it]
20%|██        | 76/382 [02:46<08:14, 1.37s/it]
```

### 3. 查看GPU使用情况

```bash
# 实时监控GPU
watch -n 1 nvidia-smi

# 或者一次性查看
nvidia-smi
```

**关键指标：**
- `GPU-Util`: GPU利用率（训练时应该接近100%）
- `Memory-Usage`: 显存使用（8-bit量化应该约12-15GB）
- `Power`: 功耗（RTX 5090应该较高）

### 4. 查看CPU使用情况

```bash
# 实时监控CPU（应该看到24个核心都在工作）
top
# 或者
htop
```

**多进程优化效果：**
- 应该看到多个Python进程在运行（tokenization阶段）
- CPU使用率应该接近100%（24核心）

### 5. 查看训练进程状态

```bash
# 查看训练进程
ps aux | grep train_qwen2_5_7b_qlora_demo.py | grep -v grep

# 查看进程资源使用
top -p $(pgrep -f train_qwen2_5_7b_qlora_demo.py)
```

### 6. 快速检查训练是否正常

```bash
# 一键检查脚本
cd /root/autodl-tmp/Graduation_Project
python3 << 'EOF'
import subprocess
from pathlib import Path
import time

print("=" * 70)
print("训练状态检查")
print("=" * 70)

# 1. 检查进程
result = subprocess.run(['pgrep', '-f', 'train_qwen2_5_7b_qlora_demo.py'], 
                       capture_output=True, text=True)
if result.returncode == 0:
    print("✅ 训练进程正在运行")
    pid = result.stdout.strip().split()[0]
    print(f"   PID: {pid}")
else:
    print("❌ 训练进程未运行")

# 2. 检查最新日志
log_files = sorted(Path('.').glob('training_*.log'), key=lambda x: x.stat().st_mtime, reverse=True)
if log_files:
    latest_log = log_files[0]
    print(f"\n✅ 最新日志: {latest_log.name}")
    
    # 读取最后几行
    with open(latest_log, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        if lines:
            print(f"\n最后5行日志:")
            for line in lines[-5:]:
                print(f"  {line.strip()}")
            
            # 检查Loss
            loss_lines = [l for l in lines if 'loss' in l.lower() and '{' in l]
            if loss_lines:
                print(f"\n✅ 发现Loss记录: {len(loss_lines)} 条")
                print(f"   最新Loss: {loss_lines[-1].strip()}")
            else:
                print(f"\n⏳ 尚未开始记录Loss（可能还在数据准备阶段）")
else:
    print("\n❌ 未找到日志文件")

# 3. 检查GPU
result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', 
                        '--format=csv,noheader'], capture_output=True, text=True)
if result.returncode == 0:
    print(f"\n✅ GPU状态:")
    print(f"   {result.stdout.strip()}")
else:
    print("\n⚠️  无法获取GPU状态")

print("\n" + "=" * 70)
EOF
```

## 关键指标说明

### Loss（损失值）
- **正常范围**: 训练开始时应该在2-5之间，逐渐下降
- **异常情况**: 
  - Loss = 0 → 训练配置有问题（已修复）
  - Loss = NaN → 数值不稳定（已修复）
  - Loss不下降 → 学习率可能太小

### Grad Norm（梯度范数）
- **正常范围**: 0.1 - 10之间
- **异常情况**:
  - Grad Norm = NaN → 梯度爆炸或数值不稳定（已修复）
  - Grad Norm = 0 → 梯度消失

### Learning Rate（学习率）
- **初始值**: 0.0002 (2e-4)
- **变化**: 会随着训练步数逐渐衰减

### Epoch（轮次）
- **总步数**: 约382步（1528样本 / 4 batch size）
- **1个epoch**: 约382步

## 训练阶段

1. **数据准备阶段**（约1-2分钟）
   - 多进程tokenization（24核心）
   - 应该看到CPU使用率很高

2. **训练前评估阶段**（约2-5分钟）
   - 多进程预处理 + 批量推理
   - GPU利用率应该较高

3. **训练阶段**（约8-10分钟）
   - 每5步记录一次Loss
   - GPU利用率应该接近100%

4. **训练后评估阶段**（约2-5分钟）
   - 多进程预处理 + 批量推理
   - GPU利用率应该较高

5. **报告生成和服务器关闭**
   - 自动生成评估报告
   - 自动关闭服务器

## 常用命令总结

```bash
# 实时查看日志
tail -f training_*.log

# 查看Loss
tail -100 training_*.log | grep loss

# 查看GPU
nvidia-smi

# 查看进程
ps aux | grep train_qwen2_5_7b_qlora_demo.py

# 停止训练（如果需要）
pkill -f train_qwen2_5_7b_qlora_demo.py
```

## 预期训练时间

- **数据准备**: 1-2分钟（多进程优化后）
- **训练前评估**: 2-5分钟（批量推理优化后）
- **训练**: 8-10分钟（382步）
- **训练后评估**: 2-5分钟（批量推理优化后）
- **总计**: 约15-25分钟

## 注意事项

1. **不要关闭终端**：训练在后台运行，关闭终端不会影响训练
2. **日志文件会自动生成**：文件名格式为 `training_YYYYMMDD_HHMMSS.log`
3. **训练完成后服务器会自动关闭**：所有结果已保存后才关闭
4. **如果训练中断**：检查日志文件中的错误信息
