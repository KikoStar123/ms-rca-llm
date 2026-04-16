# 性能优化需求

## 用户要求
- **服务器**: 24核心CPU，单核性能一般，但核心多
- **要求**: 充分利用多核，提升速度
- **付费用户**: 要求速度快

## 需要优化的CPU密集型任务

### 1. 数据Tokenization（最高优先级）
**位置**: `train_qwen2_5_7b_qlora_demo.py` line ~436
```python
# 当前代码（单线程）
train_dataset = Dataset.from_list([build_train_item(s, tokenizer) for s in train_samples])

# 优化方案（多进程）
from concurrent.futures import ProcessPoolExecutor
with ProcessPoolExecutor(max_workers=24) as executor:
    train_items = list(executor.map(build_train_item_wrapper, train_samples))
train_dataset = Dataset.from_list(train_items)
```

**预期加速**: 24倍（1528条数据，24核心并行）

### 2. 训练前评估（evaluate_model）
**位置**: `train_qwen2_5_7b_qlora_demo.py` line ~434
```python
# 当前代码（逐个推理）
before_metrics = evaluate_model(model, tokenizer, eval_samples, before_path)

# 优化方案
# - 批量推理（batch inference）
# - 多进程预处理prompt
# - 并行后处理结果
```

**预期加速**: 2-5倍（200个样本）

### 3. 数据加载（load_samples）
**位置**: `train_qwen2_5_7b_qlora_demo.py` line ~270
```python
# 当前代码（单线程读取）
samples = load_samples(DATA_PATH)

# 优化方案（多进程读取大文件）
# 如果文件很大，可以分块并行读取
```

**预期加速**: 2-3倍

## 实现注意事项

1. **Tokenization多进程**:
   - `tokenizer`对象需要序列化，可能需要特殊处理
   - 或者使用`multiprocessing.Pool` + `initializer`传递tokenizer
   - 或者使用`ThreadPoolExecutor`（如果tokenizer是线程安全的）

2. **评估多进程**:
   - 模型推理主要在GPU，CPU部分主要是预处理和后处理
   - 可以批量推理提升GPU利用率
   - 预处理可以用多进程

3. **Workers数量**:
   - 默认: 24个（服务器核心数）
   - 可配置: 通过环境变量或参数控制

## 下次训练时的优化

- [x] 修改`build_train_item`调用为多进程版本 ✅
- [x] 优化`evaluate_model`为批量+多进程 ✅
- [x] 添加workers参数配置 ✅（NUM_WORKERS，默认24核心）
- [ ] 测试多进程tokenization性能提升（待下次训练验证）
- [x] 确保tokenizer在多进程环境下正常工作 ✅（每个进程重新加载tokenizer）

## 服务器自动关闭功能（已实现）

**用户要求**: AutoDL GPU服务器按时间收费，训练完成后自动关闭以节省费用

**实现位置**: `train_qwen2_5_7b_qlora_demo.py` line ~565-580

**关闭时机**:
1. ✅ 训练完成
2. ✅ 训练后评估完成
3. ✅ 评估报告生成完成
4. ✅ 所有结果保存完成
5. ✅ 然后自动关闭服务器

**实现方式**:
```python
# 使用shutdown命令关闭服务器（AutoDL支持）
subprocess.run(["shutdown", "-h", "now"], check=False)
# 如果失败，尝试poweroff
subprocess.run(["poweroff"], check=False)
```

**注意事项**:
- 所有结果文件已保存后才关闭
- 使用`sys.stdout.flush()`确保输出被写入
- 如果关闭命令失败，会打印警告信息

## 预期效果

- **数据准备时间**: 从10-15分钟降到1-2分钟（10倍加速）
- **评估时间**: 从10分钟降到2-5分钟（2-5倍加速）
- **总体训练准备时间**: 从20-30分钟降到5-10分钟
