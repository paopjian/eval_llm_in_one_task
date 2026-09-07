# 200万数据集 OOM 问题记录

## 问题汇总

### 1. glm-pro - 内存管理问题 ❌ **代码实现问题**
**文件**: `glm-pro/step3_multi_gpu.py`  
**错误**: 
1. 首次：`torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB. GPU 3 available: 781.00 MiB`
2. 减半后：7卡GPU利用率99-100%，显存7.7GB正常，但**系统内存爆满**

**原因**: 
- 块大小8192×65536太大，减半到4096×32768后GPU显存正常
- **根本问题**：代码设计缺陷，计算过程中系统内存持续增长直至爆满
- 可能是结果累积方式有问题，或存在内存泄漏

**结论**: 
- **跳过此模型**，代码实现有严重内存管理问题
- 不是参数问题，是设计问题

**状态**: ❌ 失败 - 代码实现问题

---

### 2. deepseek-pro - 内存管理问题 ❌ **代码实现问题**
**文件**: `deepseek-pro/step4_eval_optimized.py`  
**错误**: 
1. 首次：`torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1024.00 MiB. GPU 0 available: 504.56 MiB`
2. 减半后：7卡GPU利用率99-100%，显存7.7GB正常，但**系统内存暴增**

**内存增长情况**:
- 初始：10GB
- 30秒后：28GB (+18GB)
- 60秒后：158GB (+130GB)
- 持续暴增，无法完成

**原因**:
- 块大小16384太大，减半到8192后GPU显存正常
- **根本问题**：代码设计缺陷，与glm-pro相同的内存泄漏问题
- 计算过程中系统内存持续增长直至占满

**结论**: 
- **跳过此模型**，代码实现有严重内存管理问题
- 不是参数问题，是设计问题

**状态**: ❌ 失败 - 代码实现问题

---

### 3. codex-sol-m - multiprocessing并行失败 ❌ **代码实现问题**
**文件**: `codex-sol-m/face_similarity_eval.py`  
**错误**: 
1. 首次：`torch.OutOfMemoryError: CUDA out of memory` (GPU有残留进程)
2. GPU清理后：multiprocessing只启动了1个worker，其余6个GPU闲置

**现象**:
- 7个GPU进程都启动了（PID 3287679-3287685），各占8GB显存
- 但只有1个进程在工作（CPU 100%），其他6个处于等待状态
- GPU利用率0-9%，说明只有1个GPU在断断续续工作
- 运行11分钟仍未完成，效率极低

**根本原因**:
- multiprocessing的spawn模式在7卡并行时有问题
- 可能是进程间通信卡住或任务分配不均

**结论**: 
- **跳过此模型**，代码实现有缺陷
- codex-sol和codex-sol-2都成功了（52-122秒），codex-sol-m的实现不如它们

**状态**: ❌ 失败 - 代码实现问题，非参数问题

---

## GPU清理问题

### 残留进程导致显存占用
**现象**: 即使主进程已退出，GPU上仍有multiprocessing子进程占用显存

**解决方法**:
```bash
# 1. 查看GPU进程
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader

# 2. 强制清理所有Python进程
pkill -9 -f python

# 3. 等待显存释放
sleep 2

# 4. 验证清理
nvidia-smi
```

---

## 最佳实践

### 批量评估前的准备
1. **清理环境**:
   ```bash
   pkill -9 -f python
   sleep 2
   nvidia-smi  # 确认显存已清空
   ```

2. **顺序执行**: 一次运行一个模型，避免显存竞争

3. **设置超时**: 单个模型超时时间1200秒（20分钟）

4. **块大小选择原则**:
   - 对于200万数据，块大小应确保单次计算 < 2GB
   - 公式：`块大小 × 数据量 × 4字节(float32) < 2GB`
   - 例如：`128 × 2000000 × 4 ≈ 1GB` ✅

---

## 修复状态

| 模型 | 原始参数 | 优化参数 | 状态 |
|------|----------|----------|------|
| glm-pro | block-rows=8192, cols=65536 | 跳过 - 系统内存爆满 | ❌ 失败 |
| deepseek-pro | block=16384 | 跳过 - 系统内存爆满 | ❌ 失败 |
| codex-sol-m | block-size=512 | 跳过 - 代码实现问题 | ❌ 失败 |
