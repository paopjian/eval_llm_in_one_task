# Scripts 目录说明

本目录包含所有测试、评估和监控脚本。

## 📂 目录结构

### 批量测试脚本
- `batch_test_200w.sh` - 200万数据集批量测试（基础版）
- `batch_test_200w_optimized.sh` - 200万数据集批量测试（优化版，含显存管理）
- `run_all_models_200w.sh` - 运行所有模型的200万数据集测试
- `run_all_models_benchmark.sh` - 全模型性能基准测试

### 评估脚本
- `run_200w_evaluation.sh` - 200万数据集评估（Shell版本）
- `run_200w_evaluation.py` - 200万数据集评估（Python版本）
- `run_200w_evaluation_v2.py` - 200万数据集评估v2版本

### 测试工具
- `test_all_models.py` - 测试所有模型实现
- `test_baseline_cluster_utils.py` - 基准cluster_utils测试
- `check_all_models.py` - 检查所有模型状态

### 可视化工具
- `generate_benchmark_charts.py` - 生成性能基准图表

### 监控脚本
- `memory_monitor.sh` - 内存使用监控
- `monitor_evaluation.sh` - 评估进程监控
- `monitor_v2.sh` - 监控脚本v2版本

## 🚀 使用示例

### 运行完整基准测试
```bash
cd scripts
bash run_all_models_benchmark.sh
```

### 运行200万数据集测试
```bash
cd scripts
bash batch_test_200w_optimized.sh
```

### 生成性能图表
```bash
cd scripts
python generate_benchmark_charts.py
```

### 监控内存使用
```bash
cd scripts
bash memory_monitor.sh
```

## ⚙️ 环境要求

所有脚本默认使用 `cvlface` conda环境，请根据实际情况修改：
```bash
conda activate cvlface  # 或: source <您的conda路径>/bin/activate cvlface
```

## 📝 注意事项

1. 运行前确保已生成测试数据（使用根目录的 `generate_test_data.py`）
2. 批量测试脚本会自动清理GPU显存
3. 监控脚本需要在后台运行
4. 结果日志会保存到 `../logs/` 目录
