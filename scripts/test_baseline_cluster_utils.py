#!/usr/bin/env python3
"""
基于cluster_utils的全模型基准测试
使用统一的cluster_utils方法测试所有模型生成的代码
"""

import os
import sys
import time
import pickle
import json
from datetime import datetime
from pathlib import Path
import importlib.util

# 确保在cvlface环境
print("正在检查环境...")
try:
    import torch
    import numpy as np
    print(f"✅ PyTorch {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ CUDA devices: {torch.cuda.device_count()}")
except ImportError as e:
    print(f"❌ 环境错误: {e}")
    print("请使用: source /root/miniconda3/bin/activate cvlface")
    sys.exit(1)

def load_cluster_utils():
    """动态加载cluster_utils.py作为基准参考"""
    # 使用相对路径，假设cluster_utils.py在上层目录
    script_dir = Path(__file__).parent
    cluster_utils_path = script_dir.parent / 'cluster_utils.py'

    if not cluster_utils_path.exists():
        print(f"❌ 找不到cluster_utils.py: {cluster_utils_path}")
        return None

    spec = importlib.util.spec_from_file_location("cluster_utils", str(cluster_utils_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_with_cluster_utils(data_file, method_name='基准方法'):
    """使用cluster_utils基准方法测试"""
    print(f"\n{'='*70}")
    print(f"测试: {method_name}")
    print(f"数据: {data_file}")
    print(f"{'='*70}")

    # 加载cluster_utils
    cluster_utils = load_cluster_utils()
    if cluster_utils is None:
        return None

    # 加载数据
    print("加载数据...")
    t0 = time.time()
    with open(data_file, 'rb') as f:
        features, _, labels, _ = pickle.load(f)

    # 确保是float32和连续内存
    features = np.ascontiguousarray(features, dtype=np.float32)
    labels = np.ascontiguousarray(labels, dtype=np.int64)

    # L2归一化
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / norms

    load_time = time.time() - t0
    n_samples = len(labels)
    n_pairs = n_samples * (n_samples - 1) // 2

    print(f"✅ 数据加载完成")
    print(f"   样本数: {n_samples:,}")
    print(f"   样本对: {n_pairs/1e9:.2f}B")
    print(f"   加载耗时: {load_time:.2f}秒")

    # 运行评估
    print("\n开始计算相似度矩阵...")
    t0 = time.time()

    try:
        # 调用cluster_utils的大规模评估函数v4
        # 这是最新最优化的版本，支持动态负载均衡
        pos_hist, neg_hist = cluster_utils.get_sim_matrix_large_scale_v4(
            query_feats_list=features,
            query_ids=labels,
            num_gpus=7,
            block_size=2048*5,
            hist_bins=20_000_000,
            hist_range=(-1.0, 1.0),
            collect_pairs_config=None,
            memory_mode='low_memory',
            show_progress=True
        )

        compute_time = time.time() - t0

        print(f"✅ 计算完成")
        print(f"⏱️  总耗时: {compute_time:.2f}秒")
        print(f"📊 吞吐率: {n_pairs/compute_time/1e9:.2f}B对/秒")
        print(f"📈 正样本对数: {pos_hist.sum():,}")
        print(f"📉 负样本对数: {neg_hist.sum():,}")

        return {
            'status': 'success',
            'total_time': compute_time,
            'load_time': load_time,
            'samples': n_samples,
            'pairs': n_pairs,
            'throughput': n_pairs/compute_time/1e9,
            'pos_pairs': int(pos_hist.sum()),
            'neg_pairs': int(neg_hist.sum())
        }

    except Exception as e:
        import traceback
        compute_time = time.time() - t0
        print(f"❌ 计算失败: {str(e)}")
        print(f"详细错误:")
        traceback.print_exc()
        return {
            'status': 'failed',
            'error': str(e),
            'time': compute_time
        }

def main():
    print("="*70)
    print("基于cluster_utils的基准测试")
    print("="*70)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 测试数据集
    datasets = [
        ('test_data_1min.pkl', '1分钟级', 75000),
        ('test_data_10min.pkl', '10分钟级', 200000)
    ]

    results = {
        'test_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'method': 'cluster_utils基准方法',
        'results': {}
    }

    for data_file, desc, expected_samples in datasets:
        if not os.path.exists(data_file):
            print(f"⚠️  跳过 {desc}: 文件不存在 {data_file}\n")
            continue

        result = test_with_cluster_utils(data_file, f'cluster_utils基准 ({desc})')

        if result:
            results['results'][desc] = result

        print()

    # 保存结果
    output_file = 'cluster_utils_benchmark_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("="*70)
    print("测试完成！")
    print("="*70)
    print(f"\n结果已保存到: {output_file}\n")

    # 打印摘要
    print("结果摘要:")
    print(f"{'数据集':<15} {'状态':<10} {'总耗时':<15} {'吞吐率':<20}")
    print("-"*60)

    for desc, result in results['results'].items():
        if result['status'] == 'success':
            status_icon = '✅'
            time_str = f"{result['total_time']:.2f}秒"
            throughput_str = f"{result['throughput']:.2f}B对/秒"
        else:
            status_icon = '❌'
            time_str = '-'
            throughput_str = '-'

        print(f"{desc:<15} {status_icon} {result['status']:<8} {time_str:<15} {throughput_str:<20}")

    print()

if __name__ == '__main__':
    main()
