#!/usr/bin/env python3
"""
测试数据使用示例

展示如何使用generate_test_data.py生成的数据进行评估测试

包含：
1. 加载测试数据
2. 计算相似度矩阵（简化版）
3. 评估TPIR@FPIR指标
4. 可视化结果
"""

import pickle
import numpy as np
import time
from pathlib import Path


def load_test_data(file_path):
    """加载测试数据"""
    print(f"加载测试数据: {file_path}")
    start_time = time.time()

    with open(file_path, 'rb') as f:
        features, _, labels, paths = pickle.load(f)

    load_time = time.time() - start_time
    print(f"  加载完成，耗时: {load_time:.2f}秒")
    print(f"  样本数: {len(labels):,}")
    print(f"  特征维度: {features.shape[1]}")
    print(f"  身份数: {len(np.unique(labels)):,}")

    return features, labels, paths


def compute_similarity_histogram(features, labels, bins=10000):
    """
    计算相似度直方图（简化版，仅用于演示）

    注意：这是简化实现，实际评估请使用各模型文件夹中的完整实现
    """
    print(f"\n计算相似度直方图 (bins={bins:,})...")
    start_time = time.time()

    n_samples = len(labels)

    # 初始化直方图
    # bins范围：相似度从-1到1
    hist_pos = np.zeros(bins, dtype=np.int64)
    hist_neg = np.zeros(bins, dtype=np.int64)

    # 分块计算（避免内存溢出）
    chunk_size = 1000
    n_chunks = (n_samples + chunk_size - 1) // chunk_size

    print(f"  分 {n_chunks} 块处理...")

    for i in range(0, n_samples, chunk_size):
        end_i = min(i + chunk_size, n_samples)
        feat_i = features[i:end_i]
        label_i = labels[i:end_i]

        for j in range(i + 1, n_samples, chunk_size):
            end_j = min(j + chunk_size, n_samples)
            feat_j = features[j:end_j]
            label_j = labels[j:end_j]

            # 计算相似度
            sims = feat_i @ feat_j.T  # (chunk_i, chunk_j)

            # 确定正负样本
            is_pos = label_i[:, np.newaxis] == label_j[np.newaxis, :]

            # 转换为bin索引
            bin_indices = ((sims + 1.0) / 2.0 * (bins - 1)).astype(np.int32)
            bin_indices = np.clip(bin_indices, 0, bins - 1)

            # 累加直方图
            pos_bins = bin_indices[is_pos]
            neg_bins = bin_indices[~is_pos]

            hist_pos += np.bincount(pos_bins, minlength=bins)
            hist_neg += np.bincount(neg_bins, minlength=bins)

        if (i // chunk_size + 1) % max(1, n_chunks // 10) == 0:
            progress = (i + chunk_size) / n_samples * 100
            print(f"  进度: {progress:.1f}%")

    compute_time = time.time() - start_time
    print(f"  计算完成，耗时: {compute_time:.2f}秒")

    # 统计
    total_pos = hist_pos.sum()
    total_neg = hist_neg.sum()
    total_pairs = total_pos + total_neg

    print(f"\n统计结果:")
    print(f"  总样本对数: {total_pairs:,}")
    print(f"  正样本对数: {total_pos:,} ({total_pos/total_pairs*100:.3f}%)")
    print(f"  负样本对数: {total_neg:,} ({total_neg/total_pairs*100:.3f}%)")

    return hist_pos, hist_neg


def calculate_tpir_at_fpir(hist_pos, hist_neg, fpir_targets=[1e-5, 1e-4, 1e-3, 1e-2]):
    """
    计算TPIR@FPIR指标

    TPIR (True Positive Identification Rate): 正样本识别率
    FPIR (False Positive Identification Rate): 负样本误识率
    """
    print(f"\n计算TPIR@FPIR指标...")

    total_pos = hist_pos.sum()
    total_neg = hist_neg.sum()

    # 从高相似度到低相似度累积
    cum_pos = np.cumsum(hist_pos[::-1])[::-1]
    cum_neg = np.cumsum(hist_neg[::-1])[::-1]

    results = {}

    print(f"\n评估结果:")
    print(f"{'FPIR':<15} {'TPIR':<15} {'阈值Bin':<15}")
    print(f"{'-'*45}")

    for fpir_target in fpir_targets:
        # 找到对应的阈值
        fpir_values = cum_neg / total_neg

        # 找到第一个FPIR <= target的位置
        valid_indices = np.where(fpir_values <= fpir_target)[0]

        if len(valid_indices) > 0:
            threshold_idx = valid_indices[0]
            tpir = cum_pos[threshold_idx] / total_pos
            actual_fpir = fpir_values[threshold_idx]

            results[fpir_target] = {
                'tpir': tpir,
                'actual_fpir': actual_fpir,
                'threshold_idx': threshold_idx
            }

            print(f"{fpir_target:<15.0e} {tpir:<15.4f} {threshold_idx:<15}")
        else:
            print(f"{fpir_target:<15.0e} {'N/A':<15} {'N/A':<15}")
            results[fpir_target] = None

    return results


def main():
    """主函数：完整的使用示例"""

    print("="*80)
    print("测试数据使用示例")
    print("="*80)

    # 1. 生成测试数据（如果不存在）
    test_file = Path("example_test_data.pkl")

    if not test_file.exists():
        print(f"\n测试数据不存在，正在生成...")
        print(f"生成配置: 500个身份，每个5个样本（共2,500样本）")

        # 导入生成函数
        import sys
        sys.path.insert(0, str(Path(__file__).parent))

        try:
            from generate_test_data import generate_test_dataset

            features, labels, paths = generate_test_dataset(
                num_identities=500,
                samples_per_identity=5,
                feat_dim=512,
                noise_std=0.15,
                seed=42,
                output_path=str(test_file),
                verbose=True
            )
        except ImportError:
            print("错误: 无法导入generate_test_data模块")
            print("请确保generate_test_data.py在同一目录下")
            return
    else:
        print(f"\n使用已存在的测试数据: {test_file}")

    # 2. 加载数据
    features, labels, paths = load_test_data(test_file)

    # 3. 计算相似度直方图
    hist_pos, hist_neg = compute_similarity_histogram(
        features, labels, bins=10000
    )

    # 4. 计算TPIR@FPIR
    results = calculate_tpir_at_fpir(
        hist_pos, hist_neg,
        fpir_targets=[1e-5, 1e-4, 1e-3, 1e-2]
    )

    # 5. 总结
    print(f"\n{'='*80}")
    print("示例完成")
    print(f"{'='*80}")
    print(f"\n本示例展示了如何:")
    print(f"  1. 生成测试数据（使用generate_test_data.py）")
    print(f"  2. 加载测试数据")
    print(f"  3. 计算相似度直方图（简化版）")
    print(f"  4. 评估TPIR@FPIR指标")
    print(f"\n注意:")
    print(f"  - 这是简化实现，用于演示流程")
    print(f"  - 实际评估请使用各模型文件夹中的完整实现")
    print(f"  - 完整实现包含多GPU并行、TF32优化等高性能技术")
    print(f"\n推荐查看:")
    print(f"  - codex-sol-2/face_similarity_evaluator.py (最优实现)")
    print(f"  - deepseek/step2_single_gpu.py (清晰易读)")
    print(f"  - grok/eval_similarity.py (线程池实现)")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
