#!/usr/bin/env python3
"""
第一步：读取特征文件并分析数据结构
- 特征矩阵形状、维度、样本数、身份数
- 正负样本对数量统计
- 特征L2归一化检查
"""
import pickle
import time

import numpy as np

PKL_PATH = 's4_0618_enhance.pkl'


def load_and_analyze_data():
    print("=" * 80)
    print("第一步：数据加载与分析")
    print("=" * 80)

    t0 = time.time()
    with open(PKL_PATH, 'rb') as f:
        query_feats_list, query_feats_list_flip, query_ids, file_paths = pickle.load(f)
    print(f"\n加载耗时: {time.time() - t0:.2f}s (文件大小约 {440 / 1e3:.0f}MB → 用时含反序列化)")

    feats = np.asarray(query_feats_list)
    ids = np.asarray(query_ids)

    print(f"\n数据基本信息:")
    print(f"  特征矩阵形状: {feats.shape}, dtype: {feats.dtype}")
    print(f"  特征维度: {feats.shape[1]}")
    print(f"  样本总数: {len(ids)}")
    print(f"  身份总数: {len(np.unique(ids))}")
    print(f"  flip列表长度: {len(query_feats_list_flip)} (本任务不使用)")
    print(f"  文件路径数: {len(file_paths)}, 示例: {file_paths[0]}")

    # 样本对统计
    N = len(ids)
    total_pairs = N * (N - 1) // 2
    unique_ids, counts = np.unique(ids, return_counts=True)
    total_pos_pairs = int(sum(int(c) * (c - 1) // 2 for c in counts))
    total_neg_pairs = total_pairs - total_pos_pairs

    print(f"\n身份分布统计 (每身份图片数):")
    print(f"  最小: {counts.min()}, 最大: {counts.max()}, 均值: {counts.mean():.2f}, 中位数: {np.median(counts):.1f}")

    print(f"\n样本对统计:")
    print(f"  总样本对数: {total_pairs:,}")
    print(f"  正样本对数: {total_pos_pairs:,} ({total_pos_pairs / total_pairs * 100:.4f}%)")
    print(f"  负样本对数: {total_neg_pairs:,} ({total_neg_pairs / total_pairs * 100:.4f}%)")

    # 特征归一化检查
    norms = np.linalg.norm(feats[:10000].astype(np.float64), axis=1)
    print(f"\n特征向量检查 (前1万条):")
    print(f"  L2范数均值: {norms.mean():.6f}")
    print(f"  L2范数标准差: {norms.std():.6f}")
    print(f"  是否归一化: {'是' if abs(norms.mean() - 1.0) < 0.01 else '否'}")

    print("\n" + "=" * 80)
    print("数据分析完成！")
    print("=" * 80)
    return feats, ids, file_paths


if __name__ == '__main__':
    load_and_analyze_data()
