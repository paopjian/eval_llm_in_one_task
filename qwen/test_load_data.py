#!/usr/bin/env python3
"""
第一步：测试启动代码 —— 读取数据并统计基本信息
"""
import pickle
import time
import numpy as np


def load_and_analyze_data(pkl_path='s4_0618_enhance.pkl'):
    """读取并分析测试数据"""
    print("=" * 80)
    print("数据加载与分析")
    print("=" * 80)

    t0 = time.time()
    with open(pkl_path, 'rb') as f:
        query_feats_list, query_feats_list_flip, query_ids, file_paths = pickle.load(f)
    t_load = time.time() - t0
    print(f"\n加载耗时: {t_load:.2f} s")

    print(f"\n数据基本信息:")
    print(f"  特征矩阵形状: {query_feats_list.shape}, dtype: {query_feats_list.dtype}")
    print(f"  特征维度: {query_feats_list.shape[1]}")
    print(f"  样本总数: {len(query_ids)}")
    print(f"  身份总数: {len(np.unique(query_ids))}")
    print(f"  flip特征: {type(query_feats_list_flip)}, 长度 {len(query_feats_list_flip)}")
    print(f"  file_paths 示例: {file_paths[0]}")

    # 统计正负样本对数量
    N = len(query_ids)
    total_pairs = N * (N - 1) // 2

    unique_ids, counts = np.unique(query_ids, return_counts=True)
    total_pos_pairs = int(sum(c * (c - 1) // 2 for c in counts))
    total_neg_pairs = total_pairs - total_pos_pairs

    print(f"\n样本对统计:")
    print(f"  总样本对数: {total_pairs:,}")
    print(f"  正样本对数: {total_pos_pairs:,} ({total_pos_pairs / total_pairs * 100:.4f}%)")
    print(f"  负样本对数: {total_neg_pairs:,} ({total_neg_pairs / total_pairs * 100:.4f}%)")
    print(f"  每身份图片数: min={counts.min()}, max={counts.max()}, mean={counts.mean():.2f}")

    # 检查特征是否归一化
    norms = np.linalg.norm(query_feats_list, axis=1)
    print(f"\n特征向量检查:")
    print(f"  L2范数均值: {norms.mean():.6f}")
    print(f"  L2范数标准差: {norms.std():.6f}")
    print(f"  是否归一化: {'是' if np.abs(norms.mean() - 1.0) < 0.01 else '否'}")

    print("\n" + "=" * 80)
    print("数据分析完成！")
    print("=" * 80)

    return query_feats_list, query_ids, file_paths


if __name__ == '__main__':
    load_and_analyze_data()
