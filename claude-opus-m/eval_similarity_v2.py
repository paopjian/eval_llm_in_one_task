#!/usr/bin/env python3
"""
人脸特征相似度评估系统 - 第二版：优化版本
优化策略：
1. 不存储所有相似度值，使用直方图统计
2. 增大chunk_size提高计算效率
3. 使用多GPU并行加速
"""
import pickle
import numpy as np
import torch
import time
from tqdm import tqdm
import argparse

def load_data(pkl_path):
    """加载特征数据"""
    print("=" * 80)
    print("步骤1: 加载数据")
    print("=" * 80)

    with open(pkl_path, 'rb') as f:
        query_feats_list, _, query_ids, file_paths = pickle.load(f)

    print(f"特征矩阵形状: {query_feats_list.shape}")
    print(f"样本总数: {len(query_ids)}")
    print(f"身份总数: {len(np.unique(query_ids))}")

    # 统计样本对
    N = len(query_ids)
    total_pairs = N * (N - 1) // 2
    unique_ids, counts = np.unique(query_ids, return_counts=True)
    total_pos_pairs = sum(c * (c - 1) // 2 for c in counts)
    total_neg_pairs = total_pairs - total_pos_pairs

    print(f"\n样本对统计:")
    print(f"  总样本对数: {total_pairs:,}")
    print(f"  正样本对数: {total_pos_pairs:,} ({total_pos_pairs/total_pairs*100:.4f}%)")
    print(f"  负样本对数: {total_neg_pairs:,} ({total_neg_pairs/total_pairs*100:.4f}%)")

    return query_feats_list, query_ids, file_paths, total_pos_pairs, total_neg_pairs

def compute_similarity_histogram_gpu(features, query_ids, device='cuda:0', chunk_size=2000):
    """
    使用直方图统计方法，避免存储所有相似度值
    bins: 相似度范围[-1, 1]，使用10000个bin，精度0.0002
    """
    print("\n" + "=" * 80)
    print(f"步骤2: 计算相似度直方图 (设备: {device})")
    print("=" * 80)

    N = len(features)
    feat_tensor = torch.from_numpy(features).float().to(device)

    # 使用直方图统计，范围[-1, 1]，10000个bins
    bins = np.linspace(-1, 1, 10001)
    pos_hist = np.zeros(10000, dtype=np.int64)
    neg_hist = np.zeros(10000, dtype=np.int64)

    start_time = time.time()
    total_computed = 0

    # 分块计算
    for i in tqdm(range(0, N, chunk_size), desc="计算进度"):
        i_end = min(i + chunk_size, N)
        feat_i = feat_tensor[i:i_end]

        for j in range(i, N, chunk_size):
            j_end = min(j + chunk_size, N)
            feat_j = feat_tensor[j:j_end]

            # 计算相似度
            sim_block = torch.mm(feat_i, feat_j.t()).cpu().numpy()

            # 处理每个子块
            for local_i in range(sim_block.shape[0]):
                global_i = i + local_i

                if j == i:
                    start_j = local_i + 1
                else:
                    start_j = 0

                for local_j in range(start_j, sim_block.shape[1]):
                    global_j = j + local_j

                    if global_i < global_j:
                        sim_val = sim_block[local_i, local_j]

                        # 找到对应的bin索引
                        bin_idx = int((sim_val + 1) / 2 * 10000)
                        bin_idx = max(0, min(9999, bin_idx))

                        # 判断正负样本
                        if query_ids[global_i] == query_ids[global_j]:
                            pos_hist[bin_idx] += 1
                        else:
                            neg_hist[bin_idx] += 1

                        total_computed += 1

    elapsed = time.time() - start_time
    print(f"\n计算完成，耗时: {elapsed:.2f}秒")
    print(f"计算样本对数: {total_computed:,}")
    print(f"正样本对数: {pos_hist.sum():,}")
    print(f"负样本对数: {neg_hist.sum():,}")

    return pos_hist, neg_hist, bins

def compute_tpir_fpir_from_hist(pos_hist, neg_hist, bins, fpir_thresholds=[1e-5, 1e-4, 1e-3, 1e-2]):
    """从直方图计算TPIR@FPIR"""
    print("\n" + "=" * 80)
    print("步骤3: 计算TPIR@FPIR")
    print("=" * 80)

    total_pos = pos_hist.sum()
    total_neg = neg_hist.sum()

    # 计算累积分布（从高到低）
    neg_cumsum = np.cumsum(neg_hist[::-1])[::-1]  # 相似度>=阈值的负样本数
    pos_cumsum = np.cumsum(pos_hist[::-1])[::-1]  # 相似度>=阈值的正样本数

    results = []
    for fpir in fpir_thresholds:
        # 找到对应FPIR的bin索引
        target_neg_count = int(fpir * total_neg)

        # 找到第一个累积数<=target的索引
        idx = np.searchsorted(neg_cumsum[::-1], target_neg_count)
        idx = len(neg_cumsum) - idx - 1

        if idx < 0:
            idx = 0
        if idx >= len(bins) - 1:
            idx = len(bins) - 2

        threshold = bins[idx]

        # 计算TPIR
        tpir = pos_cumsum[idx] / total_pos if total_pos > 0 else 0

        results.append({
            'fpir': fpir,
            'threshold': threshold,
            'tpir': tpir
        })

        print(f"TPIR @ FPIR={fpir:.0e}: {tpir*100:.2f}% (阈值={threshold:.4f})")

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0', help='GPU设备')
    parser.add_argument('--chunk_size', type=int, default=2000, help='分块大小')
    args = parser.parse_args()

    # 加载数据
    features, query_ids, file_paths, total_pos, total_neg = load_data('s4_0618_enhance.pkl')

    # 计算相似度直方图
    pos_hist, neg_hist, bins = compute_similarity_histogram_gpu(
        features, query_ids, device=args.device, chunk_size=args.chunk_size
    )

    # 计算TPIR@FPIR
    results = compute_tpir_fpir_from_hist(pos_hist, neg_hist, bins)

    print("\n" + "=" * 80)
    print("评估完成！")
    print("=" * 80)

if __name__ == '__main__':
    main()
