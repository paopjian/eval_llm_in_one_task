#!/usr/bin/env python3
"""
版本1: 基础实现 - 单卡分块计算相似度和TPIR@FPIR
"""
import pickle
import numpy as np
import torch
import time
from collections import defaultdict

def load_data(data_path='s4_0618_enhance.pkl'):
    """加载特征数据"""
    print("正在加载数据...")
    with open(data_path, 'rb') as f:
        query_feats_list, _, query_ids, file_paths = pickle.load(f)
    print(f"数据加载完成: {len(query_ids)} 个样本")
    return query_feats_list, query_ids, file_paths

def compute_similarity_chunked(feats, ids, chunk_size=5000, device='cuda:0'):
    """
    分块计算相似度矩阵，只计算上三角
    收集正负样本的相似度值用于TPIR@FPIR计算
    """
    N = len(feats)
    feats_tensor = torch.from_numpy(feats).to(device)
    ids_tensor = torch.from_numpy(ids).to(device)

    # 存储正负样本相似度（先用列表，后面优化）
    pos_similarities = []
    neg_similarities = []

    total_pairs = N * (N - 1) // 2
    processed_pairs = 0

    print(f"\n开始计算相似度矩阵 (分块大小: {chunk_size})")
    print(f"总样本对数: {total_pairs:,}")
    start_time = time.time()

    for i in range(0, N, chunk_size):
        i_end = min(i + chunk_size, N)
        chunk_i = feats_tensor[i:i_end]
        ids_i = ids_tensor[i:i_end]

        # 只计算 j > i 的部分（上三角）
        for j in range(i, N, chunk_size):
            j_end = min(j + chunk_size, N)
            chunk_j = feats_tensor[j:j_end]
            ids_j = ids_tensor[j:j_end]

            # 计算相似度: 特征已归一化，直接矩阵乘法
            sim_matrix = torch.matmul(chunk_i, chunk_j.T)  # shape: (i_size, j_size)

            # 计算身份匹配矩阵
            ids_match = ids_i.unsqueeze(1) == ids_j.unsqueeze(0)  # shape: (i_size, j_size)

            # 处理上三角部分
            if i == j:
                # 对角块：只取严格上三角
                mask = torch.triu(torch.ones_like(sim_matrix, dtype=torch.bool), diagonal=1)
                sim_values = sim_matrix[mask]
                match_values = ids_match[mask]
            else:
                # 非对角块：全部使用
                sim_values = sim_matrix.flatten()
                match_values = ids_match.flatten()

            # 分离正负样本
            pos_mask = match_values
            neg_mask = ~match_values

            if pos_mask.any():
                pos_similarities.append(sim_values[pos_mask].cpu().numpy())
            if neg_mask.any():
                neg_similarities.append(sim_values[neg_mask].cpu().numpy())

            # 更新进度
            if i == j:
                processed_pairs += mask.sum().item()
            else:
                processed_pairs += sim_values.numel()

            if processed_pairs % 10_000_000 == 0 or processed_pairs > total_pairs * 0.9:
                elapsed = time.time() - start_time
                progress = processed_pairs / total_pairs * 100
                speed = processed_pairs / elapsed
                eta = (total_pairs - processed_pairs) / speed
                print(f"  进度: {progress:.2f}% | "
                      f"已处理: {processed_pairs:,} 对 | "
                      f"速度: {speed:,.0f} 对/秒 | "
                      f"预计剩余: {eta:.1f}秒")

    elapsed = time.time() - start_time
    print(f"\n相似度计算完成！")
    print(f"  总耗时: {elapsed:.2f} 秒")
    print(f"  平均速度: {total_pairs/elapsed:,.0f} 对/秒")

    # 合并所有相似度值
    pos_similarities = np.concatenate(pos_similarities)
    neg_similarities = np.concatenate(neg_similarities)

    return pos_similarities, neg_similarities

def compute_tpir_at_fpir(pos_sims, neg_sims, fpir_targets=[1e-5, 1e-4, 1e-3, 1e-2]):
    """
    计算TPIR@FPIR指标

    FPIR(threshold) = (负样本中 sim > threshold 的数量) / (总负样本数)
    TPIR(threshold) = (正样本中 sim > threshold 的数量) / (总正样本数)
    """
    print(f"\n开始计算TPIR@FPIR指标...")
    print(f"  正样本数: {len(pos_sims):,}")
    print(f"  负样本数: {len(neg_sims):,}")

    # 对负样本相似度排序（降序）
    print("  正在排序负样本相似度...")
    neg_sims_sorted = np.sort(neg_sims)[::-1]

    results = {}
    print("\n" + "="*60)
    print("TPIR @ FPIR 评估结果")
    print("="*60)

    for fpir in fpir_targets:
        # 找到对应FPIR的阈值
        neg_count = int(fpir * len(neg_sims))
        if neg_count >= len(neg_sims):
            threshold = neg_sims_sorted[-1]
        else:
            threshold = neg_sims_sorted[neg_count]

        # 计算该阈值下的TPIR
        tpir = (pos_sims > threshold).sum() / len(pos_sims)

        results[fpir] = {'threshold': threshold, 'tpir': tpir}
        print(f"TPIR @ FPIR={fpir:.0e}: {tpir*100:.2f}% (阈值={threshold:.4f})")

    print("="*60)

    return results

def main():
    print("="*80)
    print("人脸特征相似度评估系统 - 版本1: 基础实现")
    print("="*80)

    # 1. 加载数据
    feats, ids, paths = load_data()

    # 2. 计算相似度
    pos_sims, neg_sims = compute_similarity_chunked(feats, ids, chunk_size=5000)

    print(f"\n相似度统计:")
    print(f"  正样本相似度 - 均值: {pos_sims.mean():.4f}, 标准差: {pos_sims.std():.4f}")
    print(f"  负样本相似度 - 均值: {neg_sims.mean():.4f}, 标准差: {neg_sims.std():.4f}")

    # 3. 计算TPIR@FPIR
    results = compute_tpir_at_fpir(pos_sims, neg_sims)

    print("\n任务完成！")

    return results

if __name__ == '__main__':
    main()
