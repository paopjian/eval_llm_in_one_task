#!/usr/bin/env python3
"""
人脸特征相似度评估系统 - 最优版本
单GPU分块计算，使用numpy percentile快速计算TPIR@FPIR
"""
import pickle
import numpy as np
import torch
import time

def load_data(data_path='s4_0618_enhance.pkl'):
    """加载特征数据"""
    print("正在加载数据...")
    with open(data_path, 'rb') as f:
        query_feats_list, _, query_ids, file_paths = pickle.load(f)
    print(f"数据加载完成: {len(query_ids)} 个样本\n")
    return query_feats_list, query_ids, file_paths

def compute_similarity_single_gpu(feats, ids, chunk_size=5000, device='cuda:0'):
    """
    单GPU分块计算相似度矩阵
    返回正负样本相似度数组
    """
    N = len(feats)
    feats_tensor = torch.from_numpy(feats).to(device)
    ids_tensor = torch.from_numpy(ids).to(device)

    pos_similarities = []
    neg_similarities = []

    total_pairs = N * (N - 1) // 2

    print(f"开始计算相似度矩阵")
    print(f"  设备: {device}")
    print(f"  分块大小: {chunk_size}")
    print(f"  总样本对数: {total_pairs:,}\n")

    start_time = time.time()

    for i in range(0, N, chunk_size):
        i_end = min(i + chunk_size, N)
        chunk_i = feats_tensor[i:i_end]
        ids_i = ids_tensor[i:i_end]

        for j in range(i, N, chunk_size):
            j_end = min(j + chunk_size, N)
            chunk_j = feats_tensor[j:j_end]
            ids_j = ids_tensor[j:j_end]

            # 计算相似度矩阵
            sim_matrix = torch.matmul(chunk_i, chunk_j.T)
            ids_match = ids_i.unsqueeze(1) == ids_j.unsqueeze(0)

            # 只取上三角（避免重复计算）
            if i == j:
                mask = torch.triu(torch.ones_like(sim_matrix, dtype=torch.bool), diagonal=1)
                sim_values = sim_matrix[mask]
                match_values = ids_match[mask]
            else:
                sim_values = sim_matrix.flatten()
                match_values = ids_match.flatten()

            # 分离正负样本
            pos_mask = match_values
            neg_mask = ~match_values

            if pos_mask.any():
                pos_similarities.append(sim_values[pos_mask].cpu().numpy())
            if neg_mask.any():
                neg_similarities.append(sim_values[neg_mask].cpu().numpy())

    elapsed = time.time() - start_time
    print(f"相似度计算完成！")
    print(f"  总耗时: {elapsed:.2f} 秒")
    print(f"  平均速度: {total_pairs/elapsed:,.0f} 对/秒\n")

    # 合并数组
    pos_similarities = np.concatenate(pos_similarities)
    neg_similarities = np.concatenate(neg_similarities)

    return pos_similarities, neg_similarities

def compute_tpir_at_fpir(pos_sims, neg_sims, fpir_targets=[1e-5, 1e-4, 1e-3, 1e-2]):
    """
    使用numpy percentile快速计算TPIR@FPIR指标
    """
    print(f"开始计算TPIR@FPIR指标")
    print(f"  正样本数: {len(pos_sims):,}")
    print(f"  负样本数: {len(neg_sims):,}\n")

    results = {}

    print("="*70)
    print(" "*20 + "TPIR @ FPIR 评估结果")
    print("="*70)

    start_time = time.time()

    for fpir in fpir_targets:
        # 使用percentile找到对应FPIR的阈值
        # FPIR=1e-5表示前0.001%的负样本，即99.999分位数
        percentile = (1 - fpir) * 100
        threshold = np.percentile(neg_sims, percentile)

        # 计算该阈值下的TPIR
        tpir = (pos_sims > threshold).sum() / len(pos_sims)

        results[fpir] = {'threshold': threshold, 'tpir': tpir}
        print(f"TPIR @ FPIR={fpir:.0e}: {tpir*100:6.2f}%  (阈值={threshold:.4f})")

    print("="*70)

    elapsed = time.time() - start_time
    print(f"指标计算耗时: {elapsed:.2f} 秒\n")

    return results

def print_statistics(pos_sims, neg_sims):
    """打印统计信息"""
    print("="*70)
    print(" "*25 + "相似度统计")
    print("="*70)
    print(f"正样本:")
    print(f"  数量: {len(pos_sims):,}")
    print(f"  均值: {pos_sims.mean():.4f}")
    print(f"  标准差: {pos_sims.std():.4f}")
    print(f"  最小值: {pos_sims.min():.4f}")
    print(f"  最大值: {pos_sims.max():.4f}")
    print(f"\n负样本:")
    print(f"  数量: {len(neg_sims):,}")
    print(f"  均值: {neg_sims.mean():.4f}")
    print(f"  标准差: {neg_sims.std():.4f}")
    print(f"  最小值: {neg_sims.min():.4f}")
    print(f"  最大值: {neg_sims.max():.4f}")
    print("="*70 + "\n")

def main():
    total_start = time.time()

    print("="*70)
    print(" "*15 + "人脸特征相似度评估系统")
    print("="*70 + "\n")

    # 1. 加载数据
    feats, ids, paths = load_data()

    # 2. 计算相似度
    pos_sims, neg_sims = compute_similarity_single_gpu(feats, ids, chunk_size=5000)

    # 3. 打印统计信息
    print_statistics(pos_sims, neg_sims)

    # 4. 计算TPIR@FPIR
    results = compute_tpir_at_fpir(pos_sims, neg_sims)

    # 5. 总结
    total_time = time.time() - total_start
    print("="*70)
    print(f"任务全部完成！总耗时: {total_time:.2f} 秒 ({total_time/60:.1f} 分钟)")
    print("="*70)

    return results, pos_sims, neg_sims

if __name__ == '__main__':
    main()
