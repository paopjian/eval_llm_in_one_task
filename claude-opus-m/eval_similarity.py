#!/usr/bin/env python3
"""
人脸特征相似度评估系统 - 第一版：基础实现
功能：数据加载、单卡相似度计算、TPIR@FPIR评估
"""
import pickle
import numpy as np
import torch
import time
from tqdm import tqdm

def load_data(pkl_path):
    """加载特征数据"""
    print("=" * 80)
    print("步骤1: 加载数据")
    print("=" * 80)

    with open(pkl_path, 'rb') as f:
        query_feats_list, query_feats_list_flip, query_ids, file_paths = pickle.load(f)

    print(f"特征矩阵形状: {query_feats_list.shape}")
    print(f"样本总数: {len(query_ids)}")
    print(f"身份总数: {len(np.unique(query_ids))}")

    # 检查归一化
    norms = np.linalg.norm(query_feats_list, axis=1)
    print(f"L2范数均值: {norms.mean():.6f}, 标准差: {norms.std():.6f}")

    return query_feats_list, query_ids, file_paths

def analyze_pairs(query_ids):
    """统计正负样本对数量"""
    print("\n" + "=" * 80)
    print("步骤2: 统计样本对")
    print("=" * 80)

    N = len(query_ids)
    total_pairs = N * (N - 1) // 2

    # 计算正样本对数量
    unique_ids, counts = np.unique(query_ids, return_counts=True)
    total_pos_pairs = sum(c * (c - 1) // 2 for c in counts)
    total_neg_pairs = total_pairs - total_pos_pairs

    print(f"总样本对数: {total_pairs:,}")
    print(f"正样本对数: {total_pos_pairs:,} ({total_pos_pairs/total_pairs*100:.2f}%)")
    print(f"负样本对数: {total_neg_pairs:,} ({total_neg_pairs/total_pairs*100:.2f}%)")

    return total_pos_pairs, total_neg_pairs

def compute_similarity_matrix_gpu(features, query_ids, device='cuda:0', chunk_size=1000):
    """
    在GPU上分块计算相似度矩阵，只计算上三角
    收集正负样本的相似度值
    """
    print("\n" + "=" * 80)
    print(f"步骤3: 计算相似度矩阵 (设备: {device})")
    print("=" * 80)

    N = len(features)
    feat_tensor = torch.from_numpy(features).float().to(device)

    pos_sims = []
    neg_sims = []

    start_time = time.time()

    # 分块计算，只计算上三角
    for i in tqdm(range(0, N, chunk_size), desc="计算相似度"):
        i_end = min(i + chunk_size, N)
        feat_i = feat_tensor[i:i_end]  # shape: (chunk_i, 512)

        # 只计算 j > i 的部分
        for j in range(i, N, chunk_size):
            j_end = min(j + chunk_size, N)
            feat_j = feat_tensor[j:j_end]  # shape: (chunk_j, 512)

            # 计算相似度: feat_i @ feat_j.T
            sim_block = torch.mm(feat_i, feat_j.t()).cpu().numpy()

            # 提取上三角部分
            for local_i in range(sim_block.shape[0]):
                global_i = i + local_i

                if j == i:
                    # 同一个块，只取上三角
                    start_j = local_i + 1
                else:
                    # 不同块，全部取
                    start_j = 0

                for local_j in range(start_j, sim_block.shape[1]):
                    global_j = j + local_j

                    if global_i < global_j:
                        sim_val = sim_block[local_i, local_j]

                        # 判断正负样本
                        if query_ids[global_i] == query_ids[global_j]:
                            pos_sims.append(sim_val)
                        else:
                            neg_sims.append(sim_val)

    elapsed = time.time() - start_time
    print(f"\n计算完成，耗时: {elapsed:.2f}秒")
    print(f"收集到正样本相似度: {len(pos_sims):,} 个")
    print(f"收集到负样本相似度: {len(neg_sims):,} 个")

    return np.array(pos_sims), np.array(neg_sims)

def compute_tpir_fpir(pos_sims, neg_sims, fpir_thresholds=[1e-5, 1e-4, 1e-3, 1e-2]):
    """计算TPIR@FPIR指标"""
    print("\n" + "=" * 80)
    print("步骤4: 计算TPIR@FPIR")
    print("=" * 80)

    # 对负样本相似度排序，找到对应FPIR的阈值
    neg_sims_sorted = np.sort(neg_sims)[::-1]  # 降序
    total_neg = len(neg_sims)
    total_pos = len(pos_sims)

    results = []
    for fpir in fpir_thresholds:
        # 找到对应FPIR的阈值
        num_false_accept = int(fpir * total_neg)
        if num_false_accept >= total_neg:
            threshold = neg_sims_sorted[-1] - 0.01
        else:
            threshold = neg_sims_sorted[num_false_accept]

        # 计算TPIR
        num_true_accept = np.sum(pos_sims >= threshold)
        tpir = num_true_accept / total_pos

        results.append({
            'fpir': fpir,
            'threshold': threshold,
            'tpir': tpir
        })

        print(f"TPIR @ FPIR={fpir:.0e}: {tpir*100:.2f}% (阈值={threshold:.4f})")

    return results

def main():
    # 加载数据
    features, query_ids, file_paths = load_data('s4_0618_enhance.pkl')

    # 统计样本对
    total_pos, total_neg = analyze_pairs(query_ids)

    # 计算相似度
    pos_sims, neg_sims = compute_similarity_matrix_gpu(
        features, query_ids, device='cuda:0', chunk_size=1000
    )

    # 计算TPIR@FPIR
    results = compute_tpir_fpir(pos_sims, neg_sims)

    print("\n" + "=" * 80)
    print("评估完成！")
    print("=" * 80)

if __name__ == '__main__':
    main()
