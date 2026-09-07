#!/usr/bin/env python3
"""
人脸特征相似度评估系统 - 第三版：高效多GPU并行版本
核心优化：
1. 按行分块计算相似度矩阵，充分利用GPU矩阵运算
2. 使用向量化操作批量处理正负样本判断
3. 支持多GPU并行加速
4. 使用直方图统计避免存储所有相似度值
"""
import pickle
import numpy as np
import torch
import torch.multiprocessing as mp
import time
from tqdm import tqdm
import argparse

def load_data(pkl_path):
    """加载特征数据"""
    print("=" * 80)
    print("加载数据")
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

    return query_feats_list, query_ids, file_paths

def process_row_range(rank, features, query_ids, row_start, row_end, chunk_size, device, return_dict):
    """
    处理指定行范围的相似度计算
    每个进程负责计算若干行与所有列的相似度
    """
    try:
        N = len(features)
        feat_tensor = torch.from_numpy(features).float().to(device)
        query_ids_tensor = torch.from_numpy(query_ids).long().to(device)

        # 直方图统计
        bins_count = 10000
        pos_hist = np.zeros(bins_count, dtype=np.int64)
        neg_hist = np.zeros(bins_count, dtype=np.int64)

        # 按行块处理
        for i in range(row_start, row_end, chunk_size):
            i_end = min(i + chunk_size, row_end)
            feat_i = feat_tensor[i:i_end]  # (chunk_i, 512)
            ids_i = query_ids_tensor[i:i_end]  # (chunk_i,)

            # 计算这批行与所有列的相似度
            # 只处理上三角部分 j > i
            j_start = i + 1

            for j in range(j_start, N, chunk_size):
                j_end = min(j + chunk_size, N)
                feat_j = feat_tensor[j:j_end]  # (chunk_j, 512)
                ids_j = query_ids_tensor[j:j_end]  # (chunk_j,)

                # 计算相似度矩阵块: (chunk_i, chunk_j)
                sim_block = torch.mm(feat_i, feat_j.t())

                # 判断正负样本: (chunk_i, chunk_j)
                # ids_i[:, None] shape: (chunk_i, 1)
                # ids_j[None, :] shape: (1, chunk_j)
                is_positive = (ids_i[:, None] == ids_j[None, :])

                # 转移到CPU处理
                sim_block_cpu = sim_block.cpu().numpy().flatten()
                is_positive_cpu = is_positive.cpu().numpy().flatten()

                # 计算bin索引
                bin_indices = ((sim_block_cpu + 1) / 2 * bins_count).astype(np.int32)
                bin_indices = np.clip(bin_indices, 0, bins_count - 1)

                # 分别统计正负样本
                pos_mask = is_positive_cpu
                neg_mask = ~is_positive_cpu

                # 累加到直方图
                np.add.at(pos_hist, bin_indices[pos_mask], 1)
                np.add.at(neg_hist, bin_indices[neg_mask], 1)

        return_dict[rank] = {
            'pos_hist': pos_hist,
            'neg_hist': neg_hist,
            'status': 'success'
        }

    except Exception as e:
        return_dict[rank] = {
            'status': 'error',
            'error': str(e)
        }

def compute_similarity_multigpu(features, query_ids, devices, chunk_size=1000):
    """
    多GPU并行计算相似度直方图
    """
    print("\n" + "=" * 80)
    print(f"计算相似度直方图 (使用 {len(devices)} 个GPU)")
    print("=" * 80)

    N = len(features)
    num_gpus = len(devices)

    # 计算每个GPU负责的行范围
    rows_per_gpu = N // num_gpus
    row_ranges = []
    for i in range(num_gpus):
        start = i * rows_per_gpu
        end = (i + 1) * rows_per_gpu if i < num_gpus - 1 else N
        row_ranges.append((start, end))
        print(f"GPU {devices[i]}: 行 {start} - {end} ({end - start} 行)")

    # 使用多进程并行
    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []

    start_time = time.time()

    for rank, (start, end) in enumerate(row_ranges):
        p = mp.Process(
            target=process_row_range,
            args=(rank, features, query_ids, start, end, chunk_size, devices[rank], return_dict)
        )
        p.start()
        processes.append(p)

    # 等待所有进程完成
    for p in processes:
        p.join()

    elapsed = time.time() - start_time

    # 合并结果
    pos_hist = np.zeros(10000, dtype=np.int64)
    neg_hist = np.zeros(10000, dtype=np.int64)

    for rank in range(num_gpus):
        if return_dict[rank]['status'] == 'success':
            pos_hist += return_dict[rank]['pos_hist']
            neg_hist += return_dict[rank]['neg_hist']
        else:
            print(f"GPU {rank} 错误: {return_dict[rank]['error']}")

    print(f"\n计算完成，耗时: {elapsed:.2f}秒")
    print(f"正样本对数: {pos_hist.sum():,}")
    print(f"负样本对数: {neg_hist.sum():,}")

    return pos_hist, neg_hist

def compute_tpir_fpir_from_hist(pos_hist, neg_hist, fpir_thresholds=[1e-5, 1e-4, 1e-3, 1e-2]):
    """从直方图计算TPIR@FPIR"""
    print("\n" + "=" * 80)
    print("计算TPIR@FPIR")
    print("=" * 80)

    total_pos = pos_hist.sum()
    total_neg = neg_hist.sum()

    bins = np.linspace(-1, 1, 10001)

    # 计算累积分布（从高到低）
    neg_cumsum = np.cumsum(neg_hist[::-1])[::-1]
    pos_cumsum = np.cumsum(pos_hist[::-1])[::-1]

    results = []
    for fpir in fpir_thresholds:
        target_neg_count = int(fpir * total_neg)
        idx = np.searchsorted(neg_cumsum[::-1], target_neg_count)
        idx = len(neg_cumsum) - idx - 1
        idx = max(0, min(idx, len(bins) - 2))

        threshold = bins[idx]
        tpir = pos_cumsum[idx] / total_pos if total_pos > 0 else 0

        results.append({
            'fpir': fpir,
            'threshold': threshold,
            'tpir': tpir
        })

        print(f"TPIR @ FPIR={fpir:.0e}: {tpir*100:.2f}% (阈值={threshold:.4f})")

    return results, pos_hist, neg_hist

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--devices', type=str, default='0,1,2,3,4,5,6', help='GPU设备列表，用逗号分隔')
    parser.add_argument('--chunk_size', type=int, default=1000, help='分块大小')
    args = parser.parse_args()

    devices = [f'cuda:{i}' for i in args.devices.split(',')]

    # 加载数据
    features, query_ids, file_paths = load_data('s4_0618_enhance.pkl')

    # 多GPU并行计算
    pos_hist, neg_hist = compute_similarity_multigpu(
        features, query_ids, devices, chunk_size=args.chunk_size
    )

    # 计算TPIR@FPIR
    results, _, _ = compute_tpir_fpir_from_hist(pos_hist, neg_hist)

    print("\n" + "=" * 80)
    print("评估完成！")
    print("=" * 80)

if __name__ == '__main__':
    main()
