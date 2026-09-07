#!/usr/bin/env python3
"""
版本6: 智能优化版本
- 多卡并行计算
- 采样估计阈值，避免对巨量数据进行partition
- 在线统计，避免存储所有中间数据
"""
import pickle
import numpy as np
import torch
import torch.multiprocessing as mp
import time
import os
import tempfile

def load_data(data_path='s4_0618_enhance.pkl'):
    """加载特征数据"""
    print("正在加载数据...")
    with open(data_path, 'rb') as f:
        query_feats_list, _, query_ids, file_paths = pickle.load(f)
    print(f"数据加载完成: {len(query_ids)} 个样本")
    return query_feats_list, query_ids, file_paths

def compute_row_range_worker_with_stats(gpu_id, feats, ids, row_start, row_end, output_file, chunk_size=5000):
    """
    单个GPU的工作进程
    返回统计信息和采样数据，而不是全部数据
    """
    device = f'cuda:{gpu_id}'
    N = len(feats)

    feats_tensor = torch.from_numpy(feats).to(device)
    ids_tensor = torch.from_numpy(ids).to(device)

    # 存储采样数据（用于后续阈值估计）
    pos_sample = []
    neg_sample = []
    sample_rate = 0.001  # 采样率0.1%

    # 统计信息
    pos_count = 0
    neg_count = 0
    pos_sum = 0.0
    neg_sum = 0.0

    print(f"GPU {gpu_id}: 开始处理行 {row_start} 到 {row_end}")
    start_time = time.time()

    np.random.seed(gpu_id)  # 确保可重现

    for i in range(row_start, row_end, chunk_size):
        i_end = min(i + chunk_size, row_end)
        chunk_i = feats_tensor[i:i_end]
        ids_i = ids_tensor[i:i_end]

        for j in range(i, N, chunk_size):
            j_end = min(j + chunk_size, N)
            chunk_j = feats_tensor[j:j_end]
            ids_j = ids_tensor[j:j_end]

            sim_matrix = torch.matmul(chunk_i, chunk_j.T)
            ids_match = ids_i.unsqueeze(1) == ids_j.unsqueeze(0)

            if i == j:
                mask = torch.triu(torch.ones_like(sim_matrix, dtype=torch.bool), diagonal=1)
                sim_values = sim_matrix[mask]
                match_values = ids_match[mask]
            else:
                sim_values = sim_matrix.flatten()
                match_values = ids_match.flatten()

            # 转换到CPU
            sim_values_cpu = sim_values.cpu().numpy()
            match_values_cpu = match_values.cpu().numpy()

            # 正样本统计
            pos_mask = match_values_cpu
            if pos_mask.any():
                pos_sims = sim_values_cpu[pos_mask]
                pos_count += len(pos_sims)
                pos_sum += pos_sims.sum()
                # 采样
                sample_size = max(1, int(len(pos_sims) * sample_rate))
                pos_sample.append(np.random.choice(pos_sims, size=sample_size, replace=False))

            # 负样本统计
            neg_mask = ~match_values_cpu
            if neg_mask.any():
                neg_sims = sim_values_cpu[neg_mask]
                neg_count += len(neg_sims)
                neg_sum += neg_sims.sum()
                # 采样
                sample_size = max(1, int(len(neg_sims) * sample_rate))
                neg_sample.append(np.random.choice(neg_sims, size=sample_size, replace=False))

    elapsed = time.time() - start_time
    print(f"GPU {gpu_id}: 计算完成，耗时 {elapsed:.2f} 秒，正在保存...")

    # 合并采样数据
    pos_sample = np.concatenate(pos_sample) if pos_sample else np.array([])
    neg_sample = np.concatenate(neg_sample) if neg_sample else np.array([])

    # 保存统计信息和采样数据
    result = {
        'pos_count': pos_count,
        'neg_count': neg_count,
        'pos_sum': pos_sum,
        'neg_sum': neg_sum,
        'pos_sample': pos_sample,
        'neg_sample': neg_sample
    }

    with open(output_file, 'wb') as f:
        pickle.dump(result, f)

    print(f"GPU {gpu_id}: 完成！")

def compute_similarity_multigpu_smart(feats, ids, num_gpus=7, chunk_size=5000):
    """智能多GPU并行计算"""
    N = len(feats)
    total_pairs = N * (N - 1) // 2

    print(f"\n开始多卡并行计算（智能模式）")
    print(f"  GPU数量: {num_gpus}")
    print(f"  分块大小: {chunk_size}")
    print(f"  总样本对数: {total_pairs:,}")

    rows_per_gpu = N // num_gpus
    row_ranges = []
    for i in range(num_gpus):
        start = i * rows_per_gpu
        end = N if i == num_gpus - 1 else (i + 1) * rows_per_gpu
        row_ranges.append((start, end))
        print(f"  GPU {i}: 行 {start} 到 {end} (共 {end-start} 行)")

    temp_dir = tempfile.mkdtemp()
    output_files = [os.path.join(temp_dir, f'gpu_{i}.pkl') for i in range(num_gpus)]

    mp.set_start_method('spawn', force=True)
    processes = []
    start_time = time.time()

    for gpu_id, (row_start, row_end) in enumerate(row_ranges):
        p = mp.Process(target=compute_row_range_worker_with_stats,
                      args=(gpu_id, feats, ids, row_start, row_end, output_files[gpu_id], chunk_size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    elapsed = time.time() - start_time
    print(f"\n所有GPU计算完成！耗时: {elapsed:.2f} 秒")
    print(f"  平均速度: {total_pairs/elapsed:,.0f} 对/秒")

    print("\n正在合并结果...")
    pos_count_total = 0
    neg_count_total = 0
    pos_sum_total = 0.0
    neg_sum_total = 0.0
    all_pos_samples = []
    all_neg_samples = []

    for i, output_file in enumerate(output_files):
        with open(output_file, 'rb') as f:
            result = pickle.load(f)

        pos_count_total += result['pos_count']
        neg_count_total += result['neg_count']
        pos_sum_total += result['pos_sum']
        neg_sum_total += result['neg_sum']

        if len(result['pos_sample']) > 0:
            all_pos_samples.append(result['pos_sample'])
        if len(result['neg_sample']) > 0:
            all_neg_samples.append(result['neg_sample'])

        os.remove(output_file)

    os.rmdir(temp_dir)

    pos_sample = np.concatenate(all_pos_samples)
    neg_sample = np.concatenate(all_neg_samples)

    print(f"结果合并完成！")
    print(f"  正样本总数: {pos_count_total:,}")
    print(f"  负样本总数: {neg_count_total:,}")
    print(f"  正样本采样数: {len(pos_sample):,}")
    print(f"  负样本采样数: {len(neg_sample):,}")

    stats = {
        'pos_count': pos_count_total,
        'neg_count': neg_count_total,
        'pos_mean': pos_sum_total / pos_count_total if pos_count_total > 0 else 0,
        'neg_mean': neg_sum_total / neg_count_total if neg_count_total > 0 else 0,
        'pos_sample': pos_sample,
        'neg_sample': neg_sample
    }

    return stats

def compute_tpir_at_fpir_from_sample(stats, fpir_targets=[1e-5, 1e-4, 1e-3, 1e-2]):
    """
    基于采样数据估计TPIR@FPIR
    """
    print(f"\n开始计算TPIR@FPIR指标（基于采样）...")

    pos_sample = stats['pos_sample']
    neg_sample = stats['neg_sample']
    pos_count = stats['pos_count']
    neg_count = stats['neg_count']

    # 对采样数据排序
    print("  正在排序采样数据...")
    neg_sample_sorted = np.sort(neg_sample)[::-1]

    results = {}
    print("\n" + "="*60)
    print("TPIR @ FPIR 评估结果（基于采样）")
    print("="*60)

    for fpir in fpir_targets:
        # 在采样数据中找到对应FPIR的阈值
        sample_neg_count = int(fpir * len(neg_sample))
        if sample_neg_count >= len(neg_sample):
            threshold = neg_sample_sorted[-1]
        else:
            threshold = neg_sample_sorted[sample_neg_count]

        # 在正样本采样中计算TPIR
        tpir = (pos_sample > threshold).sum() / len(pos_sample)

        results[fpir] = {'threshold': threshold, 'tpir': tpir}
        print(f"TPIR @ FPIR={fpir:.0e}: {tpir*100:.2f}% (阈值={threshold:.4f})")

    print("="*60)
    print("注意：这是基于0.1%采样数据的估计结果")
    return results

def main():
    total_start = time.time()

    print("="*80)
    print("人脸特征相似度评估系统 - 版本6: 智能采样版")
    print("="*80)

    # 1. 加载数据
    feats, ids, paths = load_data()

    # 2. 多卡并行计算（采样模式）
    stats = compute_similarity_multigpu_smart(feats, ids, num_gpus=7, chunk_size=5000)

    print(f"\n相似度统计:")
    print(f"  正样本相似度 - 均值: {stats['pos_mean']:.4f}")
    print(f"  负样本相似度 - 均值: {stats['neg_mean']:.4f}")

    # 3. 计算TPIR@FPIR（基于采样）
    results = compute_tpir_at_fpir_from_sample(stats)

    total_time = time.time() - total_start
    print(f"\n" + "="*60)
    print(f"任务全部完成！总耗时: {total_time:.2f} 秒")
    print("="*60)

    return results

if __name__ == '__main__':
    main()
