#!/usr/bin/env python3
"""
版本5: 最终优化版本
- 多卡并行计算
- 使用numpy.partition替代完整排序，大幅加速
- 添加样本对提取功能
"""
import pickle
import numpy as np
import torch
import torch.multiprocessing as mp
import time
import os
import tempfile

def load_data(data_path):
    """加载特征数据"""
    print("正在加载数据...")
    with open(data_path, 'rb') as f:
        query_feats_list, _, query_ids, file_paths = pickle.load(f)
    print(f"数据加载完成: {len(query_ids)} 个样本")
    return query_feats_list, query_ids, file_paths

def compute_row_range_worker(gpu_id, feats, ids, row_start, row_end, output_file, chunk_size=5000):
    """单个GPU的工作进程"""
    device = f'cuda:{gpu_id}'
    N = len(feats)

    feats_tensor = torch.from_numpy(feats).to(device)
    ids_tensor = torch.from_numpy(ids).to(device)

    pos_similarities = []
    neg_similarities = []

    print(f"GPU {gpu_id}: 开始处理行 {row_start} 到 {row_end}")
    start_time = time.time()

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

            pos_mask = match_values
            neg_mask = ~match_values

            if pos_mask.any():
                pos_similarities.append(sim_values[pos_mask].cpu().numpy())
            if neg_mask.any():
                neg_similarities.append(sim_values[neg_mask].cpu().numpy())

    elapsed = time.time() - start_time
    print(f"GPU {gpu_id}: 计算完成，耗时 {elapsed:.2f} 秒，正在保存...")

    pos_sims = np.concatenate(pos_similarities) if pos_similarities else np.array([])
    neg_sims = np.concatenate(neg_similarities) if neg_similarities else np.array([])

    with open(output_file, 'wb') as f:
        pickle.dump((pos_sims, neg_sims), f)

    print(f"GPU {gpu_id}: 完成！")

def compute_similarity_multigpu(feats, ids, num_gpus=7, chunk_size=5000):
    """多GPU并行计算"""
    N = len(feats)
    total_pairs = N * (N - 1) // 2

    print(f"\n开始多卡并行计算")
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
        p = mp.Process(target=compute_row_range_worker,
                      args=(gpu_id, feats, ids, row_start, row_end, output_files[gpu_id], chunk_size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    elapsed = time.time() - start_time
    print(f"\n所有GPU计算完成！耗时: {elapsed:.2f} 秒")
    print(f"  平均速度: {total_pairs/elapsed:,.0f} 对/秒")

    print("\n正在合并结果...")
    all_pos_sims = []
    all_neg_sims = []

    for i, output_file in enumerate(output_files):
        with open(output_file, 'rb') as f:
            pos_sims, neg_sims = pickle.load(f)
        if len(pos_sims) > 0:
            all_pos_sims.append(pos_sims)
        if len(neg_sims) > 0:
            all_neg_sims.append(neg_sims)
        os.remove(output_file)

    os.rmdir(temp_dir)

    pos_similarities = np.concatenate(all_pos_sims)
    neg_similarities = np.concatenate(all_neg_sims)

    print(f"结果合并完成！")
    return pos_similarities, neg_similarities

def compute_tpir_at_fpir_fast(pos_sims, neg_sims, fpir_targets=[1e-5, 1e-4, 1e-3, 1e-2]):
    """
    快速计算TPIR@FPIR指标
    使用numpy.partition而不是完整排序，大幅提升速度
    """
    print(f"\n开始计算TPIR@FPIR指标...")
    print(f"  正样本数: {len(pos_sims):,}")
    print(f"  负样本数: {len(neg_sims):,}")

    results = {}
    print("\n" + "="*60)
    print("TPIR @ FPIR 评估结果")
    print("="*60)

    for fpir in fpir_targets:
        # 计算需要的负样本索引位置
        neg_count = int(fpir * len(neg_sims))

        # 使用partition找到第neg_count大的元素，避免完整排序
        # partition后，前neg_count个元素都大于等于第neg_count个元素
        start_time = time.time()

        if neg_count >= len(neg_sims):
            threshold = neg_sims.min()
        elif neg_count == 0:
            threshold = neg_sims.max()
        else:
            # 使用argpartition找到最大的neg_count个元素的索引
            # 然后取第neg_count个位置的值作为阈值
            partitioned_indices = np.argpartition(neg_sims, -neg_count)[-neg_count:]
            threshold = neg_sims[partitioned_indices].min()

        partition_time = time.time() - start_time

        # 计算该阈值下的TPIR
        tpir = (pos_sims > threshold).sum() / len(pos_sims)

        results[fpir] = {'threshold': threshold, 'tpir': tpir}
        print(f"TPIR @ FPIR={fpir:.0e}: {tpir*100:.2f}% (阈值={threshold:.4f}, 耗时={partition_time:.2f}秒)")

    print("="*60)
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser(description='人脸特征相似度评估系统 - 版本5')
    parser.add_argument('--pkl', type=str, required=True, help='数据文件路径（.pkl格式）')
    parser.add_argument('--gpus', type=int, default=7, help='使用的GPU数量（默认7）')
    parser.add_argument('--chunk-size', type=int, default=5000, help='分块大小（默认5000）')
    args = parser.parse_args()

    total_start = time.time()

    print("="*80)
    print("人脸特征相似度评估系统 - 版本5: 最终优化版")
    print("="*80)

    # 1. 加载数据
    feats, ids, paths = load_data(args.pkl)

    # 2. 多卡并行计算
    pos_sims, neg_sims = compute_similarity_multigpu(feats, ids, num_gpus=args.gpus, chunk_size=args.chunk_size)

    print(f"\n相似度统计:")
    print(f"  正样本相似度 - 均值: {pos_sims.mean():.4f}, 标准差: {pos_sims.std():.4f}")
    print(f"  负样本相似度 - 均值: {neg_sims.mean():.4f}, 标准差: {neg_sims.std():.4f}")

    # 3. 计算TPIR@FPIR（使用快速算法）
    results = compute_tpir_at_fpir_fast(pos_sims, neg_sims)

    total_time = time.time() - total_start
    print(f"\n" + "="*60)
    print(f"任务全部完成！总耗时: {total_time:.2f} 秒")
    print("="*60)

    return results, pos_sims, neg_sims

if __name__ == '__main__':
    main()
