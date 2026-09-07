#!/usr/bin/env python3
"""
版本3: 优化的多卡并行
策略：每个GPU负责连续的行区间，减少进程间通信
"""
import pickle
import numpy as np
import torch
import torch.multiprocessing as mp
import time

def load_data(data_path='s4_0618_enhance.pkl'):
    """加载特征数据"""
    print("正在加载数据...")
    with open(data_path, 'rb') as f:
        query_feats_list, _, query_ids, file_paths = pickle.load(f)
    print(f"数据加载完成: {len(query_ids)} 个样本")
    return query_feats_list, query_ids, file_paths

def compute_row_range_worker(gpu_id, feats, ids, row_start, row_end, shared_dict, chunk_size=5000):
    """
    单个GPU负责计算从row_start到row_end的所有行
    只计算上三角部分：对于行i，计算j从i到N的所有列
    """
    device = f'cuda:{gpu_id}'
    N = len(feats)

    # 将数据移到GPU
    feats_tensor = torch.from_numpy(feats).to(device)
    ids_tensor = torch.from_numpy(ids).to(device)

    pos_similarities = []
    neg_similarities = []

    print(f"GPU {gpu_id}: 开始处理行 {row_start} 到 {row_end}")
    start_time = time.time()

    # 遍历该GPU负责的每一行
    for i in range(row_start, row_end, chunk_size):
        i_end = min(i + chunk_size, row_end)
        chunk_i = feats_tensor[i:i_end]
        ids_i = ids_tensor[i:i_end]

        # 对于这些行，计算j从i到N的所有列（上三角）
        for j in range(i, N, chunk_size):
            j_end = min(j + chunk_size, N)
            chunk_j = feats_tensor[j:j_end]
            ids_j = ids_tensor[j:j_end]

            # 计算相似度
            sim_matrix = torch.matmul(chunk_i, chunk_j.T)
            ids_match = ids_i.unsqueeze(1) == ids_j.unsqueeze(0)

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

    elapsed = time.time() - start_time
    print(f"GPU {gpu_id}: 完成！耗时 {elapsed:.2f} 秒")

    # 合并结果
    pos_sims = np.concatenate(pos_similarities) if pos_similarities else np.array([])
    neg_sims = np.concatenate(neg_similarities) if neg_similarities else np.array([])

    # 保存结果到共享字典
    shared_dict[gpu_id] = (pos_sims, neg_sims)

def compute_similarity_multigpu_optimized(feats, ids, num_gpus=7, chunk_size=5000):
    """
    优化的多GPU并行：按行区间划分任务
    """
    N = len(feats)
    total_pairs = N * (N - 1) // 2

    print(f"\n开始优化的多卡并行计算")
    print(f"  GPU数量: {num_gpus}")
    print(f"  分块大小: {chunk_size}")
    print(f"  总样本对数: {total_pairs:,}")

    # 按行划分任务给各个GPU
    rows_per_gpu = N // num_gpus
    row_ranges = []
    for i in range(num_gpus):
        start = i * rows_per_gpu
        end = N if i == num_gpus - 1 else (i + 1) * rows_per_gpu
        row_ranges.append((start, end))
        print(f"  GPU {i}: 行 {start} 到 {end} (共 {end-start} 行)")

    # 使用Manager创建共享字典
    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    shared_dict = manager.dict()

    # 启动工作进程
    processes = []
    start_time = time.time()

    for gpu_id, (row_start, row_end) in enumerate(row_ranges):
        p = mp.Process(target=compute_row_range_worker,
                      args=(gpu_id, feats, ids, row_start, row_end, shared_dict, chunk_size))
        p.start()
        processes.append(p)

    # 等待所有进程完成
    for p in processes:
        p.join()

    elapsed = time.time() - start_time
    print(f"\n所有GPU计算完成！")
    print(f"  总耗时: {elapsed:.2f} 秒")
    print(f"  平均速度: {total_pairs/elapsed:,.0f} 对/秒")

    # 合并所有GPU的结果
    print("\n正在合并结果...")
    all_pos_sims = []
    all_neg_sims = []

    for gpu_id in range(num_gpus):
        pos_sims, neg_sims = shared_dict[gpu_id]
        if len(pos_sims) > 0:
            all_pos_sims.append(pos_sims)
        if len(neg_sims) > 0:
            all_neg_sims.append(neg_sims)

    pos_similarities = np.concatenate(all_pos_sims)
    neg_similarities = np.concatenate(all_neg_sims)

    print(f"结果合并完成！")

    return pos_similarities, neg_similarities

def compute_tpir_at_fpir(pos_sims, neg_sims, fpir_targets=[1e-5, 1e-4, 1e-3, 1e-2]):
    """计算TPIR@FPIR指标"""
    print(f"\n开始计算TPIR@FPIR指标...")
    print(f"  正样本数: {len(pos_sims):,}")
    print(f"  负样本数: {len(neg_sims):,}")

    print("  正在排序负样本相似度...")
    neg_sims_sorted = np.sort(neg_sims)[::-1]

    results = {}
    print("\n" + "="*60)
    print("TPIR @ FPIR 评估结果")
    print("="*60)

    for fpir in fpir_targets:
        neg_count = int(fpir * len(neg_sims))
        if neg_count >= len(neg_sims):
            threshold = neg_sims_sorted[-1]
        else:
            threshold = neg_sims_sorted[neg_count]

        tpir = (pos_sims > threshold).sum() / len(pos_sims)
        results[fpir] = {'threshold': threshold, 'tpir': tpir}
        print(f"TPIR @ FPIR={fpir:.0e}: {tpir*100:.2f}% (阈值={threshold:.4f})")

    print("="*60)
    return results

def main():
    print("="*80)
    print("人脸特征相似度评估系统 - 版本3: 优化的多卡并行")
    print("="*80)

    # 1. 加载数据
    feats, ids, paths = load_data()

    # 2. 优化的多卡并行计算
    pos_sims, neg_sims = compute_similarity_multigpu_optimized(feats, ids, num_gpus=7, chunk_size=5000)

    print(f"\n相似度统计:")
    print(f"  正样本相似度 - 均值: {pos_sims.mean():.4f}, 标准差: {pos_sims.std():.4f}")
    print(f"  负样本相似度 - 均值: {neg_sims.mean():.4f}, 标准差: {neg_sims.std():.4f}")

    # 3. 计算TPIR@FPIR
    results = compute_tpir_at_fpir(pos_sims, neg_sims)

    print("\n任务完成！")
    return results

if __name__ == '__main__':
    main()
