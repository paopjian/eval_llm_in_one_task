#!/usr/bin/env python3
"""
版本2: 多卡并行加速
使用7张GPU并行计算相似度矩阵
"""
import pickle
import numpy as np
import torch
import torch.multiprocessing as mp
import time
from collections import defaultdict

def load_data(data_path='s4_0618_enhance.pkl'):
    """加载特征数据"""
    print("正在加载数据...")
    with open(data_path, 'rb') as f:
        query_feats_list, _, query_ids, file_paths = pickle.load(f)
    print(f"数据加载完成: {len(query_ids)} 个样本")
    return query_feats_list, query_ids, file_paths

def compute_similarity_worker(gpu_id, feats, ids, task_queue, result_queue, chunk_size=5000):
    """
    单个GPU的工作进程
    从任务队列获取计算任务，将结果放入结果队列
    """
    device = f'cuda:{gpu_id}'
    feats_tensor = torch.from_numpy(feats).to(device)
    ids_tensor = torch.from_numpy(ids).to(device)

    while True:
        task = task_queue.get()
        if task is None:  # 结束信号
            break

        i, j = task
        i_end = min(i + chunk_size, len(feats))
        j_end = min(j + chunk_size, len(feats))

        chunk_i = feats_tensor[i:i_end]
        chunk_j = feats_tensor[j:j_end]
        ids_i = ids_tensor[i:i_end]
        ids_j = ids_tensor[j:j_end]

        # 计算相似度
        sim_matrix = torch.matmul(chunk_i, chunk_j.T)
        ids_match = ids_i.unsqueeze(1) == ids_j.unsqueeze(0)

        # 处理上三角
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

        pos_sims = sim_values[pos_mask].cpu().numpy() if pos_mask.any() else np.array([])
        neg_sims = sim_values[neg_mask].cpu().numpy() if neg_mask.any() else np.array([])

        # 返回结果
        result_queue.put((len(sim_values), pos_sims, neg_sims))

def compute_similarity_multigpu(feats, ids, num_gpus=7, chunk_size=5000):
    """
    多GPU并行计算相似度矩阵
    """
    N = len(feats)
    total_pairs = N * (N - 1) // 2

    print(f"\n开始多卡并行计算 (GPU数量: {num_gpus}, 分块大小: {chunk_size})")
    print(f"总样本对数: {total_pairs:,}")

    # 创建任务队列和结果队列
    mp.set_start_method('spawn', force=True)
    task_queue = mp.Queue()
    result_queue = mp.Queue()

    # 生成所有任务 (i, j) 对
    tasks = []
    for i in range(0, N, chunk_size):
        for j in range(i, N, chunk_size):
            tasks.append((i, j))

    print(f"总任务数: {len(tasks)}")

    # 将任务放入队列
    for task in tasks:
        task_queue.put(task)

    # 添加结束信号
    for _ in range(num_gpus):
        task_queue.put(None)

    # 启动工作进程
    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(target=compute_similarity_worker,
                      args=(gpu_id, feats, ids, task_queue, result_queue, chunk_size))
        p.start()
        processes.append(p)

    # 收集结果
    pos_similarities = []
    neg_similarities = []
    processed_pairs = 0
    start_time = time.time()

    for _ in range(len(tasks)):
        pair_count, pos_sims, neg_sims = result_queue.get()
        processed_pairs += pair_count

        if len(pos_sims) > 0:
            pos_similarities.append(pos_sims)
        if len(neg_sims) > 0:
            neg_similarities.append(neg_sims)

        # 更新进度
        if processed_pairs % 100_000_000 == 0 or processed_pairs > total_pairs * 0.9:
            elapsed = time.time() - start_time
            progress = processed_pairs / total_pairs * 100
            speed = processed_pairs / elapsed if elapsed > 0 else 0
            eta = (total_pairs - processed_pairs) / speed if speed > 0 else 0
            print(f"  进度: {progress:.2f}% | "
                  f"已处理: {processed_pairs:,} 对 | "
                  f"速度: {speed:,.0f} 对/秒 | "
                  f"预计剩余: {eta:.1f}秒")

    # 等待所有进程结束
    for p in processes:
        p.join()

    elapsed = time.time() - start_time
    print(f"\n相似度计算完成！")
    print(f"  总耗时: {elapsed:.2f} 秒")
    print(f"  平均速度: {total_pairs/elapsed:,.0f} 对/秒")
    print(f"  加速比: {47.0/elapsed:.2f}x (相比单卡)")

    # 合并结果
    pos_similarities = np.concatenate(pos_similarities)
    neg_similarities = np.concatenate(neg_similarities)

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
    print("人脸特征相似度评估系统 - 版本2: 多卡并行")
    print("="*80)

    # 1. 加载数据
    feats, ids, paths = load_data()

    # 2. 多卡并行计算相似度
    pos_sims, neg_sims = compute_similarity_multigpu(feats, ids, num_gpus=7, chunk_size=5000)

    print(f"\n相似度统计:")
    print(f"  正样本相似度 - 均值: {pos_sims.mean():.4f}, 标准差: {pos_sims.std():.4f}")
    print(f"  负样本相似度 - 均值: {neg_sims.mean():.4f}, 标准差: {neg_sims.std():.4f}")

    # 3. 计算TPIR@FPIR
    results = compute_tpir_at_fpir(pos_sims, neg_sims)

    print("\n任务完成！")
    return results

if __name__ == '__main__':
    main()
