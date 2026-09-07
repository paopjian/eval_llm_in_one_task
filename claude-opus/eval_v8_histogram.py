#!/usr/bin/env python3
"""
v8: 直方图优化版本 - 内存从82GB降到40KB
关键优化: 不保存所有相似度值，只保存分布直方图
"""
import torch
import numpy as np
import pickle
import time
from multiprocessing import Process, Manager

def load_data():
    """加载特征数据"""
    print("正在加载数据...")
    with open('../s4_0618_enhance.pkl', 'rb') as f:
        feats, feats_flip, ids, paths = pickle.load(f)

    print(f"数据加载完成！")
    print(f"  样本数: {len(feats):,}")
    print(f"  特征维度: {feats.shape[1]}")
    print(f"  唯一ID数: {len(np.unique(ids)):,}")

    return feats, ids, paths

def compute_histogram_worker(gpu_id, start_idx, end_idx, feats, ids, num_bins, result_dict):
    """
    单GPU工作进程 - 计算直方图而非存储所有值

    Args:
        gpu_id: GPU编号
        start_idx, end_idx: 处理的行范围
        feats: 特征数组
        ids: ID数组
        num_bins: 直方图bins数量
        result_dict: 共享字典存储结果
    """
    device = torch.device(f'cuda:{gpu_id}')
    chunk_size = 5000

    # 初始化直方图（使用int64避免溢出）
    pos_hist = np.zeros(num_bins, dtype=np.int64)
    neg_hist = np.zeros(num_bins, dtype=np.int64)

    feats_tensor = torch.from_numpy(feats).float().to(device)
    ids_np = ids

    total_pairs = 0
    start_time = time.time()

    print(f"GPU {gpu_id}: 开始处理行 {start_idx} 到 {end_idx}")

    # 遍历当前GPU负责的行
    for i in range(start_idx, end_idx, chunk_size):
        i_end = min(i + chunk_size, end_idx)
        chunk_i = feats_tensor[i:i_end]
        ids_i = ids_np[i:i_end]

        # 只计算上三角矩阵 (j > i_end)
        for j in range(i_end, len(feats), chunk_size):
            j_end = min(j + chunk_size, len(feats))
            chunk_j = feats_tensor[j:j_end]
            ids_j = ids_np[j:j_end]

            # 计算相似度
            sim_matrix = torch.matmul(chunk_i, chunk_j.T).cpu().numpy()

            # 判断正负样本
            ids_match = ids_i[:, np.newaxis] == ids_j[np.newaxis, :]

            # 提取相似度值
            pos_sims = sim_matrix[ids_match]
            neg_sims = sim_matrix[~ids_match]

            # 更新直方图（值域[0,1]映射到[0, num_bins-1]）
            if len(pos_sims) > 0:
                pos_bins = np.clip((pos_sims * num_bins).astype(np.int32), 0, num_bins - 1)
                np.add.at(pos_hist, pos_bins, 1)

            if len(neg_sims) > 0:
                neg_bins = np.clip((neg_sims * num_bins).astype(np.int32), 0, num_bins - 1)
                np.add.at(neg_hist, neg_bins, 1)

            total_pairs += sim_matrix.size

    elapsed = time.time() - start_time
    speed = total_pairs / elapsed if elapsed > 0 else 0

    print(f"GPU {gpu_id}: 计算完成，耗时 {elapsed:.2f} 秒")
    print(f"GPU {gpu_id}: 直方图大小 {pos_hist.nbytes + neg_hist.nbytes} 字节")

    # 只返回直方图（40KB），而非所有值（GB级）
    result_dict[gpu_id] = {
        'pos_hist': pos_hist,
        'neg_hist': neg_hist,
        'total_pairs': total_pairs,
        'elapsed': elapsed
    }

def merge_histograms(results, num_gpus):
    """合并多个GPU的直方图"""
    print("\n正在合并直方图...")
    start_time = time.time()

    # 初始化
    num_bins = results[0]['pos_hist'].shape[0]
    pos_hist_merged = np.zeros(num_bins, dtype=np.int64)
    neg_hist_merged = np.zeros(num_bins, dtype=np.int64)

    total_pairs = 0
    total_compute_time = 0

    # 简单的数组相加（非常快！）
    for gpu_id in range(num_gpus):
        pos_hist_merged += results[gpu_id]['pos_hist']
        neg_hist_merged += results[gpu_id]['neg_hist']
        total_pairs += results[gpu_id]['total_pairs']
        total_compute_time += results[gpu_id]['elapsed']

    merge_time = time.time() - start_time
    print(f"直方图合并完成！耗时: {merge_time:.2f} 秒")
    print(f"  合并的数据量: {(pos_hist_merged.nbytes + neg_hist_merged.nbytes) * num_gpus / 1024:.2f} KB")

    return pos_hist_merged, neg_hist_merged, total_pairs, total_compute_time

def compute_tpir_from_histogram(pos_hist, neg_hist, fpir_values, num_bins):
    """
    从直方图计算TPIR@FPIR

    Args:
        pos_hist: 正样本直方图
        neg_hist: 负样本直方图
        fpir_values: 要计算的FPIR值列表
        num_bins: 直方图bins数量
    """
    print("\n开始从直方图计算TPIR@FPIR...")
    start_time = time.time()

    total_pos = pos_hist.sum()
    total_neg = neg_hist.sum()

    print(f"  正样本数: {total_pos:,}")
    print(f"  负样本数: {total_neg:,}")

    # 从右到左累加（相似度从高到低）
    neg_cumsum = np.cumsum(neg_hist[::-1])[::-1]  # 累积和（从右到左）
    pos_cumsum = np.cumsum(pos_hist[::-1])[::-1]

    results = []
    print("\n" + "="*60)
    print("TPIR @ FPIR 评估结果（基于直方图）")
    print("="*60)

    for fpir in fpir_values:
        # 找到对应FPIR的阈值bin
        target_neg_count = total_neg * fpir

        # 二分查找（在直方图上，非常快！）
        bin_idx = np.searchsorted(neg_cumsum[::-1], target_neg_count)
        bin_idx = num_bins - 1 - bin_idx

        # 阈值是bin的中心值
        threshold = (bin_idx + 0.5) / num_bins

        # 计算TPIR
        tpir = pos_cumsum[bin_idx] / total_pos if total_pos > 0 else 0

        print(f"TPIR @ FPIR={fpir:.0e}: {tpir*100:.2f}% (阈值={threshold:.4f})")

        results.append({
            'fpir': fpir,
            'tpir': tpir,
            'threshold': threshold
        })

    print("="*60)

    elapsed = time.time() - start_time
    print(f"\nTPIR计算完成！耗时: {elapsed:.2f} 秒")

    return results

def main():
    # 加载数据
    feats, ids, paths = load_data()
    N = len(feats)

    # 参数设置
    num_gpus = 7
    num_bins = 10000  # 直方图精度：0.0001

    print(f"\n配置:")
    print(f"  GPU数量: {num_gpus}")
    print(f"  直方图bins: {num_bins:,}")
    print(f"  单个直方图大小: {num_bins * 8 / 1024:.2f} KB")
    print(f"  总内存占用: {num_bins * 8 * 2 / 1024:.2f} KB (vs 之前的82GB!)")

    # 分配任务
    rows_per_gpu = N // num_gpus

    # 多进程计算
    manager = Manager()
    result_dict = manager.dict()
    processes = []

    print(f"\n启动 {num_gpus} 个GPU进程...")
    overall_start = time.time()

    for gpu_id in range(num_gpus):
        start_idx = gpu_id * rows_per_gpu
        end_idx = (gpu_id + 1) * rows_per_gpu if gpu_id < num_gpus - 1 else N

        p = Process(
            target=compute_histogram_worker,
            args=(gpu_id, start_idx, end_idx, feats, ids, num_bins, result_dict)
        )
        p.start()
        processes.append(p)

    # 等待所有进程完成
    for p in processes:
        p.join()

    compute_time = time.time() - overall_start
    print(f"\n所有GPU计算完成！耗时: {compute_time:.2f} 秒")

    # 合并直方图
    pos_hist, neg_hist, total_pairs, total_compute_time = merge_histograms(
        dict(result_dict), num_gpus
    )

    avg_speed = total_pairs / total_compute_time if total_compute_time > 0 else 0
    print(f"  平均速度: {avg_speed:,.0f} 对/秒")

    # 统计信息
    total_pos = pos_hist.sum()
    total_neg = neg_hist.sum()

    # 从直方图估算均值
    bin_centers = (np.arange(num_bins) + 0.5) / num_bins
    pos_mean = np.sum(pos_hist * bin_centers) / total_pos if total_pos > 0 else 0
    neg_mean = np.sum(neg_hist * bin_centers) / total_neg if total_neg > 0 else 0

    print(f"\n相似度统计（从直方图估算）:")
    print(f"  正样本相似度 - 均值: {pos_mean:.4f}")
    print(f"  负样本相似度 - 均值: {neg_mean:.4f}")

    # 计算TPIR@FPIR
    fpir_values = [1e-5, 1e-4, 1e-3, 1e-2]
    results = compute_tpir_from_histogram(pos_hist, neg_hist, fpir_values, num_bins)

    total_time = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"任务全部完成！总耗时: {total_time:.2f} 秒")
    print(f"{'='*60}")
    print(f"\n关键优化:")
    print(f"  ✅ 内存占用: {num_bins * 8 * 2 / 1024:.2f} KB (vs 82GB)")
    print(f"  ✅ 数据传输: {num_bins * 8 * 2 * num_gpus / 1024:.2f} KB (vs 82GB * 7)")
    print(f"  ✅ 合并速度: 直方图相加（毫秒级）")
    print(f"  ✅ 阈值计算: 在直方图上二分查找（毫秒级）")

if __name__ == '__main__':
    main()
