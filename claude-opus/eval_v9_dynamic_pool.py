#!/usr/bin/env python3
"""
v9: 动态任务池 + FP16优化
关键改进:
1. 动态任务队列 - 解决负载不均衡
2. FP16精度 - 利用Tensor Core加速2-3倍
3. bins数量优化 - 从10000降到2000
"""
import torch
import numpy as np
import pickle
import time
from multiprocessing import Process, Manager, Queue

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

def create_task_pool(N, block_size=2048):
    """
    创建任务池 - 将上三角矩阵分成小块

    Args:
        N: 矩阵大小
        block_size: 每个任务块大小

    Returns:
        tasks: [(i_start, i_end, j_start, j_end), ...]
    """
    tasks = []

    # 遍历上三角矩阵，分成block_size×block_size的小块
    for i in range(0, N, block_size):
        i_end = min(i + block_size, N)

        # 只处理j > i_end的部分（上三角）
        for j in range(i_end, N, block_size):
            j_end = min(j + block_size, N)

            tasks.append((i, i_end, j, j_end))

    print(f"任务池创建完成：共{len(tasks):,}个任务块")
    print(f"  块大小: {block_size}×{block_size}")
    print(f"  平均每块: {block_size * block_size:,} 对")

    return tasks

def dynamic_worker(gpu_id, task_queue, result_dict, feats, ids, num_bins, use_fp16=True):
    """
    动态worker - 从任务队列取任务，完成后立即取下一个

    Args:
        gpu_id: GPU编号
        task_queue: 任务队列
        result_dict: 结果字典（共享）
        feats: 特征数组
        ids: ID数组
        num_bins: 直方图bins数量
        use_fp16: 是否使用FP16精度
    """
    import sys
    device = torch.device(f'cuda:{gpu_id}')

    # 加载数据到GPU
    feats_tensor = torch.from_numpy(feats).float().to(device)
    if use_fp16:
        feats_tensor = feats_tensor.half()  # 转FP16

    ids_np = ids

    # 初始化本地直方图
    pos_hist = np.zeros(num_bins, dtype=np.int64)
    neg_hist = np.zeros(num_bins, dtype=np.int64)

    total_pairs = 0
    task_count = 0
    start_time = time.time()

    print(f"GPU {gpu_id}: 启动，使用{'FP16' if use_fp16 else 'FP32'}精度", flush=True)
    sys.stdout.flush()

    # 持续从队列取任务
    while True:
        try:
            task = task_queue.get(timeout=1)
            if task is None:  # 结束信号
                break

            i_start, i_end, j_start, j_end = task
            task_count += 1

            # 每100个任务打印一次进度
            if task_count % 100 == 0:
                print(f"GPU {gpu_id}: 已完成{task_count}个任务", flush=True)
                sys.stdout.flush()

            # 计算当前块
            chunk_i = feats_tensor[i_start:i_end]
            chunk_j = feats_tensor[j_start:j_end]
            ids_i = ids_np[i_start:i_end]
            ids_j = ids_np[j_start:j_end]

            # 相似度计算（FP16会自动利用Tensor Core）
            sim_matrix = torch.matmul(chunk_i, chunk_j.T)

            # 转回FP32用于后续计算
            if use_fp16:
                sim_matrix = sim_matrix.float()

            sim_matrix = sim_matrix.cpu().numpy()

            # 判断正负样本
            ids_match = ids_i[:, np.newaxis] == ids_j[np.newaxis, :]

            # 提取相似度值
            pos_sims = sim_matrix[ids_match]
            neg_sims = sim_matrix[~ids_match]

            # 更新直方图
            if len(pos_sims) > 0:
                pos_bins = np.clip((pos_sims * num_bins).astype(np.int32), 0, num_bins - 1)
                np.add.at(pos_hist, pos_bins, 1)

            if len(neg_sims) > 0:
                neg_bins = np.clip((neg_sims * num_bins).astype(np.int32), 0, num_bins - 1)
                np.add.at(neg_hist, neg_bins, 1)

            total_pairs += sim_matrix.size

        except Exception as e:
            if "Empty" not in str(e):
                print(f"GPU {gpu_id}: 错误 - {e}", flush=True)
                sys.stdout.flush()
            break

    elapsed = time.time() - start_time

    print(f"GPU {gpu_id}: 完成{task_count:,}个任务，耗时{elapsed:.2f}秒", flush=True)
    print(f"GPU {gpu_id}: 平均速度 {total_pairs/elapsed:,.0f} 对/秒", flush=True)
    sys.stdout.flush()

    # 返回结果到共享字典
    result_dict[gpu_id] = {
        'gpu_id': gpu_id,
        'pos_hist': pos_hist,
        'neg_hist': neg_hist,
        'total_pairs': total_pairs,
        'elapsed': elapsed,
        'task_count': task_count
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
    total_tasks = 0

    # 简单的数组相加
    for result in results:
        pos_hist_merged += result['pos_hist']
        neg_hist_merged += result['neg_hist']
        total_pairs += result['total_pairs']
        total_compute_time += result['elapsed']
        total_tasks += result['task_count']

    merge_time = time.time() - start_time
    print(f"直方图合并完成！耗时: {merge_time:.3f} 秒")
    print(f"  总任务数: {total_tasks:,}")
    print(f"  总计算量: {total_pairs:,} 对")

    return pos_hist_merged, neg_hist_merged, total_pairs, total_compute_time

def compute_tpir_from_histogram(pos_hist, neg_hist, fpir_values, num_bins):
    """从直方图计算TPIR@FPIR"""
    print("\n开始从直方图计算TPIR@FPIR...")
    start_time = time.time()

    total_pos = pos_hist.sum()
    total_neg = neg_hist.sum()

    print(f"  正样本数: {total_pos:,}")
    print(f"  负样本数: {total_neg:,}")

    # 从右到左累加
    neg_cumsum = np.cumsum(neg_hist[::-1])[::-1]
    pos_cumsum = np.cumsum(pos_hist[::-1])[::-1]

    results = []
    print("\n" + "="*60)
    print("TPIR @ FPIR 评估结果（基于直方图）")
    print("="*60)

    for fpir in fpir_values:
        target_neg_count = total_neg * fpir
        bin_idx = np.searchsorted(neg_cumsum[::-1], target_neg_count)
        bin_idx = num_bins - 1 - bin_idx

        threshold = (bin_idx + 0.5) / num_bins
        tpir = pos_cumsum[bin_idx] / total_pos if total_pos > 0 else 0

        print(f"TPIR @ FPIR={fpir:.0e}: {tpir*100:.2f}% (阈值={threshold:.4f})")

        results.append({
            'fpir': fpir,
            'tpir': tpir,
            'threshold': threshold
        })

    print("="*60)

    elapsed = time.time() - start_time
    print(f"\nTPIR计算完成！耗时: {elapsed:.3f} 秒")

    return results

def main():
    # 加载数据
    feats, ids, paths = load_data()
    N = len(feats)

    # 参数设置
    num_gpus = 7
    num_bins = 2000  # 从10000降到2000（精度0.0005）
    block_size = 2048  # 任务块大小
    use_fp16 = True  # 使用FP16精度

    print(f"\n配置:")
    print(f"  GPU数量: {num_gpus}")
    print(f"  直方图bins: {num_bins:,}")
    print(f"  任务块大小: {block_size}×{block_size}")
    print(f"  计算精度: {'FP16' if use_fp16 else 'FP32'}")
    print(f"  单个直方图大小: {num_bins * 8 / 1024:.2f} KB")

    # 创建任务池
    tasks = create_task_pool(N, block_size)

    # 创建任务队列和结果字典
    manager = Manager()
    task_queue = manager.Queue()
    result_dict = manager.dict()

    # 将任务放入队列
    for task in tasks:
        task_queue.put(task)

    # 放入结束信号
    for _ in range(num_gpus):
        task_queue.put(None)

    print(f"\n启动 {num_gpus} 个GPU进程（动态任务池）...")
    overall_start = time.time()

    # 启动worker进程
    processes = []
    for gpu_id in range(num_gpus):
        p = Process(
            target=dynamic_worker,
            args=(gpu_id, task_queue, result_dict, feats, ids, num_bins, use_fp16)
        )
        p.start()
        processes.append(p)

    # 等待所有进程完成
    for p in processes:
        p.join()

    compute_time = time.time() - overall_start
    print(f"\n所有GPU计算完成！耗时: {compute_time:.2f} 秒")

    # 收集结果
    results = [result_dict[i] for i in range(num_gpus)]

    # 合并直方图
    pos_hist, neg_hist, total_pairs, total_compute_time = merge_histograms(
        results, num_gpus
    )

    avg_speed = total_pairs / compute_time if compute_time > 0 else 0
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
    tpir_results = compute_tpir_from_histogram(pos_hist, neg_hist, fpir_values, num_bins)

    total_time = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"任务全部完成！总耗时: {total_time:.2f} 秒")
    print(f"{'='*60}")

    print(f"\n关键优化:")
    print(f"  ✅ 动态任务池: 负载自动均衡")
    print(f"  ✅ FP16精度: 利用Tensor Core加速")
    print(f"  ✅ bins优化: {num_bins:,} bins（精度{1/num_bins:.6f}）")
    print(f"  ✅ 内存占用: {num_bins * 8 * 2 / 1024:.2f} KB")

    # 负载均衡统计
    print(f"\n负载均衡统计:")
    for result in results:
        print(f"  GPU {result['gpu_id']}: {result['task_count']:,}个任务, "
              f"{result['elapsed']:.2f}秒")

    # 计算负载均衡度
    times = [r['elapsed'] for r in results]
    max_time = max(times)
    min_time = min(times)
    balance_ratio = min_time / max_time if max_time > 0 else 0
    print(f"\n负载均衡度: {balance_ratio*100:.1f}% "
          f"(最快{min_time:.2f}s vs 最慢{max_time:.2f}s)")

if __name__ == '__main__':
    main()
