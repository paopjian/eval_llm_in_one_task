#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人脸特征相似度评估系统 (Face Feature Similarity Evaluation System)
-------------------------------------------------------------------------
核心功能：
1. 高效读取并解析大规模人脸特征向量 (L2归一化特征、身份标签、图片路径)。
2. 自动区分正负样本对 (同一身份为正样本对，不同身份为负样本对)。
3. 支持多GPU并行分块加速计算 (FP16 Tensor Core GEMM + GPU级高分辨率直方图统计)。
4. 精确计算 TPIR@FPIR (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6 等关键评估点及对应阈值)。
5. 支持提取低于阈值的困难正样本对 (False Negatives) 与高于阈值的困难负样本对 (False Positives)，用于错误归因分析。
6. 使用指定中文字体绘制高清评估图表 (正负样本相似度分布图、TPIR@FPIR ROC特征曲线)。
"""

import os
import sys
import time
import pickle
import argparse
import numpy as np
import torch
import torch.multiprocessing as mp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def load_and_analyze_dataset(data_path):
    """
    加载特征数据集并输出基础统计信息
    """
    print("=" * 80)
    print(f"1. 加载数据集: {data_path}")
    print("=" * 80)
    t0 = time.time()
    with open(data_path, 'rb') as f:
        query_feats, query_feats_flip, query_ids, file_paths = pickle.load(f)
    t_load = time.time() - t0

    N, D = query_feats.shape
    unique_ids, counts = np.unique(query_ids, return_counts=True)
    total_pairs = N * (N - 1) // 2
    total_pos_pairs = int(np.sum(counts.astype(np.int64) * (counts.astype(np.int64) - 1) // 2))
    total_neg_pairs = total_pairs - total_pos_pairs

    # 检查特征归一化
    norms = np.linalg.norm(query_feats, axis=1)
    is_normalized = bool(np.abs(norms.mean() - 1.0) < 0.01)

    print(f"数据加载与结构分析完成 (耗时: {t_load:.2f}s):")
    print(f"  特征矩阵形状: {query_feats.shape} (dtype: {query_feats.dtype})")
    print(f"  特征维度: {D}")
    print(f"  样本总数: {N:,}")
    print(f"  身份总数: {len(unique_ids):,}")
    print(f"  特征是否归一化: {'是 (余弦相似度 = 矩阵乘法)' if is_normalized else '否'}")
    print(f"  总样本对数: {total_pairs:,}")
    print(f"  正样本对数 (Positive Pairs): {total_pos_pairs:,} ({total_pos_pairs/total_pairs*100:.4f}%)")
    print(f"  负样本对数 (Negative Pairs): {total_neg_pairs:,} ({total_neg_pairs/total_pairs*100:.4f}%)")

    return {
        'query_feats': query_feats,
        'query_ids': query_ids,
        'file_paths': file_paths,
        'N': N,
        'D': D,
        'total_pairs': total_pairs,
        'total_pos_pairs': total_pos_pairs,
        'total_neg_pairs': total_neg_pairs,
        'unique_ids': unique_ids,
        'counts': counts
    }


def compute_positive_pairs(query_feats, query_ids):
    """
    精确计算所有正样本对的相似度 (同一身份的所有组合)
    返回正样本对的相似度数组和元数据 (用于提取困难正样本)
    """
    print("\n" + "=" * 80)
    print("2. 计算正样本对相似度 (Positive Pairs Computation)")
    print("=" * 80)
    t0 = time.time()

    # 按身份连续区间分块 (query_ids已按ID单调排序)
    diff = np.diff(query_ids)
    change_indices = np.where(diff > 0)[0] + 1
    starts = np.concatenate(([0], change_indices))
    ends = np.concatenate((change_indices, [len(query_ids)]))

    pos_sims_list = []
    pos_pairs_meta = [] # (sim, idx1, idx2)

    for s, e in zip(starts, ends):
        length = e - s
        if length <= 1:
            continue
        feats_sub = query_feats[s:e]
        sim_mat = np.dot(feats_sub, feats_sub.T)
        triu_r, triu_c = np.triu_indices(length, k=1)
        sim_vals = sim_mat[triu_r, triu_c]
        pos_sims_list.append(sim_vals)

        for r, c, val in zip(triu_r, triu_c, sim_vals):
            pos_pairs_meta.append((float(val), s + int(r), s + int(c)))

    all_pos_sims = np.concatenate(pos_sims_list).astype(np.float32)
    t_pos = time.time() - t0

    print(f"正样本对计算完成 (耗时: {t_pos:.3f}s):")
    print(f"  正样本对数量: {len(all_pos_sims):,}")
    print(f"  相似度 均值: {all_pos_sims.mean():.4f}, 标准差: {all_pos_sims.std():.4f}")
    print(f"  相似度 最小值: {all_pos_sims.min():.4f}, 最大值: {all_pos_sims.max():.4f}")
    print(f"  分位数 [1%, 5%, 50%, 95%, 99%]: {np.quantile(all_pos_sims, [0.01, 0.05, 0.50, 0.95, 0.99]).round(4)}")

    return all_pos_sims, pos_pairs_meta


def _gpu_worker(gpu_id, feats_shared, ids_shared, tasks, num_bins, total_bins, return_dict, extract_config=None):
    """
    GPU Worker 进程：
    在单张GPU上分块计算 GEMM，精确剔除同身份正样本，直接统计纯负样本直方图，同时提取高相似度负样本对。
    """
    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')

    # 加载数据到 GPU (提前转为FP16与CUDA tensor以加速后续GEMM)
    with torch.no_grad():
        feats_gpu = feats_shared.to(device=device, dtype=torch.float16, non_blocking=True)
        ids_gpu = ids_shared.to(device=device, dtype=torch.int64, non_blocking=True)

        neg_hist = torch.zeros(total_bins, dtype=torch.int64, device=device)
        scale = total_bins / 2.0  # 映射 [-1, 1] 到 [0, total_bins)

        extract_neg_threshold = extract_config.get('extract_neg_threshold', None) if extract_config else None
        max_extract_per_gpu = extract_config.get('max_extract_per_gpu', 500) if extract_config else 500
        extracted_neg_pairs = []

        for task in tasks:
            i_s, i_e, j_s, j_e, is_diag = task
            f_i = feats_gpu[i_s:i_e]
            f_j = feats_gpu[j_s:j_e]
            id_i = ids_gpu[i_s:i_e]
            id_j = ids_gpu[j_s:j_e]

            # 矩阵乘法计算余弦相似度
            sim_block = torch.mm(f_i, f_j.t())

            # 区分同ID与不同ID样本
            if is_diag:
                triu_mask = torch.triu(torch.ones(i_e - i_s, j_e - j_s, device=device, dtype=torch.bool), diagonal=1)
                diff_mask = (~(id_i.unsqueeze(1) == id_j.unsqueeze(0))) & triu_mask
                sim_vals = sim_block[diff_mask]
            elif id_i[-1] >= id_j[0]:
                diff_mask = ~(id_i.unsqueeze(1) == id_j.unsqueeze(0))
                sim_vals = sim_block[diff_mask]
            else:
                diff_mask = None
                sim_vals = sim_block.view(-1)

            # 提取高相似度负样本 (困难负样本)
            if extract_neg_threshold is not None and len(extracted_neg_pairs) < max_extract_per_gpu:
                high_mask = sim_block >= extract_neg_threshold
                if diff_mask is not None:
                    high_mask = high_mask & diff_mask
                if high_mask.any():
                    rows, cols = torch.where(high_mask)
                    r_np = rows.cpu().numpy()
                    c_np = cols.cpu().numpy()
                    for r, c in zip(r_np, c_np):
                        global_i = i_s + int(r)
                        global_j = j_s + int(c)
                        val = float(sim_block[r, c].item())
                        extracted_neg_pairs.append((val, global_i, global_j))
                        if len(extracted_neg_pairs) >= max_extract_per_gpu:
                            break

            # 直方图统计
            bin_idx = ((sim_vals.float() + 1.0) * scale).long().clamp_(0, total_bins - 1)
            neg_hist += torch.bincount(bin_idx, minlength=total_bins)

    return_dict[gpu_id] = {
        'neg_hist': neg_hist.cpu().numpy(),
        'extracted_neg_pairs': extracted_neg_pairs
    }


def compute_negative_pairs_multi_gpu(query_feats, query_ids, num_gpus=7, chunk_size=4096, total_bins=100000, extract_config=None):
    """
    多卡并行调度器：
    生成分块任务，根据计算量进行负载均衡分配，启动多进程并行计算负样本直方图。
    """
    print("\n" + "=" * 80)
    print(f"3. 多卡并行全量负样本对计算 (使用 {num_gpus} 张 GPU, 分块大小: {chunk_size}, 直方图Bins: {total_bins})")
    print("=" * 80)

    t0 = time.time()
    N = len(query_feats)
    num_chunks = (N + chunk_size - 1) // chunk_size

    # 生成所有分块任务 (bi, bj) 满足 bi <= bj
    all_tasks = []
    for bi in range(num_chunks):
        i_s = bi * chunk_size
        i_e = min((bi + 1) * chunk_size, N)
        for bj in range(bi, num_chunks):
            j_s = bj * chunk_size
            j_e = min((bj + 1) * chunk_size, N)
            is_diag = (bi == bj)
            weight = (i_e - i_s) * (j_e - j_s) * (0.5 if is_diag else 1.0)
            all_tasks.append((weight, (i_s, i_e, j_s, j_e, is_diag)))

    # 贪心负载均衡分配
    gpu_tasks = [[] for _ in range(num_gpus)]
    gpu_loads = [0.0] * num_gpus
    all_tasks.sort(key=lambda x: x[0], reverse=True)
    for weight, task in all_tasks:
        min_gpu = int(np.argmin(gpu_loads))
        gpu_tasks[min_gpu].append(task)
        gpu_loads[min_gpu] += weight

    print(f"任务规划完成: 总分块数: {len(all_tasks)}")
    for g in range(num_gpus):
        print(f"  GPU {g}: 分配 {len(gpu_tasks[g])} 个分块 (相对负载: {gpu_loads[g]/sum(gpu_loads)*100:.2f}%)")

    # 共享内存传递数据
    feats_shared = torch.from_numpy(query_feats).share_memory_()
    ids_shared = torch.from_numpy(query_ids).share_memory_()

    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    return_dict = manager.dict()

    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=_gpu_worker,
            args=(gpu_id, feats_shared, ids_shared, gpu_tasks[gpu_id], None, total_bins, return_dict, extract_config)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # 聚合所有GPU的直方图与提取样本
    total_neg_hist = np.zeros(total_bins, dtype=np.int64)
    all_extracted_neg = []
    for gpu_id in range(num_gpus):
        res = return_dict[gpu_id]
        total_neg_hist += res['neg_hist']
        all_extracted_neg.extend(res['extracted_neg_pairs'])

    t_calc = time.time() - t0
    total_neg_counted = int(total_neg_hist.sum())
    throughput = total_neg_counted / t_calc / 1e9
    print(f"负样本矩阵计算完成 (耗时: {t_calc:.2f}s, 计算吞吐率: {throughput:.3f} Giga-Pairs/sec):")
    print(f"  统计到的纯负样本对总数: {total_neg_counted:,}")

    return total_neg_hist, all_extracted_neg, t_calc


def evaluate_tpir_at_fpir(all_pos_sims, total_neg_hist, total_neg_pairs, total_bins=100000):
    """
    精确计算 TPIR @ 各 FPIR 关键评估点
    使用高精度直方图累积分布函数 (CDF) 确定阈值，结合正样本对排序精确定位 TPIR。
    """
    print("\n" + "=" * 80)
    print("4. TPIR @ FPIR 性能指标评估")
    print("=" * 80)

    scale = total_bins / 2.0
    bin_edges = np.linspace(-1.0, 1.0, total_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # 正样本直方图
    pos_bin_idx = np.clip(((all_pos_sims + 1.0) * scale).astype(np.int64), 0, total_bins - 1)
    pos_hist = np.bincount(pos_bin_idx, minlength=total_bins)

    total_pos = len(all_pos_sims)
    total_neg = int(total_neg_hist.sum())

    # 负样本从右向左累积分布 (sim >= threshold 的数量)
    neg_gt = np.cumsum(total_neg_hist[::-1])[::-1]
    fpir_curve = neg_gt / total_neg

    pos_gt = np.cumsum(pos_hist[::-1])[::-1]
    tpir_curve = pos_gt / total_pos

    target_fpirs = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    results = []

    print("\n" + "-" * 80)
    print(f"{'FPIR 目标':<12} | {'判定阈值 Threshold':<20} | {'实际 FPIR':<18} | {'TPIR 识别率':<16} | {'预期范围'}")
    print("-" * 80)

    sorted_pos_sims = np.sort(all_pos_sims)

    expected_ranges = {
        1e-2: "95% - 97%",
        1e-3: "90% - 93%",
        1e-4: "82% - 85%",
        1e-5: "60% - 65%"
    }

    for target_fpir in target_fpirs:
        # 在 fpir_curve 中定位阈值
        idx = np.searchsorted(-fpir_curve, -target_fpir, side='right')
        idx = min(idx, total_bins - 1)

        threshold = bin_edges[idx]
        actual_fpir = fpir_curve[idx]

        # 计算正样本中大于阈值的比例
        actual_tpir = (sorted_pos_sims > threshold).sum() / total_pos

        expected_str = expected_ranges.get(target_fpir, "-")
        results.append({
            'target_fpir': target_fpir,
            'threshold': threshold,
            'actual_fpir': actual_fpir,
            'tpir': actual_tpir,
            'expected': expected_str
        })
        print(f"{target_fpir:<12.0e} | {threshold:<20.4f} | {actual_fpir:<18.6e} | {actual_tpir*100:<15.2f}% | {expected_str}")
    print("-" * 80)

    return {
        'results': results,
        'bin_centers': bin_centers,
        'bin_edges': bin_edges,
        'pos_hist': pos_hist,
        'neg_hist': total_neg_hist,
        'fpir_curve': fpir_curve,
        'tpir_curve': tpir_curve,
        'total_pos': total_pos,
        'total_neg': total_neg
    }


def extract_and_analyze_extreme_pairs(dataset, pos_pairs_meta, all_extracted_neg, eval_data, pos_threshold=0.20, neg_threshold=0.50, top_k=10):
    """
    提取并展示异常/困难样本对：
    1. 困难正样本对 (同一身份但相似度极低，属于 False Negatives 风险样本)
    2. 困难负样本对 (不同身份但相似度极高，属于 False Positives 风险样本)
    """
    print("\n" + "=" * 80)
    print("5. 困难样本对提取与错误分析 (Hard Pairs & Error Analysis)")
    print("=" * 80)

    file_paths = dataset['file_paths']
    query_ids = dataset['query_ids']

    # 1. 提取相似度最低的正样本对
    pos_pairs_meta.sort(key=lambda x: x[0])
    hard_pos_pairs = [p for p in pos_pairs_meta if p[0] < pos_threshold][:top_k]

    print(f"\n[困难正样本对 (同一身份但相似度 < {pos_threshold})] 相似度最低前 {len(hard_pos_pairs)} 对:")
    print("-" * 80)
    for rank, (sim, idx1, idx2) in enumerate(hard_pos_pairs, 1):
        print(f"#{rank:02d} 相似度: {sim:+.4f} | ID: {query_ids[idx1]}")
        print(f"    图片1: {file_paths[idx1]}")
        print(f"    图片2: {file_paths[idx2]}")

    # 2. 提取相似度最高的负样本对
    hard_neg_pairs = []
    for sim, idx1, idx2 in all_extracted_neg:
        if query_ids[idx1] != query_ids[idx2] and sim >= neg_threshold:
            hard_neg_pairs.append((sim, idx1, idx2))
    hard_neg_pairs.sort(key=lambda x: x[0], reverse=True)
    hard_neg_pairs = hard_neg_pairs[:top_k]

    print(f"\n[困难负样本对 (不同身份但相似度 >= {neg_threshold})] 相似度最高前 {len(hard_neg_pairs)} 对:")
    print("-" * 80)
    for rank, (sim, idx1, idx2) in enumerate(hard_neg_pairs, 1):
        print(f"#{rank:02d} 相似度: {sim:+.4f} | ID1: {query_ids[idx1]} vs ID2: {query_ids[idx2]}")
        print(f"    图片1: {file_paths[idx1]}")
        print(f"    图片2: {file_paths[idx2]}")

    return {
        'hard_pos_pairs': hard_pos_pairs,
        'hard_neg_pairs': hard_neg_pairs
    }


def plot_charts(eval_data, output_path='evaluation_results.png', font_path='font/SourceHanSansSC-Normal.otf'):
    """
    绘制高质量评估图表：
    - 图1: 正负样本对余弦相似度分布图 (PDF)
    - 图2: TPIR @ FPIR ROC 评估曲线 (对数坐标)
    """
    print("\n" + "=" * 80)
    print(f"6. 绘制高清晰度评估图表: {output_path}")
    print("=" * 80)

    # 加载指定中文字体
    custom_font = None
    if os.path.exists(font_path):
        custom_font = fm.FontProperties(fname=font_path)
        plt.rcParams['font.sans-serif'] = [custom_font.get_name(), 'DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        font_kw = {'fontproperties': custom_font}
    else:
        font_kw = {}

    bin_centers = eval_data['bin_centers']
    pos_hist = eval_data['pos_hist']
    neg_hist = eval_data['neg_hist']
    total_pos = eval_data['total_pos']
    total_neg = eval_data['total_neg']
    fpir_curve = eval_data['fpir_curve']
    tpir_curve = eval_data['tpir_curve']
    results = eval_data['results']

    # 归一化为概率密度
    bin_width = bin_centers[1] - bin_centers[0]
    pos_pdf = pos_hist / (total_pos * bin_width)
    neg_pdf = neg_hist / (total_neg * bin_width)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    # 1. 相似度分布图
    ax1 = axes[0]
    ax1.plot(bin_centers, neg_pdf, label='负样本对分布 (Negative Pairs)', color='#1f77b4', linewidth=2.0)
    ax1.plot(bin_centers, pos_pdf, label='正样本对分布 (Positive Pairs)', color='#d62728', linewidth=2.0)
    ax1.fill_between(bin_centers, neg_pdf, color='#1f77b4', alpha=0.25)
    ax1.fill_between(bin_centers, pos_pdf, color='#d62728', alpha=0.25)

    ax1.set_xlim(-0.4, 1.0)
    ax1.set_xlabel('余弦相似度 (Cosine Similarity)', fontsize=12, **font_kw)
    ax1.set_ylabel('概率密度 (Probability Density)', fontsize=12, **font_kw)
    ax1.set_title('正负样本对相似度分布图 (Similarity Distributions)', fontsize=14, fontweight='bold', **font_kw)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right', frameon=True, prop=custom_font)

    # 标出典型评估阈值
    for r in results:
        if r['target_fpir'] in [1e-2, 1e-4]:
            th = r['threshold']
            ax1.axvline(th, color='darkgreen', linestyle=':', alpha=0.8, linewidth=1.5)
            ax1.text(th, ax1.get_ylim()[1] * 0.65, f" FPIR={r['target_fpir']:.0e}\n (Th={th:.3f})",
                     fontsize=9, color='darkgreen', **font_kw)

    # 2. TPIR @ FPIR ROC 曲线
    ax2 = axes[1]
    valid_mask = (fpir_curve >= 1e-7) & (fpir_curve <= 1.0)
    ax2.plot(fpir_curve[valid_mask], tpir_curve[valid_mask] * 100, color='#2ca02c', linewidth=2.5, label='TPIR @ FPIR 特征曲线')

    colors = ['#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#ff7f0e']
    for i, r in enumerate(results):
        fp = r['actual_fpir']
        tp = r['tpir'] * 100
        th = r['threshold']
        if fp >= 1e-7:
            ax2.scatter([fp], [tp], color=colors[i % len(colors)], s=60, zorder=5)
            offset_y = -7.0 if tp > 30 else 5.0
            ax2.annotate(
                f"FPIR={r['target_fpir']:.0e}\nTPIR={tp:.2f}%\nTh={th:.3f}",
                xy=(fp, tp),
                xytext=(fp * 2.2, tp + offset_y),
                arrowprops=dict(arrowstyle="->", color=colors[i % len(colors)], lw=1.2),
                fontsize=9,
                fontproperties=custom_font,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=colors[i % len(colors)], alpha=0.9)
            )

    ax2.set_xscale('log')
    ax2.set_xlim(1e-6, 1.0)
    ax2.set_ylim(0, 105)
    ax2.set_xlabel('错误接受率 FPIR (False Positive Identification Rate)', fontsize=12, **font_kw)
    ax2.set_ylabel('正确识别率 TPIR (%) (True Positive Identification Rate)', fontsize=12, **font_kw)
    ax2.set_title('TPIR @ FPIR 评估特征曲线 (ROC Curve)', fontsize=14, fontweight='bold', **font_kw)
    ax2.grid(True, which='both', linestyle='--', alpha=0.5)
    ax2.legend(loc='lower right', frameon=True, prop=custom_font)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"评估图表已成功保存至: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="大规模人脸特征相似度评估与分析系统")
    parser.add_argument('--data_path', type=str, default='s4_0618_enhance.pkl', help='特征pkl文件路径')
    parser.add_argument('--num_gpus', type=int, default=7, help='使用的GPU数量 (1~7)')
    parser.add_argument('--chunk_size', type=int, default=4096, help='分块大小')
    parser.add_argument('--num_bins', type=int, default=100000, help='直方图分箱数量')
    parser.add_argument('--pos_thresh', type=float, default=0.20, help='提取困难正样本阈值')
    parser.add_argument('--neg_thresh', type=float, default=0.50, help='提取困难负样本阈值')
    parser.add_argument('--output_chart', type=str, default='evaluation_results.png', help='输出图表路径')
    parser.add_argument('--font_path', type=str, default='font/SourceHanSansSC-Normal.otf', help='中文字体路径')
    args = parser.parse_args()

    t_start = time.time()

    # 1. 加载数据集
    dataset = load_and_analyze_dataset(args.data_path)

    # 2. 计算正样本对相似度
    all_pos_sims, pos_pairs_meta = compute_positive_pairs(
        dataset['query_feats'],
        dataset['query_ids']
    )

    # 3. 多卡并行计算负样本对
    extract_config = {
        'extract_neg_threshold': args.neg_thresh,
        'max_extract_per_gpu': 500
    }
    total_neg_hist, all_extracted_neg, t_calc = compute_negative_pairs_multi_gpu(
        dataset['query_feats'],
        dataset['query_ids'],
        num_gpus=args.num_gpus,
        chunk_size=args.chunk_size,
        total_bins=args.num_bins,
        extract_config=extract_config
    )

    # 4. TPIR @ FPIR 指标评估
    eval_data = evaluate_tpir_at_fpir(
        all_pos_sims,
        total_neg_hist,
        dataset['total_neg_pairs'],
        total_bins=args.num_bins
    )

    # 5. 提取异常困难样本对
    extract_and_analyze_extreme_pairs(
        dataset,
        pos_pairs_meta,
        all_extracted_neg,
        eval_data,
        pos_threshold=args.pos_thresh,
        neg_threshold=args.neg_thresh,
        top_k=10
    )

    # 6. 绘制图表
    plot_charts(
        eval_data,
        output_path=args.output_chart,
        font_path=args.font_path
    )

    t_total = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"评估完成! 总耗时: {t_total:.2f} 秒 (206.5亿负样本对计算耗时: {t_calc:.2f} 秒, 吞吐率: {dataset['total_neg_pairs']/t_calc/1e9:.3f} Giga-Pairs/s)")
    print("=" * 80)


if __name__ == '__main__':
    main()
