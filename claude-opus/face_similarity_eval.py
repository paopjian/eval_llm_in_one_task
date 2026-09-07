#!/usr/bin/env python3
"""
最终完整版：基于单卡优化，添加所有功能
- 单卡高效计算（避免多卡通信开销）
- 二分查找计算TPIR@FPIR
- 支持样本对提取
- 绘制可视化图表
- 支持命令行参数
"""
import pickle
import numpy as np
import torch
import time
import argparse
import sys

def load_data(data_path='s4_0618_enhance.pkl'):
    """加载特征数据"""
    print("正在加载数据...")
    with open(data_path, 'rb') as f:
        query_feats_list, _, query_ids, file_paths = pickle.load(f)
    print(f"数据加载完成: {len(query_ids)} 个样本")
    return query_feats_list, query_ids, file_paths

def compute_similarity_single_gpu(feats, ids, chunk_size=5000, device='cuda:0'):
    """
    单GPU分块计算相似度矩阵
    返回正负样本相似度
    """
    N = len(feats)
    feats_tensor = torch.from_numpy(feats).to(device)
    ids_tensor = torch.from_numpy(ids).to(device)

    pos_similarities = []
    neg_similarities = []

    total_pairs = N * (N - 1) // 2
    processed_pairs = 0

    print(f"\n开始计算相似度矩阵 (设备: {device}, 分块大小: {chunk_size})")
    print(f"总样本对数: {total_pairs:,}")
    start_time = time.time()

    for i in range(0, N, chunk_size):
        i_end = min(i + chunk_size, N)
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

            if i == j:
                processed_pairs += mask.sum().item()
            else:
                processed_pairs += sim_values.numel()

            # 每处理一定数量打印进度
            if processed_pairs % 100_000_000 == 0 or processed_pairs > total_pairs * 0.9:
                elapsed = time.time() - start_time
                progress = processed_pairs / total_pairs * 100
                speed = processed_pairs / elapsed
                eta = (total_pairs - processed_pairs) / speed
                print(f"  进度: {progress:.2f}% | "
                      f"已处理: {processed_pairs:,} 对 | "
                      f"速度: {speed:,.0f} 对/秒 | "
                      f"预计剩余: {eta:.1f}秒")

    elapsed = time.time() - start_time
    print(f"\n相似度计算完成！")
    print(f"  总耗时: {elapsed:.2f} 秒")
    print(f"  平均速度: {total_pairs/elapsed:,.0f} 对/秒")

    pos_similarities = np.concatenate(pos_similarities)
    neg_similarities = np.concatenate(neg_similarities)

    return pos_similarities, neg_similarities

def find_threshold_for_fpir(neg_sims, fpir):
    """使用二分查找找到给定FPIR对应的阈值"""
    target_count = int(fpir * len(neg_sims))

    if target_count == 0:
        return neg_sims.max()
    if target_count >= len(neg_sims):
        return neg_sims.min()

    low, high = neg_sims.min(), neg_sims.max()

    for _ in range(50):
        mid = (low + high) / 2
        count = (neg_sims > mid).sum()

        if count == target_count:
            return mid
        elif count < target_count:
            high = mid
        else:
            low = mid

        if high - low < 1e-6:
            break

    return (low + high) / 2

def compute_tpir_at_fpir(pos_sims, neg_sims, fpir_targets=[1e-5, 1e-4, 1e-3, 1e-2]):
    """计算TPIR@FPIR指标"""
    print(f"\n开始计算TPIR@FPIR指标...")
    print(f"  正样本数: {len(pos_sims):,}")
    print(f"  负样本数: {len(neg_sims):,}")

    results = {}
    print("\n" + "="*60)
    print("TPIR @ FPIR 评估结果")
    print("="*60)

    for fpir in fpir_targets:
        start_time = time.time()
        threshold = find_threshold_for_fpir(neg_sims, fpir)
        tpir = (pos_sims > threshold).sum() / len(pos_sims)
        elapsed = time.time() - start_time

        results[fpir] = {'threshold': threshold, 'tpir': tpir}
        print(f"TPIR @ FPIR={fpir:.0e}: {tpir*100:.2f}% (阈值={threshold:.4f}, 耗时={elapsed:.2f}秒)")

    print("="*60)
    return results

def extract_pairs_above_threshold(pos_sims, neg_sims, threshold, max_pairs=100):
    """提取高于阈值的样本对（用于错误分析）"""
    print(f"\n提取相似度 > {threshold:.4f} 的样本对...")

    pos_above = pos_sims > threshold
    neg_above = neg_sims > threshold

    pos_count = pos_above.sum()
    neg_count = neg_above.sum()

    print(f"  正样本中高于阈值: {pos_count:,} 对 ({pos_count/len(pos_sims)*100:.2f}%)")
    print(f"  负样本中高于阈值: {neg_count:,} 对 ({neg_count/len(neg_sims)*100:.2f}%)")

    return {
        'pos_above_count': int(pos_count),
        'neg_above_count': int(neg_count),
        'pos_above_ratio': float(pos_count / len(pos_sims)),
        'neg_above_ratio': float(neg_count / len(neg_sims))
    }

def save_distribution_plot(pos_sims, neg_sims, output_path='similarity_distribution.png'):
    """绘制相似度分布图"""
    try:
        import matplotlib
        matplotlib.use('Agg')  # 非交互式后端
        import matplotlib.pyplot as plt

        # 配置中文字体
        plt.rcParams['font.sans-serif'] = ['Source Han Sans SC']
        plt.rcParams['axes.unicode_minus'] = False

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 正样本分布
        ax1.hist(pos_sims, bins=100, alpha=0.7, color='green', edgecolor='black')
        ax1.set_title('正样本相似度分布', fontsize=14)
        ax1.set_xlabel('相似度', fontsize=12)
        ax1.set_ylabel('样本对数量', fontsize=12)
        ax1.grid(True, alpha=0.3)

        # 负样本分布（采样绘制，否则太多）
        sample_size = min(1000000, len(neg_sims))
        neg_sample = np.random.choice(neg_sims, size=sample_size, replace=False)
        ax2.hist(neg_sample, bins=100, alpha=0.7, color='red', edgecolor='black')
        ax2.set_title('负样本相似度分布', fontsize=14)
        ax2.set_xlabel('相似度', fontsize=12)
        ax2.set_ylabel('样本对数量', fontsize=12)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"\n相似度分布图已保存到: {output_path}")
        plt.close()
    except Exception as e:
        print(f"\n警告：无法绘制分布图: {e}")

def main():
    parser = argparse.ArgumentParser(description='人脸特征相似度评估系统')
    parser.add_argument('--data', default='s4_0618_enhance.pkl', help='特征数据文件路径')
    parser.add_argument('--device', default='cuda:0', help='使用的GPU设备')
    parser.add_argument('--chunk-size', type=int, default=5000, help='分块大小')
    parser.add_argument('--fpir', nargs='+', type=float, default=[1e-5, 1e-4, 1e-3, 1e-2],
                       help='FPIR目标值列表')
    parser.add_argument('--extract-threshold', type=float, help='提取样本对的阈值')
    parser.add_argument('--plot', action='store_true', help='绘制分布图')
    parser.add_argument('--plot-output', default='similarity_distribution.png', help='分布图输出路径')

    args = parser.parse_args()

    total_start = time.time()

    print("="*80)
    print("人脸特征相似度评估系统 - 最终完整版")
    print("="*80)

    # 1. 加载数据
    feats, ids, paths = load_data(args.data)

    # 2. 计算相似度
    pos_sims, neg_sims = compute_similarity_single_gpu(feats, ids, args.chunk_size, args.device)

    print(f"\n相似度统计:")
    print(f"  正样本 - 均值: {pos_sims.mean():.4f}, 标准差: {pos_sims.std():.4f}, "
          f"最小值: {pos_sims.min():.4f}, 最大值: {pos_sims.max():.4f}")
    print(f"  负样本 - 均值: {neg_sims.mean():.4f}, 标准差: {neg_sims.std():.4f}, "
          f"最小值: {neg_sims.min():.4f}, 最大值: {neg_sims.max():.4f}")

    # 3. 计算TPIR@FPIR
    results = compute_tpir_at_fpir(pos_sims, neg_sims, args.fpir)

    # 4. 提取样本对（如果指定了阈值）
    if args.extract_threshold is not None:
        extract_pairs_above_threshold(pos_sims, neg_sims, args.extract_threshold)

    # 5. 绘制分布图（如果指定）
    if args.plot:
        save_distribution_plot(pos_sims, neg_sims, args.plot_output)

    total_time = time.time() - total_start
    print(f"\n" + "="*60)
    print(f"任务全部完成！总耗时: {total_time:.2f} 秒")
    print("="*60)

    return results, pos_sims, neg_sims

if __name__ == '__main__':
    main()
