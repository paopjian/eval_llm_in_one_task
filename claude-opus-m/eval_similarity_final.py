#!/usr/bin/env python3
"""
人脸特征相似度评估系统 - 完整版
功能：
1. 多GPU并行计算相似度
2. 计算TPIR@FPIR指标
3. 绘制相似度分布图和TPIR@FPIR曲线
4. 支持提取above/below阈值的样本对
"""
import pickle
import numpy as np
import torch
import torch.multiprocessing as mp
import time
import argparse
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager

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
    """处理指定行范围的相似度计算"""
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
            feat_i = feat_tensor[i:i_end]
            ids_i = query_ids_tensor[i:i_end]

            # 第1部分：处理块内的上三角
            for i_local in range(i_end - i):
                global_i = i + i_local
                feat_i_single = feat_tensor[global_i:global_i+1]
                id_i_single = query_ids_tensor[global_i:global_i+1]

                if i_local + 1 < i_end - i:
                    j_local_start = i_local + 1
                    feat_j_inblock = feat_tensor[i+j_local_start:i_end]
                    ids_j_inblock = query_ids_tensor[i+j_local_start:i_end]

                    sim_block = torch.mm(feat_i_single, feat_j_inblock.t())
                    is_positive = (id_i_single[:, None] == ids_j_inblock[None, :])

                    sim_block_cpu = sim_block.cpu().numpy().flatten()
                    is_positive_cpu = is_positive.cpu().numpy().flatten()

                    bin_indices = ((sim_block_cpu + 1) / 2 * bins_count).astype(np.int32)
                    bin_indices = np.clip(bin_indices, 0, bins_count - 1)

                    np.add.at(pos_hist, bin_indices[is_positive_cpu], 1)
                    np.add.at(neg_hist, bin_indices[~is_positive_cpu], 1)

            # 第2部分：处理块外的列
            for j in range(i_end, N, chunk_size):
                j_end = min(j + chunk_size, N)
                feat_j = feat_tensor[j:j_end]
                ids_j = query_ids_tensor[j:j_end]

                sim_block = torch.mm(feat_i, feat_j.t())
                is_positive = (ids_i[:, None] == ids_j[None, :])

                sim_block_cpu = sim_block.cpu().numpy().flatten()
                is_positive_cpu = is_positive.cpu().numpy().flatten()

                bin_indices = ((sim_block_cpu + 1) / 2 * bins_count).astype(np.int32)
                bin_indices = np.clip(bin_indices, 0, bins_count - 1)

                np.add.at(pos_hist, bin_indices[is_positive_cpu], 1)
                np.add.at(neg_hist, bin_indices[~is_positive_cpu], 1)

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

def compute_similarity_multigpu(features, query_ids, devices, chunk_size=2000):
    """多GPU并行计算相似度直方图"""
    print("\n" + "=" * 80)
    print(f"计算相似度直方图 (使用 {len(devices)} 个GPU)")
    print("=" * 80)

    N = len(features)
    num_gpus = len(devices)

    rows_per_gpu = N // num_gpus
    row_ranges = []
    for i in range(num_gpus):
        start = i * rows_per_gpu
        end = (i + 1) * rows_per_gpu if i < num_gpus - 1 else N
        row_ranges.append((start, end))
        print(f"GPU {devices[i]}: 行 {start} - {end} ({end - start} 行)")

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

    for p in processes:
        p.join()

    elapsed = time.time() - start_time

    pos_hist = np.zeros(10000, dtype=np.int64)
    neg_hist = np.zeros(10000, dtype=np.int64)

    for rank in range(num_gpus):
        if return_dict[rank]['status'] == 'success':
            pos_hist += return_dict[rank]['pos_hist']
            neg_hist += return_dict[rank]['neg_hist']
            print(f"GPU {rank}: 正样本={return_dict[rank]['pos_hist'].sum():,}, "
                  f"负样本={return_dict[rank]['neg_hist'].sum():,}")
        else:
            print(f"GPU {rank} 错误: {return_dict[rank]['error']}")

    print(f"\n计算完成，耗时: {elapsed:.2f}秒")
    print(f"正样本对数总计: {pos_hist.sum():,}")
    print(f"负样本对数总计: {neg_hist.sum():,}")

    return pos_hist, neg_hist

def compute_tpir_fpir_curve(pos_hist, neg_hist, num_points=1000):
    """计算完整的TPIR@FPIR曲线"""
    total_pos = pos_hist.sum()
    total_neg = neg_hist.sum()

    # 计算累积分布（从高到低）
    neg_cumsum = np.cumsum(neg_hist[::-1])[::-1]
    pos_cumsum = np.cumsum(pos_hist[::-1])[::-1]

    # 计算FPIR和TPIR
    fpir_curve = neg_cumsum / total_neg
    tpir_curve = pos_cumsum / total_pos

    # 均匀采样
    indices = np.linspace(0, len(fpir_curve)-1, num_points).astype(int)
    fpir_curve = fpir_curve[indices]
    tpir_curve = tpir_curve[indices]

    return fpir_curve, tpir_curve

def compute_tpir_at_fpir(pos_hist, neg_hist, fpir_thresholds=[1e-5, 1e-4, 1e-3, 1e-2]):
    """计算特定FPIR点的TPIR值"""
    print("\n" + "=" * 80)
    print("计算TPIR@FPIR")
    print("=" * 80)

    total_pos = pos_hist.sum()
    total_neg = neg_hist.sum()
    bins = np.linspace(-1, 1, 10001)

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

    return results

def plot_results(pos_hist, neg_hist, output_file='evaluation_results.png'):
    """绘制相似度分布图和TPIR@FPIR曲线"""
    print("\n" + "=" * 80)
    print("绘制评估图表")
    print("=" * 80)

    # 设置中文字体
    font_path = 'font/SourceHanSansSC-Normal.otf'
    font_prop = font_manager.FontProperties(fname=font_path)
    # 注册字体
    font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
    matplotlib.rcParams['axes.unicode_minus'] = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 图1：相似度分布
    bins = np.linspace(-1, 1, 10001)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # 归一化为概率密度
    pos_density = pos_hist / pos_hist.sum()
    neg_density = neg_hist / neg_hist.sum()

    ax1.plot(bin_centers, pos_density, label='正样本对', linewidth=2, color='#38BDF8')
    ax1.plot(bin_centers, neg_density, label='负样本对', linewidth=2, color='#E9A568')
    ax1.set_xlabel('余弦相似度', fontproperties=font_prop, fontsize=14)
    ax1.set_ylabel('概率密度', fontproperties=font_prop, fontsize=14)
    ax1.set_title('相似度分布', fontproperties=font_prop, fontsize=16, fontweight='bold')
    ax1.legend(prop=font_prop, fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-0.2, 1.0])

    # 图2：TPIR@FPIR曲线
    fpir_curve, tpir_curve = compute_tpir_fpir_curve(pos_hist, neg_hist)

    ax2.plot(fpir_curve, tpir_curve, linewidth=2, color='#6EE7B7')
    ax2.set_xlabel('FPIR (False Positive Identification Rate)', fontsize=14)
    ax2.set_ylabel('TPIR (True Positive Identification Rate)', fontsize=14)
    ax2.set_title('TPIR @ FPIR 曲线', fontproperties=font_prop, fontsize=16, fontweight='bold')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_xlim([1e-6, 1])
    ax2.set_ylim([0, 1])

    # 标记关键点
    key_fpirs = [1e-5, 1e-4, 1e-3, 1e-2]
    for fpir_target in key_fpirs:
        idx = np.argmin(np.abs(fpir_curve - fpir_target))
        ax2.plot(fpir_curve[idx], tpir_curve[idx], 'ro', markersize=8)
        ax2.text(fpir_curve[idx], tpir_curve[idx] + 0.03,
                f'TPIR={tpir_curve[idx]*100:.1f}%\n@FPIR={fpir_target:.0e}',
                fontsize=9, ha='center')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"图表已保存到: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='人脸特征相似度评估系统')
    parser.add_argument('--devices', type=str, default='0,1,2,3,4,5,6', help='GPU设备列表')
    parser.add_argument('--chunk_size', type=int, default=2000, help='分块大小')
    parser.add_argument('--output', type=str, default='evaluation_results.png', help='输出图片路径')
    args = parser.parse_args()

    devices = [f'cuda:{i}' for i in args.devices.split(',')]

    # 加载数据
    features, query_ids, file_paths = load_data('s4_0618_enhance.pkl')

    # 多GPU并行计算
    pos_hist, neg_hist = compute_similarity_multigpu(
        features, query_ids, devices, chunk_size=args.chunk_size
    )

    # 计算TPIR@FPIR
    results = compute_tpir_at_fpir(pos_hist, neg_hist)

    # 绘制图表
    plot_results(pos_hist, neg_hist, output_file=args.output)

    print("\n" + "=" * 80)
    print("评估完成！")
    print("=" * 80)

if __name__ == '__main__':
    main()
