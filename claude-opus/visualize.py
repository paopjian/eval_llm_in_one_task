#!/usr/bin/env python3
"""
相似度分布可视化脚本
基于已有的相似度数据绘制分布图
"""
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 配置中文字体（仓库内相对路径，兼容外部克隆）
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
font_path = os.path.join(_REPO_ROOT, 'font', 'SourceHanSansSC-Normal.otf')
font_prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False

def plot_similarity_distribution(pos_sims, neg_sims, output_path='similarity_distribution.png'):
    """绘制相似度分布图"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 正样本分布
    ax1 = axes[0, 0]
    ax1.hist(pos_sims, bins=100, alpha=0.7, color='#2E7D32', edgecolor='black', linewidth=0.5)
    ax1.set_title('正样本相似度分布', fontproperties=font_prop, fontsize=16, fontweight='bold')
    ax1.set_xlabel('相似度', fontproperties=font_prop, fontsize=13)
    ax1.set_ylabel('样本对数量', fontproperties=font_prop, fontsize=13)
    ax1.axvline(pos_sims.mean(), color='red', linestyle='--', linewidth=2, label=f'均值={pos_sims.mean():.4f}')
    ax1.legend(prop=font_prop, fontsize=11)
    ax1.grid(True, alpha=0.3)

    # 2. 负样本分布（采样）
    ax2 = axes[0, 1]
    sample_size = min(1000000, len(neg_sims))
    neg_sample = np.random.choice(neg_sims, size=sample_size, replace=False)
    ax2.hist(neg_sample, bins=100, alpha=0.7, color='#C62828', edgecolor='black', linewidth=0.5)
    ax2.set_title('负样本相似度分布（采样100万）', fontproperties=font_prop, fontsize=16, fontweight='bold')
    ax2.set_xlabel('相似度', fontproperties=font_prop, fontsize=13)
    ax2.set_ylabel('样本对数量', fontproperties=font_prop, fontsize=13)
    ax2.axvline(neg_sims.mean(), color='blue', linestyle='--', linewidth=2, label=f'均值={neg_sims.mean():.4f}')
    ax2.legend(prop=font_prop, fontsize=11)
    ax2.grid(True, alpha=0.3)

    # 3. 正负样本对比（重叠分布）
    ax3 = axes[1, 0]
    ax3.hist(pos_sims, bins=100, alpha=0.5, color='green', label='正样本', density=True)
    ax3.hist(neg_sample, bins=100, alpha=0.5, color='red', label='负样本（采样）', density=True)
    ax3.set_title('正负样本相似度对比', fontproperties=font_prop, fontsize=16, fontweight='bold')
    ax3.set_xlabel('相似度', fontproperties=font_prop, fontsize=13)
    ax3.set_ylabel('密度', fontproperties=font_prop, fontsize=13)
    ax3.legend(prop=font_prop, fontsize=11)
    ax3.grid(True, alpha=0.3)

    # 4. TPIR-FPIR曲线
    ax4 = axes[1, 1]
    fpir_values = np.logspace(-5, -1, 50)
    tpir_values = []

    for fpir in fpir_values:
        percentile = (1 - fpir) * 100
        threshold = np.percentile(neg_sims, percentile)
        tpir = (pos_sims > threshold).sum() / len(pos_sims)
        tpir_values.append(tpir)

    ax4.semilogx(fpir_values, tpir_values, linewidth=2.5, color='#1976D2')
    ax4.set_title('TPIR @ FPIR 曲线', fontproperties=font_prop, fontsize=16, fontweight='bold')
    ax4.set_xlabel('FPIR (False Positive Identification Rate)', fontsize=13)
    ax4.set_ylabel('TPIR (True Positive Identification Rate)', fontsize=13)
    ax4.grid(True, alpha=0.3, which='both')

    # 标注关键点
    key_fpirs = [1e-5, 1e-4, 1e-3, 1e-2]
    for fpir in key_fpirs:
        percentile = (1 - fpir) * 100
        threshold = np.percentile(neg_sims, percentile)
        tpir = (pos_sims > threshold).sum() / len(pos_sims)
        ax4.plot(fpir, tpir, 'ro', markersize=8)
        ax4.text(fpir, tpir + 0.02, f'{tpir*100:.1f}%', fontsize=10, ha='center')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\n可视化图表已保存到: {output_path}")
    plt.close()

def load_results_from_pickle(pickle_path):
    """从pickle文件加载结果"""
    print(f"正在加载结果文件: {pickle_path}")
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return data

def main():
    import sys

    if len(sys.argv) > 1:
        # 从pickle文件加载
        pickle_path = sys.argv[1]
        pos_sims, neg_sims = load_results_from_pickle(pickle_path)
    else:
        # 重新计算（较慢）
        print("提示：可以先运行eval_final.py生成结果，然后传入pickle路径")
        print("用法: python visualize.py results.pkl")
        print("\n没有提供pickle文件，将重新计算...")

        from eval_final import load_data, compute_similarity_single_gpu
        feats, ids, paths = load_data()
        pos_sims, neg_sims = compute_similarity_single_gpu(feats, ids)

    print(f"\n数据统计:")
    print(f"  正样本数: {len(pos_sims):,}")
    print(f"  负样本数: {len(neg_sims):,}")

    # 绘制可视化
    plot_similarity_distribution(pos_sims, neg_sims)

if __name__ == '__main__':
    main()
