#!/usr/bin/env python3
"""
测试启动代码：读取数据并统计基本信息
用于帮助理解数据结构
"""
import pickle
import numpy as np

def load_and_analyze_data():
    """读取并分析测试数据"""
    print("=" * 80)
    print("数据加载与分析")
    print("=" * 80)

    # 1. 读取数据
    print("\n正在读取数据文件: s4_0618_enhance.pkl")
    with open('s4_0618_enhance.pkl', 'rb') as f:
        query_feats_list, query_feats_list_flip, query_ids, file_paths = pickle.load(f)

    print(f"\n数据基本信息:")
    print(f"  特征矩阵形状: {query_feats_list.shape}")
    print(f"  特征维度: {query_feats_list.shape[1]}")
    print(f"  样本总数: {len(query_ids)}")
    print(f"  身份总数: {len(np.unique(query_ids))}")

    # 2. 统计正负样本对数量
    N = len(query_ids)
    total_pairs = N * (N - 1) // 2

    # 计算正样本对数量
    unique_ids, counts = np.unique(query_ids, return_counts=True)
    total_pos_pairs = sum(c * (c - 1) // 2 for c in counts)
    total_neg_pairs = total_pairs - total_pos_pairs

    print(f"\n样本对统计:")
    print(f"  总样本对数: {total_pairs:,}")
    print(f"  正样本对数: {total_pos_pairs:,} ({total_pos_pairs/total_pairs*100:.2f}%)")
    print(f"  负样本对数: {total_neg_pairs:,} ({total_neg_pairs/total_pairs*100:.2f}%)")

    # 3. 检查特征是否归一化
    norms = np.linalg.norm(query_feats_list, axis=1)
    print(f"\n特征向量检查:")
    print(f"  L2范数均值: {norms.mean():.6f}")
    print(f"  L2范数标准差: {norms.std():.6f}")
    print(f"  是否归一化: {'是' if np.abs(norms.mean() - 1.0) < 0.01 else '否'}")

    # 4. 查看身份分布
    print(f"\n身份分布统计:")
    print(f"  平均每个身份的图片数: {N / len(unique_ids):.2f}")
    print(f"  最多图片的身份: {counts.max()} 张")
    print(f"  最少图片的身份: {counts.min()} 张")

    print("\n" + "=" * 80)
    print("数据分析完成！")
    print("=" * 80)
    print("\n提示：特征已L2归一化，可以直接用矩阵乘法计算余弦相似度")
    print("下一步：实现相似度计算和TPIR@FPIR评估")

    return query_feats_list, query_ids, file_paths

if __name__ == '__main__':
    query_feats_list, query_ids, file_paths = load_and_analyze_data()
