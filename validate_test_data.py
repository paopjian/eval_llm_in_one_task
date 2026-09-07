#!/usr/bin/env python3
"""
快速验证测试数据的质量和可用性

功能：
1. 加载测试数据
2. 验证数据格式和完整性
3. 计算基本统计信息
4. 估算计算复杂度
5. 快速采样验证评估指标

使用方法：
    python validate_test_data.py test_data_medium.pkl
    python validate_test_data.py --file test_data_large.pkl --sample 1000
"""

import pickle
import numpy as np
import argparse
import time
from pathlib import Path


def load_data(file_path):
    """加载测试数据"""
    print(f"加载数据: {file_path}")
    start_time = time.time()

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    load_time = time.time() - start_time

    if len(data) == 4:
        features, _, labels, paths = data
    else:
        raise ValueError(f"数据格式错误：期望4个元素，得到{len(data)}个")

    print(f"  加载耗时: {load_time:.2f}秒")
    return features, labels, paths


def validate_format(features, labels, paths):
    """验证数据格式"""
    print(f"\n{'='*80}")
    print("数据格式验证")
    print(f"{'='*80}")

    # 检查类型
    print(f"\n数据类型:")
    print(f"  features: {type(features)} {features.dtype if hasattr(features, 'dtype') else ''}")
    print(f"  labels: {type(labels)} {labels.dtype if hasattr(labels, 'dtype') else ''}")
    print(f"  paths: {type(paths)}")

    # 检查形状
    print(f"\n数据形状:")
    print(f"  features: {features.shape}")
    print(f"  labels: {labels.shape}")
    print(f"  paths: {len(paths)} 个路径")

    # 验证一致性
    n_samples = len(labels)
    errors = []

    if features.shape[0] != n_samples:
        errors.append(f"特征数量({features.shape[0]})与标签数量({n_samples})不匹配")

    if len(paths) != n_samples:
        errors.append(f"路径数量({len(paths)})与标签数量({n_samples})不匹配")

    if features.ndim != 2:
        errors.append(f"特征维度错误：期望2维，得到{features.ndim}维")

    if labels.ndim != 1:
        errors.append(f"标签维度错误：期望1维，得到{labels.ndim}维")

    if len(errors) > 0:
        print(f"\n❌ 发现 {len(errors)} 个错误:")
        for err in errors:
            print(f"  - {err}")
        return False
    else:
        print(f"\n✅ 数据格式验证通过")
        return True


def compute_statistics(features, labels):
    """计算统计信息"""
    print(f"\n{'='*80}")
    print("统计信息")
    print(f"{'='*80}")

    n_samples = len(labels)
    feat_dim = features.shape[1]

    # 基本信息
    print(f"\n基本信息:")
    print(f"  样本总数: {n_samples:,}")
    print(f"  特征维度: {feat_dim}")

    # 身份统计
    unique_ids, counts = np.unique(labels, return_counts=True)
    n_identities = len(unique_ids)

    print(f"\n身份统计:")
    print(f"  身份总数: {n_identities:,}")
    print(f"  平均每身份样本数: {n_samples / n_identities:.2f}")
    print(f"  最小每身份样本数: {counts.min()}")
    print(f"  最大每身份样本数: {counts.max()}")

    # 样本对统计
    total_pairs = n_samples * (n_samples - 1) // 2
    total_pos_pairs = sum(c * (c - 1) // 2 for c in counts)
    total_neg_pairs = total_pairs - total_pos_pairs

    print(f"\n样本对统计:")
    print(f"  总样本对数: {total_pairs:,}")
    print(f"  正样本对数: {total_pos_pairs:,} ({total_pos_pairs/total_pairs*100:.3f}%)")
    print(f"  负样本对数: {total_neg_pairs:,} ({total_neg_pairs/total_pairs*100:.3f}%)")

    # 计算复杂度估算
    print(f"\n计算复杂度:")
    if total_pairs < 1e6:
        print(f"  量级: {total_pairs/1e3:.1f}K 对 (千级)")
        print(f"  预计评估耗时: <1秒")
    elif total_pairs < 1e9:
        print(f"  量级: {total_pairs/1e6:.1f}M 对 (百万级)")
        print(f"  预计评估耗时: 1-10秒")
    else:
        print(f"  量级: {total_pairs/1e9:.2f}B 对 (十亿级)")
        print(f"  预计评估耗时: 10秒-数分钟")

    # 特征质量
    print(f"\n特征质量:")
    norms = np.linalg.norm(features, axis=1)
    print(f"  L2范数均值: {norms.mean():.6f} (期望: 1.000000)")
    print(f"  L2范数标准差: {norms.std():.8f} (期望: <0.00001)")
    print(f"  L2范数范围: [{norms.min():.6f}, {norms.max():.6f}]")

    # 判断归一化质量
    if norms.std() < 1e-5:
        print(f"  ✅ L2归一化质量: 优秀")
    elif norms.std() < 1e-4:
        print(f"  ⚠️  L2归一化质量: 良好")
    else:
        print(f"  ❌ L2归一化质量: 较差，可能影响评估")

    return {
        'n_samples': n_samples,
        'n_identities': n_identities,
        'total_pairs': total_pairs,
        'pos_pairs': total_pos_pairs,
        'neg_pairs': total_neg_pairs
    }


def sample_evaluation(features, labels, sample_size=1000, seed=42):
    """采样评估相似度分布"""
    print(f"\n{'='*80}")
    print(f"采样评估 (采样 {sample_size} 对)")
    print(f"{'='*80}")

    np.random.seed(seed)
    n_samples = len(labels)

    # 随机采样样本对
    print(f"\n采样样本对...")
    idx1 = np.random.randint(0, n_samples, sample_size)
    idx2 = np.random.randint(0, n_samples, sample_size)

    # 去除自配对
    mask = idx1 != idx2
    idx1 = idx1[mask]
    idx2 = idx2[mask]

    # 计算相似度
    print(f"计算相似度...")
    feat1 = features[idx1]
    feat2 = features[idx2]
    similarities = np.sum(feat1 * feat2, axis=1)

    # 确定正负样本
    is_positive = labels[idx1] == labels[idx2]
    pos_sims = similarities[is_positive]
    neg_sims = similarities[~is_positive]

    print(f"\n相似度分布:")
    print(f"  采样对数: {len(similarities)}")
    print(f"  正样本对数: {len(pos_sims)} ({len(pos_sims)/len(similarities)*100:.2f}%)")
    print(f"  负样本对数: {len(neg_sims)} ({len(neg_sims)/len(similarities)*100:.2f}%)")

    if len(pos_sims) > 0:
        print(f"\n正样本相似度:")
        print(f"  均值: {pos_sims.mean():.4f}")
        print(f"  标准差: {pos_sims.std():.4f}")
        print(f"  范围: [{pos_sims.min():.4f}, {pos_sims.max():.4f}]")
        print(f"  分位数: 25%={np.percentile(pos_sims, 25):.4f}, "
              f"50%={np.percentile(pos_sims, 50):.4f}, "
              f"75%={np.percentile(pos_sims, 75):.4f}")

    if len(neg_sims) > 0:
        print(f"\n负样本相似度:")
        print(f"  均值: {neg_sims.mean():.4f}")
        print(f"  标准差: {neg_sims.std():.4f}")
        print(f"  范围: [{neg_sims.min():.4f}, {neg_sims.max():.4f}]")
        print(f"  分位数: 25%={np.percentile(neg_sims, 25):.4f}, "
              f"50%={np.percentile(neg_sims, 50):.4f}, "
              f"75%={np.percentile(neg_sims, 75):.4f}")

    # 评估可分性
    if len(pos_sims) > 0 and len(neg_sims) > 0:
        separation = pos_sims.mean() - neg_sims.mean()
        print(f"\n可分性评估:")
        print(f"  正负样本均值差: {separation:.4f}")

        if separation > 0.3:
            print(f"  ✅ 可分性: 优秀 (容易区分)")
        elif separation > 0.2:
            print(f"  ✅ 可分性: 良好")
        elif separation > 0.1:
            print(f"  ⚠️  可分性: 中等")
        else:
            print(f"  ❌ 可分性: 较差 (难以区分)")


def main():
    parser = argparse.ArgumentParser(
        description='快速验证测试数据的质量和可用性'
    )
    parser.add_argument('file', type=str, nargs='?',
                        help='测试数据文件路径')
    parser.add_argument('--file', dest='file_arg', type=str,
                        help='测试数据文件路径（替代位置参数）')
    parser.add_argument('--sample', type=int, default=1000,
                        help='采样评估的样本对数 (默认: 1000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    parser.add_argument('--no-sample', action='store_true',
                        help='跳过采样评估（用于超大数据集）')

    args = parser.parse_args()

    # 确定文件路径
    file_path = args.file or args.file_arg

    if not file_path:
        print("错误: 请指定测试数据文件")
        print("\n使用方法:")
        print("  python validate_test_data.py test_data.pkl")
        print("  python validate_test_data.py --file test_data.pkl")
        return

    file_path = Path(file_path)

    if not file_path.exists():
        print(f"错误: 文件不存在: {file_path}")
        return

    print(f"{'='*80}")
    print(f"测试数据验证工具")
    print(f"{'='*80}")
    print(f"\n文件: {file_path.absolute()}")
    print(f"文件大小: {file_path.stat().st_size / (1024*1024):.2f} MB")

    try:
        # 1. 加载数据
        features, labels, paths = load_data(file_path)

        # 2. 验证格式
        if not validate_format(features, labels, paths):
            print("\n❌ 数据格式验证失败")
            return

        # 3. 计算统计信息
        stats = compute_statistics(features, labels)

        # 4. 采样评估
        if not args.no_sample:
            # 对于超大数据集，自动调整采样大小
            if stats['n_samples'] > 100000:
                sample_size = min(args.sample, 10000)
                if sample_size < args.sample:
                    print(f"\n注意: 数据集较大，自动调整采样大小为 {sample_size}")
            else:
                sample_size = args.sample

            sample_evaluation(features, labels, sample_size, args.seed)
        else:
            print(f"\n跳过采样评估（使用了 --no-sample 选项）")

        # 5. 总结
        print(f"\n{'='*80}")
        print(f"验证完成")
        print(f"{'='*80}")
        print(f"\n✅ 数据集可用于评估测试")
        print(f"\n推荐使用方法:")
        print(f"  import pickle")
        print(f"  with open('{file_path}', 'rb') as f:")
        print(f"      features, _, labels, paths = pickle.load(f)")
        print(f"{'='*80}\n")

    except Exception as e:
        print(f"\n❌ 验证过程出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
