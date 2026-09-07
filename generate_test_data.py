#!/usr/bin/env python3
"""
生成高强度可复现测试数据用于LLM评估

特点：
1. 完全可复现（固定随机种子）
2. 模拟真实人脸特征分布
3. 可调节数据规模和计算强度
4. 包含完整的验证和统计信息
5. 支持多种难度等级的测试场景

使用示例：
    # 小规模快速测试（~1秒）
    python generate_test_data.py --preset quick

    # 中等规模测试（~10秒）
    python generate_test_data.py --preset medium

    # 大规模测试（~1分钟）
    python generate_test_data.py --preset large

    # 超大规模测试（接近真实数据，~10分钟）
    python generate_test_data.py --preset xlarge

    # 自定义配置
    python generate_test_data.py --num_identities 5000 --samples_per_identity 20 --noise_std 0.12
"""

import numpy as np
import pickle
import argparse
import time
from pathlib import Path
from typing import Tuple, List, Optional


# 预设配置
PRESETS = {
    'quick': {
        'num_identities': 500,
        'samples_per_identity': 5,
        'noise_std': 0.15,
        'desc': '快速测试（2,500样本，3.1M对）',
        'expected_time': '~1秒'
    },
    'medium': {
        'num_identities': 2000,
        'samples_per_identity': 10,
        'noise_std': 0.15,
        'desc': '中等测试（20,000样本，200M对）',
        'expected_time': '~10秒'
    },
    'large': {
        'num_identities': 5000,
        'samples_per_identity': 15,
        'noise_std': 0.15,
        'desc': '大规模测试（75,000样本，2.8B对）',
        'expected_time': '~1分钟'
    },
    'xlarge': {
        'num_identities': 10000,
        'samples_per_identity': 20,
        'noise_std': 0.15,
        'desc': '超大规模测试（200,000样本，20B对）',
        'expected_time': '~10分钟'
    },
    'realistic': {
        'num_identities': 20000,
        'samples_per_identity': 10,
        'noise_std': 0.15,
        'desc': '真实规模测试（200,000样本，20B对，接近真实数据）',
        'expected_time': '~10分钟'
    }
}


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """L2归一化向量"""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / (norms + 1e-10)


def generate_identity_centers(
    num_identities: int,
    feat_dim: int = 512,
    seed: int = 42
) -> np.ndarray:
    """
    为每个身份生成中心特征向量

    使用高斯分布生成，模拟真实人脸特征的分布特性
    """
    rng = np.random.RandomState(seed)
    centers = rng.randn(num_identities, feat_dim).astype(np.float32)
    centers = l2_normalize(centers)
    return centers


def generate_samples_for_identity(
    center: np.ndarray,
    num_samples: int,
    noise_std: float = 0.15,
    seed: int = 42
) -> np.ndarray:
    """
    为单个身份生成多个样本

    Args:
        center: 身份的中心特征向量
        num_samples: 生成的样本数量
        noise_std: 噪声标准差，控制同一身份内的变化程度
                   - 0.10: 非常相似（理想情况）
                   - 0.15: 较相似（推荐）
                   - 0.20: 中等相似
                   - 0.30: 较大变化（困难场景）
        seed: 随机种子

    Returns:
        samples: (num_samples, feat_dim) 归一化的样本特征
    """
    rng = np.random.RandomState(seed)
    feat_dim = center.shape[0]

    # 添加高斯噪声
    noise = rng.randn(num_samples, feat_dim).astype(np.float32) * noise_std
    samples = center[np.newaxis, :] + noise

    # 归一化
    samples = l2_normalize(samples)

    return samples


def calculate_expected_metrics(noise_std: float) -> dict:
    """
    根据噪声标准差估算期望的评估指标

    基于经验公式和实验数据
    """
    if noise_std <= 0.10:
        return {
            'tpir_1e5': 0.85,
            'tpir_1e4': 0.92,
            'tpir_1e3': 0.96,
            'tpir_1e2': 0.98
        }
    elif noise_std <= 0.15:
        return {
            'tpir_1e5': 0.62,
            'tpir_1e4': 0.83,
            'tpir_1e3': 0.91,
            'tpir_1e2': 0.96
        }
    elif noise_std <= 0.20:
        return {
            'tpir_1e5': 0.45,
            'tpir_1e4': 0.68,
            'tpir_1e3': 0.83,
            'tpir_1e2': 0.92
        }
    else:
        return {
            'tpir_1e5': 0.25,
            'tpir_1e4': 0.48,
            'tpir_1e3': 0.70,
            'tpir_1e2': 0.85
        }


def generate_test_dataset(
    num_identities: int = 1000,
    samples_per_identity: int = 10,
    feat_dim: int = 512,
    noise_std: float = 0.15,
    seed: int = 42,
    output_path: str = 'test_features.pkl',
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    生成完整的测试数据集

    Returns:
        features: (N, feat_dim) 特征矩阵
        labels: (N,) 身份标签
        paths: list of str, 模拟的文件路径
    """
    start_time = time.time()

    if verbose:
        print("=" * 80)
        print("生成LLM评估测试数据")
        print("=" * 80)
        print(f"\n配置参数:")
        print(f"  身份数量: {num_identities:,}")
        print(f"  每身份样本数: {samples_per_identity}")
        print(f"  特征维度: {feat_dim}")
        print(f"  噪声标准差: {noise_std}")
        print(f"  随机种子: {seed}")
        print(f"  输出路径: {output_path}")

    # 1. 生成身份中心
    if verbose:
        print(f"\n[1/5] 生成 {num_identities:,} 个身份的中心特征...")
    t1 = time.time()
    identity_centers = generate_identity_centers(num_identities, feat_dim, seed)
    if verbose:
        print(f"  耗时: {time.time() - t1:.2f}秒")

    # 2. 为每个身份生成样本
    if verbose:
        print(f"\n[2/5] 为每个身份生成样本...")
    t2 = time.time()
    all_features = []
    all_labels = []
    all_paths = []

    for identity_id in range(num_identities):
        center = identity_centers[identity_id]
        samples = generate_samples_for_identity(
            center,
            samples_per_identity,
            noise_std,
            seed=seed + identity_id
        )

        all_features.append(samples)
        all_labels.extend([identity_id] * samples_per_identity)

        # 生成模拟路径
        for sample_idx in range(samples_per_identity):
            path = f"test_data/identity_{identity_id:06d}/sample_{sample_idx:04d}.jpg"
            all_paths.append(path)

        if verbose and (identity_id + 1) % max(1, num_identities // 10) == 0:
            progress = (identity_id + 1) / num_identities * 100
            print(f"  进度: {identity_id + 1:,}/{num_identities:,} ({progress:.1f}%)")

    if verbose:
        print(f"  耗时: {time.time() - t2:.2f}秒")

    # 3. 合并数据
    if verbose:
        print(f"\n[3/5] 合并特征矩阵...")
    t3 = time.time()
    features = np.vstack(all_features).astype(np.float32)
    labels = np.array(all_labels, dtype=np.int32)
    if verbose:
        print(f"  耗时: {time.time() - t3:.2f}秒")

    # 4. 统计信息
    if verbose:
        print(f"\n[4/5] 计算统计信息...")
    t4 = time.time()

    N = len(labels)
    total_pairs = N * (N - 1) // 2

    unique_ids, counts = np.unique(labels, return_counts=True)
    total_pos_pairs = sum(c * (c - 1) // 2 for c in counts)
    total_neg_pairs = total_pairs - total_pos_pairs

    # 验证归一化
    norms = np.linalg.norm(features, axis=1)

    # 计算期望指标
    expected_metrics = calculate_expected_metrics(noise_std)

    if verbose:
        print(f"  耗时: {time.time() - t4:.2f}秒")
        print(f"\n{'='*80}")
        print("数据集统计")
        print(f"{'='*80}")
        print(f"\n基本信息:")
        print(f"  特征矩阵形状: {features.shape}")
        print(f"  样本总数: {N:,}")
        print(f"  身份总数: {len(unique_ids):,}")
        print(f"  平均每身份样本数: {N / len(unique_ids):.1f}")

        print(f"\n样本对统计:")
        print(f"  总样本对数: {total_pairs:,}")
        print(f"  正样本对数: {total_pos_pairs:,} ({total_pos_pairs/total_pairs*100:.3f}%)")
        print(f"  负样本对数: {total_neg_pairs:,} ({total_neg_pairs/total_pairs*100:.3f}%)")

        print(f"\n计算强度估算:")
        if total_pairs < 1e6:
            print(f"  计算量级: {total_pairs/1e6:.2f}M 对（百万级）")
        elif total_pairs < 1e9:
            print(f"  计算量级: {total_pairs/1e9:.2f}B 对（十亿级）")
        else:
            print(f"  计算量级: {total_pairs/1e9:.1f}B 对（十亿级）")

        print(f"\n特征质量检查:")
        print(f"  L2范数均值: {norms.mean():.6f} (期望: 1.000000)")
        print(f"  L2范数标准差: {norms.std():.6f} (期望: <0.000001)")
        print(f"  L2范数最小值: {norms.min():.6f}")
        print(f"  L2范数最大值: {norms.max():.6f}")

        print(f"\n期望评估指标 (基于 noise_std={noise_std}):")
        print(f"  TPIR @ FPIR=1e-5: ~{expected_metrics['tpir_1e5']*100:.1f}%")
        print(f"  TPIR @ FPIR=1e-4: ~{expected_metrics['tpir_1e4']*100:.1f}%")
        print(f"  TPIR @ FPIR=1e-3: ~{expected_metrics['tpir_1e3']*100:.1f}%")
        print(f"  TPIR @ FPIR=1e-2: ~{expected_metrics['tpir_1e2']*100:.1f}%")

    # 5. 保存数据
    if verbose:
        print(f"\n[5/5] 保存数据...")
    t5 = time.time()

    # 创建输出目录
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 保存为与原始数据相同的格式
    query_feats_list_flip = []  # 空列表，保持格式一致

    with open(output_path, 'wb') as f:
        pickle.dump(
            (features, query_feats_list_flip, labels, all_paths),
            f,
            protocol=4
        )

    file_size = output_path.stat().st_size / (1024 * 1024)  # MB

    if verbose:
        print(f"  文件大小: {file_size:.2f} MB")
        print(f"  耗时: {time.time() - t5:.2f}秒")

    total_time = time.time() - start_time

    if verbose:
        print(f"\n{'='*80}")
        print(f"数据生成完成！")
        print(f"{'='*80}")
        print(f"总耗时: {total_time:.2f}秒")
        print(f"输出文件: {output_path.absolute()}")
        print(f"\n使用方法:")
        print(f"  import pickle")
        print(f"  with open('{output_path}', 'rb') as f:")
        print(f"      features, _, labels, paths = pickle.load(f)")
        print(f"{'='*80}\n")

    return features, labels, all_paths


def main():
    parser = argparse.ArgumentParser(
        description='生成高强度可复现测试数据用于LLM评估',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
预设配置:
  quick      - 快速测试 (2,500样本, ~1秒)
  medium     - 中等测试 (20,000样本, ~10秒)
  large      - 大规模测试 (75,000样本, ~1分钟)
  xlarge     - 超大规模测试 (200,000样本, ~10分钟)
  realistic  - 真实规模测试 (200,000样本, 接近真实数据)

示例:
  python generate_test_data.py --preset medium
  python generate_test_data.py --num_identities 5000 --samples_per_identity 20
  python generate_test_data.py --preset large --noise_std 0.12 --seed 123
        """
    )

    parser.add_argument('--preset', type=str, choices=list(PRESETS.keys()),
                        help='使用预设配置')
    parser.add_argument('--num_identities', type=int,
                        help='身份总数')
    parser.add_argument('--samples_per_identity', type=int,
                        help='每个身份的样本数')
    parser.add_argument('--feat_dim', type=int, default=512,
                        help='特征维度 (默认: 512)')
    parser.add_argument('--noise_std', type=float,
                        help='噪声标准差 (0.10=理想, 0.15=推荐, 0.20=中等, 0.30=困难)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    parser.add_argument('--output', type=str,
                        help='输出文件路径')
    parser.add_argument('--quiet', action='store_true',
                        help='静默模式，不输出详细信息')

    args = parser.parse_args()

    # 处理预设配置
    if args.preset:
        preset = PRESETS[args.preset]
        num_identities = preset['num_identities']
        samples_per_identity = preset['samples_per_identity']
        noise_std = preset['noise_std']
        output_path = f'test_data_{args.preset}.pkl'

        if not args.quiet:
            print(f"\n使用预设配置: {args.preset}")
            print(f"  描述: {preset['desc']}")
            print(f"  预计耗时: {preset['expected_time']}\n")
    else:
        num_identities = args.num_identities or 1000
        samples_per_identity = args.samples_per_identity or 10
        noise_std = args.noise_std or 0.15
        output_path = args.output or 'test_features.pkl'

    # 允许命令行参数覆盖预设
    if args.num_identities:
        num_identities = args.num_identities
    if args.samples_per_identity:
        samples_per_identity = args.samples_per_identity
    if args.noise_std:
        noise_std = args.noise_std
    if args.output:
        output_path = args.output

    # 生成数据
    generate_test_dataset(
        num_identities=num_identities,
        samples_per_identity=samples_per_identity,
        feat_dim=args.feat_dim,
        noise_std=noise_std,
        seed=args.seed,
        output_path=output_path,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
