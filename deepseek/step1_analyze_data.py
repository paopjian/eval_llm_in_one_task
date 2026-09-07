#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step1: 特征文件读取与数据结构理解
- 读取 s4_0618_enhance.pkl
- 统计样本/身份/正负样本对
- 检查 L2 归一化
"""
import numpy as np
from faireval_lib import load_data, pair_counts, fmt_pair


def main():
    print('=' * 78)
    print('Step1 数据读取与分析  (s4_0618_enhance.pkl)')
    print('=' * 78)
    feats, ids, paths, t_load = load_data()
    N = len(ids)
    n_uniq = int(np.unique(ids).size)
    total_pairs, pos_pairs, neg_pairs = pair_counts(ids)

    print('\n[数据基本信息]')
    print('  特征矩阵形状 : %s (L2归一化 float32)' % (feats.shape,))
    print('  样本总数     : %s' % fmt_pair(N))
    print('  身份总数     : %s' % fmt_pair(n_uniq))
    print('  加载耗时     : %.2f s' % t_load)

    print('\n[样本对统计]  (定义: 同身份 i<j 为正样本对, 否则为负样本对)')
    print('  总样本对数   : %s' % fmt_pair(total_pairs))
    print('  正样本对数   : %s  (%s%%)' % (fmt_pair(pos_pairs),
                                           '%.5f' % (100.0 * pos_pairs / total_pairs)))
    print('  负样本对数   : %s  (%s%%)' % (fmt_pair(neg_pairs),
                                           '%.4f' % (100.0 * neg_pairs / total_pairs)))

    print('\n[归一化与身份分布检查]')
    norms = np.linalg.norm(feats, axis=1)
    print('  L2范数 均值=%.6f 标准差=%.3e  → %s' %
          (norms.mean(), norms.std(),
           '已归一化' if abs(norms.mean() - 1.0) < 0.01 else '未归一化!'))
    uniq, counts = np.unique(ids, return_counts=True)
    print('  每个身份图片数: 平均=%.2f 最小=%d 最大=%d' %
          (N / n_uniq, counts.min(), counts.max()))
    print('  文件路径示例  : %s' % paths[0])
    print('  flip特征列表  : 空(本任务不使用)')
    print('\n' + '=' * 78)
    print('Step1 完成: 数据共 %s 张人脸特征, 可进入相似度计算阶段' % fmt_pair(N))
    return feats, ids, paths


if __name__ == '__main__':
    main()
