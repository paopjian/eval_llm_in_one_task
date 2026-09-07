#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step2: 单卡(GPU)分块相似度计算 + 直方图统计 + TPIR@FPIR 指标
- 先跑 CPU float64 全量子集自检, 验证数值正确性
- 输出: outputs/histograms_single.npz, outputs/metrics_single.json
"""
import os
import time
import argparse
import numpy as np
import faireval_lib as fl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--block', type=int, default=fl.DEFAULT_BLOCK)
    ap.add_argument('--nbins', type=int, default=fl.DEFAULT_NBINS)
    ap.add_argument('--no-selfcheck', action='store_true')
    args = ap.parse_args()
    os.makedirs(fl.OUT_DIR, exist_ok=True)

    print('=' * 78)
    print('Step2 单卡相似度计算与评估   GPU=%d  block=%d  nbins=%d' %
          (args.gpu, args.block, args.nbins))
    print('=' * 78)
    feats, ids, paths, t_load = fl.load_data()
    N = len(ids)
    total_pairs, pos_pairs, neg_pairs = fl.pair_counts(ids)
    print('样本N=%s 正对=%s 负对=%s (载入 %.2fs)' %
          (fl.fmt_pair(N), fl.fmt_pair(pos_pairs), fl.fmt_pair(neg_pairs), t_load))

    if not args.no_selfcheck:
        print('\n[自检] 前2000样本: GPU直方图管线 vs CPU float64 全量直接计算')
        t0 = time.perf_counter()
        ok = fl.selfcheck(gpu_id=args.gpu, n_subset=2000, nbins=args.nbins)
        print('        自检耗时 %.1fs' % (time.perf_counter() - t0))
        assert ok, '自检失败, 请检查实现!'

    print('\n[计算] 单卡全量上三角分块统计 (%.0f亿次浮点运算 fp32) ...' %
          ((2 * (N * (N - 1) // 2) * 512) / 1e9))
    res = fl.run_hist_pass([args.gpu], args.nbins, args.block, N)
    ph, nh = res['pos'], res['neg']
    for g, tc, tw in res['per_gpu']:
        print('  GPU%d: GPU计算=%.2fs, 含载入/初始化=%.2fs' % (g, tc, tw))
    print('  墙钟耗时: %.2fs' % res['wall'])
    s_pos, s_neg = int(ph.sum()), int(nh.sum())
    print('  直方图计数: 正=%s/%s 负=%s/%s  %s' %
          (fl.fmt_pair(s_pos), fl.fmt_pair(pos_pairs),
           fl.fmt_pair(s_neg), fl.fmt_pair(neg_pairs),
           '一致OK' if s_pos == pos_pairs and s_neg == neg_pairs else '不一致FAIL!'))
    assert s_pos == pos_pairs and s_neg == neg_pairs, '计数不一致!'

    print('\n[指标] TPIR@FPIR (阈值由负样本分布反解)')
    fpir_pts = [1e-5, 1e-4, 1e-3, 1e-2]
    metrics = fl.compute_metrics(ph, nh, pos_pairs, neg_pairs, fpir_pts, args.nbins)
    print('  %-10s %-14s %-10s %s' % ('FPIR', '阈值(相似度)', 'TPIR', '期望区间'))
    expect = {1e-5: (0.60, 0.65), 1e-4: (0.82, 0.85), 1e-3: (0.90, 0.93), 1e-2: (0.95, 0.97)}
    for p in metrics['points']:
        lo, hi = expect.get(p['fpir'], (0.0, 1.0))
        mark = 'OK' if lo <= p['tpir'] <= hi else '超出[%.2f,%.2f]!' % (lo, hi)
        print('  %-10.0e %-14.6f %-10.4f %s' % (p['fpir'], p['threshold'], p['tpir'], mark))

    np.savez(os.path.join(fl.OUT_DIR, 'histograms_single.npz'),
             pos_hist=ph, neg_hist=nh, nbins=args.nbins,
             total_pos=pos_pairs, total_neg=neg_pairs, block=args.block)
    fl.save_json(os.path.join(fl.OUT_DIR, 'metrics_single.json'), dict(
        n_samples=N, total_pairs=total_pairs, pos_pairs=pos_pairs, neg_pairs=neg_pairs,
        nbins=args.nbins, block=args.block, method='single_gpu', gpus=[args.gpu],
        t_load=t_load, t_hist_pass=res['wall'], t_calc=res['per_gpu'][0][1],
        points=metrics['points']))
    print('\n已保存: outputs/histograms_single.npz, outputs/metrics_single.json')
    print('=' * 78)


if __name__ == '__main__':
    main()
