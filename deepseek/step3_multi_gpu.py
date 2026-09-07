#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step3: 多卡并行版相似度统计与评估
- 行块轮询分配各GPU(上三角负载均衡), 每卡独立子进程
- 结果与单卡完全一致(整数级一致)
- 输出: outputs/histograms_multi.npz, outputs/metrics_multi.json
"""
import os
import json
import argparse
import numpy as np
import faireval_lib as fl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', type=str, default='0,1,2,3,4,5,6')
    ap.add_argument('--block', type=int, default=fl.DEFAULT_BLOCK)
    ap.add_argument('--nbins', type=int, default=fl.DEFAULT_NBINS)
    args = ap.parse_args()
    gpus = [int(x) for x in args.gpus.split(',') if x != '']
    os.makedirs(fl.OUT_DIR, exist_ok=True)

    print('=' * 78)
    print('Step3 多卡并行相似度统计与评估  GPU数=%d %s  block=%d  nbins=%d' %
          (len(gpus), gpus, args.block, args.nbins))
    print('=' * 78)
    feats, ids, paths, t_load = fl.load_data()
    N = len(ids)
    total_pairs, pos_pairs, neg_pairs = fl.pair_counts(ids)
    print('样本N=%s 正对=%s 负对=%s (载入 %.2fs)' %
          (fl.fmt_pair(N), fl.fmt_pair(pos_pairs), fl.fmt_pair(neg_pairs), t_load))

    print('\n[计算] 多卡并行上三角分块统计 (行块轮询, 各卡独立子进程) ...')
    res = fl.run_hist_pass(gpus, args.nbins, args.block, N)
    ph, nh = res['pos'], res['neg']
    for g, tc, tw in res['per_gpu']:
        print('  GPU%d: GPU计算=%.2fs, 进程墙钟=%.2fs' % (g, tc, tw))
    print('  总体墙钟: %.2fs  (GPU纯计算合计 %.2fs)' %
          (res['wall'], sum(x[1] for x in res['per_gpu'])))
    s_pos, s_neg = int(ph.sum()), int(nh.sum())
    print('  直方图计数: 正=%s/%s 负=%s/%s  %s' %
          (fl.fmt_pair(s_pos), fl.fmt_pair(pos_pairs),
           fl.fmt_pair(s_neg), fl.fmt_pair(neg_pairs),
           '一致OK' if s_pos == pos_pairs and s_neg == neg_pairs else '不一致FAIL!'))
    assert s_pos == pos_pairs and s_neg == neg_pairs, '计数不一致!'

    spath = os.path.join(fl.OUT_DIR, 'histograms_single.npz')
    if os.path.exists(spath):
        sd = np.load(spath)
        if int(sd['nbins']) == args.nbins and int(sd['block']) == args.block:
            same = bool((sd['pos_hist'] == ph).all() and (sd['neg_hist'] == nh).all())
            print('  与单卡直方图(整数级)一致性: %s' % ('一致OK' if same else '不一致!'))
            import json as _json
            with open(os.path.join(fl.OUT_DIR, 'metrics_single.json')) as f:
                m1 = _json.load(f)
            print('  加速比: 单卡%.1fs / 多卡%.1fs = %.2fx' %
                  (m1['t_hist_pass'], res['wall'],
                   m1['t_hist_pass'] / max(res['wall'], 1e-9)))

    print('\n[指标] TPIR@FPIR')
    fpir_pts = [1e-5, 1e-4, 1e-3, 1e-2]
    metrics = fl.compute_metrics(ph, nh, pos_pairs, neg_pairs, fpir_pts, args.nbins)
    print('  %-10s %-14s %-10s %s' % ('FPIR', '阈值(相似度)', 'TPIR', '期望区间'))
    expect = {1e-5: (0.60, 0.65), 1e-4: (0.82, 0.85), 1e-3: (0.90, 0.93), 1e-2: (0.95, 0.97)}
    for p in metrics['points']:
        lo, hi = expect.get(p['fpir'], (0.0, 1.0))
        mark = 'OK' if lo <= p['tpir'] <= hi else '超出[%.2f,%.2f]!' % (lo, hi)
        print('  %-10.0e %-14.6f %-10.4f %s' % (p['fpir'], p['threshold'], p['tpir'], mark))

    np.savez(os.path.join(fl.OUT_DIR, 'histograms_multi.npz'),
             pos_hist=ph, neg_hist=nh, nbins=args.nbins,
             total_pos=pos_pairs, total_neg=neg_pairs, block=args.block)
    fl.save_json(os.path.join(fl.OUT_DIR, 'metrics_multi.json'), dict(
        n_samples=N, total_pairs=total_pairs, pos_pairs=pos_pairs, neg_pairs=neg_pairs,
        nbins=args.nbins, block=args.block, method='multi_gpu', gpus=gpus,
        t_load=t_load, t_hist_pass=res['wall'],
        t_calc=sum(x[1] for x in res['per_gpu']),
        per_gpu=[dict(gpu=g, calc=tc, wall=tw) for g, tc, tw in res['per_gpu']],
        points=metrics['points']))
    print('\n已保存: outputs/histograms_multi.npz, outputs/metrics_multi.json')
    print('=' * 78)


if __name__ == '__main__':
    main()
