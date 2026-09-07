#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_eval.py — 一键评估入口: 数据→多卡相似度统计→TPIR@FPIR→可视化(→样本对提取)

用法示例:
  python run_eval.py                          # 全流程(自动检测GPU)
  python run_eval.py --gpus 0,1,2             # 指定GPU
  python run_eval.py --extract-fpir 1e-3      # 追加 above/below 样本对提取
"""
import os
import json
import time
import pickle
import argparse
import numpy as np
import faireval_lib as fl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', type=str, required=True, help='数据文件路径（.pkl格式）')
    ap.add_argument('--gpus', type=str, default='auto')
    ap.add_argument('--block', type=int, default=fl.DEFAULT_BLOCK)
    ap.add_argument('--nbins', type=int, default=fl.DEFAULT_NBINS)
    ap.add_argument('--fpir-points', type=str, default='1e-5,1e-4,1e-3,1e-2')
    ap.add_argument('--extract-fpir', type=float, default=None,
                    help='可选: 提取该FPIR阈值对应的 above(负对)/below(正对) 样本对')
    ap.add_argument('--extract-max', type=int, default=fl.MAX_EXTRACT_DEFAULT)
    ap.add_argument('--single', action='store_true', help='单卡模式(默认多卡)')
    args = ap.parse_args()

    if args.gpus == 'auto':
        import subprocess
        out = subprocess.run(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
                             capture_output=True, text=True)
        gpus = [int(x.strip()) for x in out.stdout.split() if x.strip().isdigit()]
        if not gpus:
            gpus = [0]
    else:
        gpus = [int(x) for x in args.gpus.split(',') if x != '']
    if args.single:
        gpus = gpus[:1]
    fpir_pts = [float(x) for x in args.fpir_points.split(',') if x.strip()]
    os.makedirs(fl.OUT_DIR, exist_ok=True)
    t_start = time.perf_counter()
    print('=' * 78)
    print(' 人脸特征相似度评估  |  NxN上三角 |  TPIR@FPIR |  %d卡并行' % len(gpus))
    print('=' * 78)

    # ---------- 1. 数据 ----------
    feats, ids, paths, t_load = fl.load_data(args.pkl)
    N = len(ids)
    total_pairs, pos_pairs, neg_pairs = fl.pair_counts(ids)
    print('[1/4 数据] N=%s, 身份=%d, 正对=%s, 负对=%s  (载入 %.2fs)' %
          (fl.fmt_pair(N), int(np.unique(ids).size), fl.fmt_pair(pos_pairs),
           fl.fmt_pair(neg_pairs), t_load))

    # ---------- 2. 相似度直方图统计 ----------
    print('[2/4 计算] 分块上三角 matmul (block=%d, nbins=%d), GPU%s ...' %
          (args.block, args.nbins, gpus))
    res = fl.run_hist_pass(gpus, args.nbins, args.block, N)
    ph, nh = res['pos'], res['neg']
    for g, tc, tw in res['per_gpu']:
        print('       GPU%d: GPU计算 %.2fs / 进程墙钟 %.2fs' % (g, tc, tw))
    s_pos, s_neg = int(ph.sum()), int(nh.sum())
    assert s_pos == pos_pairs and s_neg == neg_pairs, '直方图计数不一致!'
    print('       直方图累计: 正=%s 负=%s 一致OK | 总体墙钟 %.2fs' %
          (fl.fmt_pair(s_pos), fl.fmt_pair(s_neg), res['wall']))

    # ---------- 3. 指标 ----------
    print('[3/4 指标] TPIR@FPIR')
    metrics = fl.compute_metrics(ph, nh, pos_pairs, neg_pairs, fpir_pts, args.nbins)
    expect = {1e-5: (0.60, 0.65), 1e-4: (0.82, 0.85), 1e-3: (0.90, 0.93), 1e-2: (0.95, 0.97)}
    print('  %-10s %-16s %-12s %s' % ('FPIR', '阈值(相似度)', 'TPIR', '期望区间判定'))
    all_ok = True
    for p in metrics['points']:
        lo, hi = expect.get(round(p['fpir'], 6), (0.0, 1.0))
        ok = lo <= p['tpir'] <= hi
        all_ok &= ok
        print('  %-10.0e %-16.6f %-12.4f %s' %
              (p['fpir'], p['threshold'], p['tpir'],
               'OK' if ok else '超出期望[%.2f,%.2f]' % (lo, hi)))
    print('  与文档期望范围一致性: %s' % ('全部OK' if all_ok else '存在偏差, 请检查!'))

    # ---------- 4. 保存 + 可视化 ----------
    np.savez(os.path.join(fl.OUT_DIR, 'histograms.npz'), pos_hist=ph, neg_hist=nh,
             nbins=args.nbins, total_pos=pos_pairs, total_neg=neg_pairs,
             block=args.block)
    meta = dict(n_samples=N, n_ids=int(np.unique(ids).size),
                total_pairs=total_pairs, pos_pairs=pos_pairs, neg_pairs=neg_pairs,
                nbins=args.nbins, block=args.block,
                method='multi_gpu' if len(gpus) > 1 else 'single_gpu',
                gpus=gpus, t_load=t_load, t_hist_pass=res['wall'],
                t_calc=sum(x[1] for x in res['per_gpu']),
                t_total=time.perf_counter() - t_start, points=metrics['points'])
    fl.save_json(os.path.join(fl.OUT_DIR, 'metrics.json'), meta)
    print('[4/4 可视化] 中文字体: %s' % fl.setup_chinese_font())
    t0 = time.perf_counter()
    fig1 = fl.plot_distributions(ph, nh, pos_pairs, neg_pairs, metrics, args.nbins,
                                 os.path.join(fl.OUT_DIR, 'similarity_distribution.png'))
    fig2 = fl.plot_curve(metrics, os.path.join(fl.OUT_DIR, 'tpir_fpir_curve.png'))
    print('       %s' % fig1)
    print('       %s  (绘图 %.2fs)' % (fig2, time.perf_counter() - t0))
    print('       outputs/metrics.json, outputs/histograms.npz')

    # ---------- 可选: 样本对提取 ----------
    if args.extract_fpir is not None:
        print('\n[提取] FPIR=%.0e 对应阈值的错误样本对 '
              '(above=负对sim>t 误接受, below=正对sim<t 漏报)' % args.extract_fpir)
        thr = None
        for p in metrics['points']:
            if abs(p['fpir'] - args.extract_fpir) < 1e-12:
                thr = p['threshold']
        if thr is None:
            thr = fl.threshold_at_fpir(nh, neg_pairs, args.extract_fpir, args.nbins)
        t0 = time.perf_counter()
        res2 = fl.run_extract_pass(gpus, args.block, N, thr, ['above', 'below'],
                                   args.extract_max)
        collected = {'above': [], 'below': []}
        for g, gd, _w in res2:
            for d in collected:
                if gd[d] is not None:
                    collected[d].append(gd[d])
        for d, desc in (('above', '负样本对 sim>阈值(误接受)'),
                        ('below', '正样本对 sim<阈值(漏报)')):
            if not collected[d]:
                print('    %-5s %s: 0 对' % (d, desc))
                continue
            rows = np.concatenate([c[0] for c in collected[d]])
            cols = np.concatenate([c[1] for c in collected[d]])
            sims = np.concatenate([c[2] for c in collected[d]])
            n_cand = len(sims)
            asc = (d == 'below')
            idx = np.argsort(sims) if asc else np.argsort(sims)[::-1]
            idx = idx[:args.extract_max]
            rows, cols, sims = rows[idx], cols[idx], sims[idx]
            fn = os.path.join(fl.OUT_DIR,
                              'pairs_%s_FPIR%.0e_n%d.pkl' % (d, args.extract_fpir, len(sims)))
            est = ((1 - fl.frac_above(ph, pos_pairs, thr, args.nbins)) * pos_pairs
                   if d == 'below' else fl.frac_above(nh, neg_pairs, thr, args.nbins) * neg_pairs)
            with open(fn, 'wb') as f:
                pickle.dump(dict(direction=d, threshold=thr,
                                 est_total_crossing=int(round(est)),
                                 n_candidates_found=int(n_cand),
                                 pairs=[(paths[int(rows[k])], paths[int(cols[k])],
                                         float(sims[k])) for k in range(len(sims))]), f)
            print('    %-5s %s: 候选=%s → 保存%s对 → %s' %
                  (d, desc, fl.fmt_pair(n_cand), fl.fmt_pair(len(sims)), fn))
        print('    提取遍历耗时 %.2fs' % (time.perf_counter() - t0))
    print('\n' + '=' * 78)
    print(' 完成! 总耗时 %.1fs | 输出目录: %s/' % (time.perf_counter() - t_start, fl.OUT_DIR))
    print('=' * 78)


if __name__ == '__main__':
    main()
