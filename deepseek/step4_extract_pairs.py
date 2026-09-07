#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step4: 提取 above/below 阈值的样本对 (错误分析用)
- above: 负样本对中 sim > threshold (误接受/假匹配)
- below: 正样本对中 sim < threshold (漏报/误拒绝)
- 多卡二次遍历 + 流式全局 top-K; 保存 pkl
"""
import os
import time
import json
import pickle
import argparse
import numpy as np
import faireval_lib as fl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', type=str, default='0,1,2,3,4,5,6')
    ap.add_argument('--block', type=int, default=fl.DEFAULT_BLOCK)
    ap.add_argument('--threshold', type=float, default=None,
                    help='相似度阈值; 缺省时由 --fpir 从历史直方图反解')
    ap.add_argument('--fpir', type=float, default=1e-3,
                    help='由该FPIR目标对应的阈值提取(仅当未给--threshold)')
    ap.add_argument('--dirs', type=str, default='above,below')
    ap.add_argument('--max', type=int, default=fl.MAX_EXTRACT_DEFAULT)
    ap.add_argument('--indices', action='store_true', help='仅保存索引+sim(不解析路径)')
    args = ap.parse_args()
    dirs = [d.strip() for d in args.dirs.split(',') if d.strip()]
    gpus = [int(x) for x in args.gpus.split(',') if x != '']
    os.makedirs(fl.OUT_DIR, exist_ok=True)

    # 阈值来源: 显式 或 从历史指标反解
    if args.threshold is None:
        meta = None
        for fn in ('metrics_multi.json', 'metrics_single.json'):
            p = os.path.join(fl.OUT_DIR, fn)
            if os.path.exists(p):
                with open(p) as f:
                    meta = json.load(f)
                break
        assert meta is not None, '未找到 outputs/metrics_*.json, 请先运行 Step2/3'
        thr = None
        tpir_at = None
        for pt in meta['points']:
            if abs(pt['fpir'] - args.fpir) < 1e-12:
                thr, tpir_at = pt['threshold'], pt['tpir']
        assert thr is not None, 'metrics json 中无 FPIR=%.0e 点' % args.fpir
        tag = 'FPIR%.0e_t%.4f' % (args.fpir, thr)
        neg_above_est = int(round(args.fpir * meta['neg_pairs']))
        pos_below_est = int(round((1 - tpir_at) * meta['pos_pairs']))
    else:
        thr = float(args.threshold)
        tag = 't%.4f' % thr
        neg_above_est = pos_below_est = None
    print('=' * 78)
    print('Step4 样本对提取  阈值=%.6f 方向=%s 每向最多=%s GPU=%d张' %
          (thr, dirs, fl.fmt_pair(args.max), len(gpus)))
    print('=' * 78)
    feats, ids, paths, t_load = fl.load_data()
    N = len(ids)

    print('\n[遍历] 多卡二次遍历提取 (仅重新计算相似度, 不做直方图) ...')
    t0 = time.perf_counter()
    res = fl.run_extract_pass(gpus, args.block, N, thr, dirs, args.max)
    t_pass = time.perf_counter() - t0
    print('  遍历+合并耗时: %.2fs' % t_pass)

    print('\n[合并] 各卡局部top-K -> 全局 top-K')
    collected = {d: [] for d in dirs}
    for g, gd, _w in res:
        for d in dirs:
            if gd[d] is not None:
                collected[d].append(gd[d])
    saved = {}
    for d in dirs:
        if not collected[d]:
            print('  方向 %-5s: 0 对(未发现符合条件的样本对)' % d)
            continue
        rows = np.concatenate([c[0] for c in collected[d]])
        cols = np.concatenate([c[1] for c in collected[d]])
        sims = np.concatenate([c[2] for c in collected[d]])
        n_cand = len(sims)
        asc = (d == 'below')
        idx = np.argsort(sims) if asc else np.argsort(sims)[::-1]
        idx = idx[:args.max]
        rows, cols, sims = rows[idx], cols[idx], sims[idx]
        saved[d] = (rows, cols, sims)
        print('  方向 %-5s: 全局候选=%s → 保存 top-%s (%s)' %
              (d, fl.fmt_pair(n_cand), fl.fmt_pair(len(sims)),
               'sim最小' if asc else 'sim最大'))

    print('\n[保存]')
    out_files = []
    for d in dirs:
        if d not in saved:
            continue
        rows, cols, sims = saved[d]
        if args.indices:
            pairs = [(int(rows[k]), int(cols[k]), float(sims[k])) for k in range(len(rows))]
        else:
            pairs = [(paths[int(rows[k])], paths[int(cols[k])], float(sims[k]))
                     for k in range(len(rows))]
        fn = os.path.join(fl.OUT_DIR, 'pairs_%s_%s_n%d.pkl' % (d, tag, len(pairs)))
        data = dict(direction=d, threshold=thr, method='multi_gpu',
                    t_extract=t_pass, pairs=pairs)
        if d == 'above' and neg_above_est is not None:
            data['est_total_crossing'] = neg_above_est
        if d == 'below' and pos_below_est is not None:
            data['est_total_crossing'] = pos_below_est
        with open(fn, 'wb') as f:
            pickle.dump(data, f)
        out_files.append(fn)
        print('  %s → %s  (%s 对)' % (d, fn, fl.fmt_pair(len(pairs))))

    print('\n[示例] (前5/后3, 方向含义: above=负对误接受 sim>t, below=正对漏报 sim<t)')
    for d in dirs:
        if d not in saved:
            continue
        rows, cols, sims = saved[d]
        n = len(rows)
        print('  方向=%s 阈值=%.6f (共%s对):' % (d, thr, fl.fmt_pair(n)))
        show = list(range(min(5, n))) + (list(range(max(5, n - 3), n)) if n > 8 else [])
        for k in show:
            print('    sim=%.4f  %s' % (sims[k], paths[rows[k]]))
            print('             %s' % paths[cols[k]])
    print('\n完成: 提取耗时 %.2fs, 输出 %d 个文件' % (t_pass, len(out_files)))
    print('=' * 78)


if __name__ == '__main__':
    main()
