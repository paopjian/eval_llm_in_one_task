#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gpu_worker.py — GPU 计算子进程 (由 faireval_lib 编排启动)

模式:
  hist    : 统计本GPU分配行块的正/负相似度直方图 -> 写 npz(pos_hist, neg_hist)
  extract : 提取 above/below 阈值样本对 -> 写 npz(dir_rows/dir_cols/dir_sims)

数据源: --data-pkl (整库) 或 --feat-npz (子集, 含 feats/ids 键)
"""
import os
import sys
import time
import argparse
import numpy as np

FONT_DEP = None  # 不需要


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['hist', 'extract'], required=True)
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--block', type=int, default=8192)
    ap.add_argument('--nbins', type=int, default=200000)
    ap.add_argument('--row-blocks', type=str, required=True)
    ap.add_argument('--threshold', type=float, default=0.0)
    ap.add_argument('--dirs', type=str, default='above,below')
    ap.add_argument('--max', type=int, default=300000)
    ap.add_argument('--feat-npz', type=str, default=None)
    ap.add_argument('--data-pkl', type=str, default='s4_0618_enhance.pkl')
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()

    t_load = time.perf_counter()
    if args.feat_npz:
        d = np.load(args.feat_npz)
        feats, ids = np.ascontiguousarray(d['feats'], dtype=np.float32), d['ids']
    else:
        from faireval_lib import load_data
        feats, ids, _paths, _ = load_data(args.data_pkl)
    N = len(feats)
    t_load = time.perf_counter() - t_load

    # 组装本GPU的行块任务
    from faireval_lib import _mk_jobs
    jobs, bounds = _mk_jobs(N, args.block)
    blk_set = set(int(x) for x in args.row_blocks.split(',') if x != '')
    row_list = [j for j in jobs if j[0] in blk_set]

    import torch
    torch.set_grad_enabled(False)
    if args.mode == 'hist':
        from faireval_lib import _worker_hist_core
        r = _worker_hist_core(args.gpu, feats, ids, row_list, args.nbins)
        np.savez(args.out, pos_hist=r['pos'], neg_hist=r['neg'],
                 t_calc=np.float64(r['t_calc']), t_wall=np.float64(r['t_wall']))
    else:
        from faireval_lib import _worker_extract_core
        dirs = [x for x in args.dirs.split(',') if x]
        r = _worker_extract_core(args.gpu, feats, ids, row_list,
                                 args.threshold, dirs, args.max)
        out = dict(t_wall=np.float64(r['t_wall']))
        for d in dirs:
            v = r['extract'][d]
            if v is not None:
                out[d + '_rows'], out[d + '_cols'], out[d + '_sims'] = v
        np.savez(args.out, **out)
    print('[gpu%d] %s done in %.2fs (load %.2fs)' %
          (args.gpu, args.mode, time.perf_counter() - t_load, t_load))


if __name__ == '__main__':
    main()
