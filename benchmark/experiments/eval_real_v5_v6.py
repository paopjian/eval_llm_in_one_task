#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实数据集 v5 vs v6 对比评估（临时脚本，不计入文档）:
    - 速度、直方图、TPIR@FPIR 指标对比
    - 提取负样本>0.5 / 正样本<0.2 的对比对，对比 v5/v6 差异
用法: python eval_real_v5_v6.py --data <pkl> --outdir <dir> [--max-n N] [--max-pairs K]
"""
import argparse
import importlib.util
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'benchmark'))

from benchmark import common  # noqa: E402
from cluster_utils_v6 import get_sim_matrix_large_scale_v6  # noqa: E402

# v5（benchmark/cluster_utils.py，重依赖，importlib 加载）
_cu_spec = importlib.util.spec_from_file_location(
    'cluster_utils', os.path.join(REPO_ROOT, 'benchmark', 'cluster_utils.py'))
cu = importlib.util.module_from_spec(_cu_spec)
_cu_spec.loader.exec_module(cu)
get_sim_matrix_large_scale_v5 = cu.get_sim_matrix_large_scale_v5

COLLECT = {
    'pos': {'threshold_mode': 'below', 'threshold': 0.2, 'max_pairs': -1},
    'neg': {'threshold_mode': 'above', 'threshold': 0.5, 'max_pairs': -1},
}


def tail_count(hist, lo=-1.0, hi=1.0, bins=200_000):
    """hist 中 sim > 0.5 与 sim < 0.2 的计数（估算口径）"""
    w = (hi - lo) / bins
    n_gt05 = int(hist[int((0.5 - lo) / w):].sum())
    n_lt02 = int(hist[:int((0.2 - lo) / w)].sum())
    return n_gt05, n_lt02


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--outdir', default='/tmp/eval_real_v5_v6')
    ap.add_argument('--max-n', type=int, default=0)
    ap.add_argument('--max-pairs', type=int, default=-1)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    with open(args.data, 'rb') as f:
        feats, feats_flip, ids, paths = pickle.load(f)
    feats = np.ascontiguousarray(feats, dtype=np.float32)
    ids = np.ascontiguousarray(ids, dtype=np.int64)
    if args.max_n:
        feats, ids = feats[:args.max_n], ids[:args.max_n]
    N = len(ids)
    gpus = list(range(torch.cuda.device_count()))
    # 特征已 L2 归一化（提取时做过），复检并确保
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    feats = feats / norms
    del norms

    name = os.path.basename(args.data)
    total = N * (N - 1) // 2
    _, cnt = np.unique(ids, return_counts=True)
    pos_theory = int((cnt.astype(np.int64) * (cnt - 1) // 2).sum())
    print(f'\n{name}: N={N:,} 总对={total:,} 正={pos_theory:,} 负={total-pos_theory:,}', flush=True)

    res = {'dataset': name, 'N': int(N), 'total_pairs': int(total),
           'pos_pairs': int(pos_theory)}

    # ================= v5 (fp32) 无收集 =================
    print('\n----- v5 (fp32, 200K bins, 无收集) -----', flush=True)
    torch.backends.cuda.matmul.allow_tf32 = False   # 显式关 TF32（v5 worker 对 fp32 不显式设置，避免继承上次 tf32）
    t0 = time.perf_counter()
    p5, n5 = get_sim_matrix_large_scale_v5(
        query_feats_list=feats, query_ids=ids, num_gpus=len(gpus),
        block_size=2048 * 5, hist_bins=200_000, hist_range=(-1.0, 1.0),
        collect_pairs_config=None, memory_mode='low_memory',
        show_progress=False, precision='fp32')
    w5 = time.perf_counter() - t0
    print(f'v5: wall={w5:.2f}s  pos_sum={int(p5.sum()):,}  neg_sum={int(n5.sum()):,}', flush=True)
    res['v5_wall_s'] = round(w5, 2)
    res['v5_tail_neg_gt05'] = tail_count(n5)[0]
    res['v5_tail_pos_lt02'] = tail_count(p5)[1]

    # ================= v6 (tf32) 无收集 =================
    print('\n----- v6 (tf32, 200K bins, 无收集) -----', flush=True)
    t0 = time.perf_counter()
    p6, n6 = get_sim_matrix_large_scale_v6(
        query_feats_list=feats, query_ids=ids, num_gpus=len(gpus),
        block_size=16384, hist_bins=200_000, hist_range=(-1.0, 1.0),
        collect_pairs_config=None, memory_mode='low_memory',
        show_progress=False, precision='tf32')
    w6 = time.perf_counter() - t0
    print(f'v6: wall={w6:.2f}s  pos_sum={int(p6.sum()):,}  neg_sum={int(n6.sum()):,}', flush=True)
    res['v6_wall_s'] = round(w6, 2)
    res['speedup'] = round(w5 / w6, 3)

    # ================= 直方图 / 指标对比 =================
    d = int(abs(p5.astype('int64') - p6.astype('int64')).sum()) + \
        int(abs(n5.astype('int64') - n6.astype('int64')).sum())
    same = bool((p5 == p6).all() and (n5 == n6).all())
    m5 = common.compute_metrics(p5, n5)
    m6 = common.compute_metrics(p6, n6)
    maxbin = max(abs(a['threshold'] - b['threshold']) / common.W
                 for a, b in zip(m5, m6))
    print(f'\n[直方图] 逐位相同={same}  总计数差={d:,}  阈值最大差={maxbin:.2f} bin', flush=True)
    print(f'[TPIR@FPIR]', flush=True)
    for a, b in zip(m5, m6):
        print(f"  FPIR={a['fpir']:.0e}  v5阈值={a['threshold']:.6f} v6阈值={b['threshold']:.6f} "
              f"| v5 TPIR={a['tpir']*100:.4f}%  v6 TPIR={b['tpir']*100:.4f}%", flush=True)
    res['hist_identical'] = same
    res['hist_total_diff'] = int(d)
    res['max_threshold_bin_diff'] = round(float(maxbin), 2)
    res['v5_metrics'] = m5
    res['v6_metrics'] = m6

    gt05_5, lt02_5 = tail_count(n5)[0], tail_count(p5)[1]
    gt05_6, lt02_6 = tail_count(n6)[0], tail_count(p6)[1]
    print(f'\n[尾部计数] neg>0.5: v5={gt05_5:,} v6={gt05_6:,} | '
          f'pos<0.2: v5={lt02_5:,} v6={lt02_6:,}', flush=True)
    res['tail_neg_gt05'] = {'v5': int(gt05_5), 'v6': int(gt05_6)}
    res['tail_pos_lt02'] = {'v5': int(lt02_5), 'v6': int(lt02_6)}

    # ================= 提取对比对 =================
    print(f'\n----- 提取对比对 (neg>0.5, pos<0.2, max_pairs={args.max_pairs}) -----', flush=True)
    cfg = {
        'pos': {'threshold_mode': 'below', 'threshold': 0.2, 'max_pairs': args.max_pairs},
        'neg': {'threshold_mode': 'above', 'threshold': 0.5, 'max_pairs': args.max_pairs},
    }

    t0 = time.perf_counter()
    torch.backends.cuda.matmul.allow_tf32 = False   # v5 fp32 显式关 TF32
    _, _, pos5, neg5 = get_sim_matrix_large_scale_v5(
        query_feats_list=feats, query_ids=ids, num_gpus=len(gpus),
        block_size=2048 * 5, hist_bins=200_000, hist_range=(-1.0, 1.0),
        collect_pairs_config=cfg, memory_mode='low_memory',
        show_progress=False, precision='fp32')
    w5c = time.perf_counter() - t0
    print(f'v5 提取: pos={len(pos5):,} neg={len(neg5):,} wall={w5c:.2f}s', flush=True)

    t0 = time.perf_counter()
    _, _, pos6, neg6 = get_sim_matrix_large_scale_v6(
        query_feats_list=feats, query_ids=ids, num_gpus=len(gpus),
        block_size=16384, hist_bins=200_000, hist_range=(-1.0, 1.0),
        collect_pairs_config=cfg, memory_mode='low_memory',
        show_progress=False, precision='tf32')
    w6c = time.perf_counter() - t0
    print(f'v6 提取: pos={len(pos6):,} neg={len(neg6):,} wall={w6c:.2f}s', flush=True)

    res['v5_collect_wall'] = round(w5c, 2)
    res['v6_collect_wall'] = round(w6c, 2)

    def pair_set(pairs):
        return {(int(i), int(j)) for (i, j, s) in pairs}

    for label, a, b in (('pos<0.2', pos5, pos6), ('neg>0.5', neg5, neg6)):
        sa, sb = pair_set(a), pair_set(b)
        inter = len(sa & sb)
        only_a = len(sa - sb)
        only_b = len(sb - sa)
        # 分数差（交集上）
        da = { (int(i), int(j)): float(s) for (i, j, s) in a }
        db = { (int(i), int(j)): float(s) for (i, j, s) in b }
        diffs = [abs(da[k] - db[k]) for k in (sa & sb)]
        print(f'\n[{label}] v5={len(a):,} v6={len(b):,} 交集={inter:,} '
              f'仅v5={only_a:,} 仅v6={only_b:,}', flush=True)
        if diffs:
            print(f'  交集内分数差: max={max(diffs):.2e} mean={np.mean(diffs):.2e}', flush=True)
        res[f'{label}_v5_count'] = len(a)
        res[f'{label}_v6_count'] = len(b)
        res[f'{label}_intersection'] = inter
        res[f'{label}_only_v5'] = only_a
        res[f'{label}_only_v6'] = only_b
        res[f'{label}_max_score_diff'] = float(max(diffs)) if diffs else 0.0

    # 保存样本（前 100 条）
    res['sample_pos_v6'] = [(int(i), int(j), float(s)) for (i, j, s) in pos6[:100]]
    res['sample_neg_v6'] = [(int(i), int(j), float(s)) for (i, j, s) in neg6[:100]]

    # 保存 v6 提取的完整对比对（npz: i/j int64, score float32）
    def save_pairs(path, pairs):
        if not pairs:
            return
        arr = np.array([(int(i), int(j), float(s)) for (i, j, s) in pairs],
                       dtype=[('i', np.int64), ('j', np.int64), ('score', np.float32)])
        np.savez_compressed(path, i=arr['i'], j=arr['j'], score=arr['score'])
        print(f'  已保存 {len(pairs):,} 对 -> {path}', flush=True)

    pfx = os.path.join(args.outdir, f'{name}.neg_gt05')
    save_pairs(pfx + '.v6.npz', neg6)
    save_pairs(os.path.join(args.outdir, f'{name}.pos_lt02.v6.npz'), pos6)

    with open(os.path.join(args.outdir, f'{name}.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=str)
    print(f'\n结果已保存: {os.path.join(args.outdir, name + ".json")}', flush=True)


if __name__ == '__main__':
    main()
