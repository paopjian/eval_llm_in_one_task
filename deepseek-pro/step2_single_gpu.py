#!/usr/bin/env python3
"""第二步：单卡分块相似度计算 + TPIR@FPIR 基线版本
- 仅使用 1 张 GPU (cuda:0)
- 分块计算上三角相似度（避免重复计算）
- 直方图统计（正样本单独在 CPU 精确计算，负样本 = 全体 - 正样本）
"""
import argparse
import os
import pickle
import time

import numpy as np
import torch

# 可复现
torch.manual_seed(0)
np.random.seed(0)


def load_data(path, use_cache):
    t0 = time.time()
    if use_cache and os.path.exists('feats_f32.npy') and os.path.exists('ids.npy'):
        F = np.load('feats_f32.npy')
        ids = np.load('ids.npy')
        paths = [p for p in open('paths.txt', encoding='utf-8').read().splitlines()]
    else:
        with open(path, 'rb') as f:
            F, _, ids, paths = pickle.load(f)
        F = np.asarray(F, dtype=np.float32)
        ids = np.asarray(ids, dtype=np.int64)
        paths = list(paths)
    return F, ids, paths, time.time() - t0


def block_rows(N, B):
    return [(s, min(s + B, N)) for s in range(0, N, B)]


def gpu_hist_pass(F_gpu, rows, B, scale):
    """在单卡上分块计算上三角相似度直方图（全体样本对）。"""
    dev = F_gpu.device
    nbins = 2 * scale + 1
    hist = torch.zeros(nbins, dtype=torch.int64, device=dev)
    nblocks = 0
    for i, (r0, r1) in enumerate(rows):
        A = F_gpu[r0:r1]
        for j in range(i, len(rows)):
            c0, c1 = rows[j]
            C = A @ F_gpu[c0:c1].T
            if i == j:
                mask = torch.triu(torch.ones(r1 - r0, c1 - c0, dtype=torch.bool, device=dev), 1)
                v = C[mask].reshape(-1)
            else:
                v = C.reshape(-1)
            q = (v * scale).round().clamp(-scale, scale).to(torch.int64) + scale
            hist += torch.bincount(q, minlength=nbins)
            nblocks += 1
    return hist.cpu().numpy(), nblocks


def pos_pass(F, ids, scale):
    """CPU 上按身份精确计算全部正样本对 (i<j) 的相似度。"""
    nbins = 2 * scale + 1
    hist = np.zeros(nbins, dtype=np.int64)
    order = np.argsort(ids, kind='stable')
    sids = ids[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sids)) + 1, len(ids)]
    total_pos = int(sum((e - s) * (e - s - 1) // 2 for s, e in zip(starts[:-1], starts[1:])))
    gi = np.empty(total_pos, np.int32)
    gj = np.empty(total_pos, np.int32)
    gs = np.empty(total_pos, np.float32)
    p = 0
    for s, e in zip(starts[:-1], starts[1:]):
        c = e - s
        if c < 2:
            continue
        pos = order[s:e]
        X = F[pos]
        S = X @ X.T
        iu, ju = np.triu_indices(c, 1)
        sims = S[iu, ju].astype(np.float32)
        n = len(sims)
        gi[p:p + n] = pos[iu]
        gj[p:p + n] = pos[ju]
        gs[p:p + n] = sims
        q = np.clip((sims * scale).round().astype(np.int64) + scale, 0, nbins - 1)
        hist += np.bincount(q, minlength=nbins)
        p += n
    return hist, gi, gj, gs


def compute_metrics(full_hist, pos_hist, pos_sims, targets, scale):
    nbins = 2 * scale + 1
    total_pairs = int(full_hist.sum())
    total_pos = len(pos_sims)
    total_neg = total_pairs - total_pos
    neg_hist = np.maximum(full_hist - pos_hist, 0)
    # cneg[k] = 相似度落在 bin k 及以上(>=k)的负样本对数
    cneg = neg_hist[::-1].cumsum()[::-1]
    pos_sorted = np.sort(pos_sims)
    results = []
    for tgt in targets:
        need = tgt * total_neg
        k = int(np.searchsorted(-cneg, -need, side='left'))
        k = min(max(k, 0), nbins - 1)
        th = (k - scale) / scale
        cnt_pos_above = total_pos - int(np.searchsorted(pos_sorted, th, side='right'))
        tpir = cnt_pos_above / total_pos
        fpir = cneg[k] / total_neg
        results.append(dict(target=tgt, threshold=float(th), fpir_actual=float(fpir),
                            tpir=float(tpir)))
    return results, neg_hist, cneg, total_pairs, total_pos, total_neg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='s4_0618_enhance.pkl')
    ap.add_argument('--use-cache', action='store_true')
    ap.add_argument('--block', type=int, default=8192)
    ap.add_argument('--bin-scale', type=int, default=20000)
    ap.add_argument('--fpir-targets', default='1e-5,1e-4,1e-3,1e-2')
    args = ap.parse_args()

    t_all = time.time()
    F, ids, paths, t_load = load_data(args.data, args.use_cache)
    N = len(ids)
    print(f'[1/4] 数据加载 {t_load:.2f}s  N={N:,}  特征={F.shape}')

    targets = [float(x) for x in args.fpir_targets.split(',')]
    rows = block_rows(N, args.block)
    print(f'[2/4] 单卡(cuda:0)分块上三角相似度计算  block={args.block}  {len(rows)} 行块')

    F_gpu = torch.from_numpy(F).cuda(0)
    torch.cuda.synchronize()
    t0 = time.time()
    full_hist, nblocks = gpu_hist_pass(F_gpu, rows, args.block, args.bin_scale)
    torch.cuda.synchronize()
    t_hist = time.time() - t0
    print(f'      GPU 直方图完成: {nblocks} 个块, 耗时 {t_hist:.2f}s')

    t0 = time.time()
    pos_hist, gi, gj, gs = pos_pass(F, ids, args.bin_scale)
    t_pos = time.time() - t0
    print(f'[3/4] 正样本对 CPU 精确计算 {len(gs):,} 对, 耗时 {t_pos:.2f}s')

    results, neg_hist, cneg, total_pairs, total_pos, total_neg = compute_metrics(
        full_hist, pos_hist, gs, targets, args.bin_scale)
    print(f'[4/4] TPIR@FPIR 结果 (总样本对 {total_pairs:,}, 正 {total_pos:,}, 负 {total_neg:,})')
    print(f'  校验: sum(full)={full_hist.sum():,}  sum(pos)={pos_hist.sum():,}')
    for r in results:
        print(f'  TPIR @ FPIR={r["target"]:.0e} : {r["tpir"]*100:.2f}%  '
              f'(阈值 {r["threshold"]:.4f}, 实际FPIR {r["fpir_actual"]:.2e})')
    print(f'总耗时: {time.time()-t_all:.2f}s')


if __name__ == '__main__':
    main()
