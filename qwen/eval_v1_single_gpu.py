#!/usr/bin/env python3
"""
第二步：v1 单卡基础实现
- 读取pkl特征
- 单GPU分块计算NxN相似度（仅上三角，i<j）
- 用直方图统计正/负样本相似度分布（内存友好，bin宽1e-6）
- 计算TPIR@FPIR指标并输出关键评估点
- 绘制相似度分布图 + TPIR@FPIR曲线
"""
import argparse
import os
import pickle
import time

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

FONT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'font', 'SourceHanSansSC-Normal.otf')


def setup_chinese_font():
    font_manager.fontManager.addfont(FONT_PATH)
    prop = font_manager.FontProperties(fname=FONT_PATH)
    plt.rcParams['font.family'] = prop.get_name()
    plt.rcParams['axes.unicode_minus'] = False


def compute_histograms_single_gpu(feats, ids, device, nbins=2_000_000, verbose=True):
    """单卡分块计算上三角相似度，直方图统计正/负样本分布。

    直方图 bin 覆盖 [-1, 1]，bin k 对应区间 [-1 + k*w, -1 + (k+1)*w)。
    """
    N = feats.shape[0]
    feats = feats.to(device)
    ids_dev = ids.to(device)

    hist_pos = torch.zeros(nbins, dtype=torch.long, device=device)
    hist_neg = torch.zeros(nbins, dtype=torch.long, device=device)

    scale = nbins / 2.0
    # 每块元素数控制在 ~1e8，避免OOM
    chunk = max(32, min(4096, int(1e8 // max(N, 1))))

    t0 = time.time()
    n_chunks = (N + chunk - 1) // chunk
    for ci, start in enumerate(range(0, N, chunk)):
        end = min(start + chunk, N)
        sim = feats[start:end] @ feats.t()                      # (c, N) fp32
        rows = torch.arange(start, end, device=device).unsqueeze(1)
        cols = torch.arange(N, device=device).unsqueeze(0)
        valid = cols > rows                                      # 上三角 i<j
        same = ids_dev[start:end].unsqueeze(1) == ids_dev.unsqueeze(0)

        idx = ((sim.clamp_(-1.0, 1.0) + 1.0) * scale).long().clamp_(0, nbins - 1)
        pos_idx = idx[valid & same]
        neg_idx = idx[valid & ~same]
        hist_pos += torch.bincount(pos_idx, minlength=nbins)
        hist_neg += torch.bincount(neg_idx, minlength=nbins)

        if verbose and (ci % 20 == 0 or ci == n_chunks - 1):
            elapsed = time.time() - t0
            print(f"  chunk {ci + 1}/{n_chunks}  已用 {elapsed:.1f}s", flush=True)
        del sim, valid, same, idx, pos_idx, neg_idx
    torch.cuda.synchronize(device)
    t_compute = time.time() - t0
    return hist_pos.cpu(), hist_neg.cpu(), t_compute


def tpir_at_fpir(hist_pos, hist_neg, target_fpirs):
    """从直方图右端累积，计算 TPIR@FPIR。

    cum_neg[k] = 负样本中落入 bin>=k 的数量 ≈ #(sim_neg >= left_edge(k))
    FPIR(bin k 左缘) = cum_neg[k]/Nneg, TPIR = cum_pos[k]/Npos
    """
    nbins = len(hist_pos)
    w = 2.0 / nbins
    edges = -1.0 + np.arange(nbins) * w  # 每个bin左缘

    cum_pos = np.cumsum(hist_pos[::-1])[::-1].astype(np.float64)
    cum_neg = np.cumsum(hist_neg[::-1])[::-1].astype(np.float64)
    n_pos, n_neg = hist_pos.sum(), hist_neg.sum()
    fpir = cum_neg / n_neg
    tpir = cum_pos / n_pos

    results = []
    for t in target_fpirs:
        # 找满足 FPIR <= t 的最大阈值bin（即最靠左的满足条件的bin）
        k = np.searchsorted(-fpir, -t)  # fpir 单调不增
        k = min(k, nbins - 1)
        results.append(dict(target_fpir=t, threshold=float(edges[k]),
                            fpir=float(fpir[k]), tpir=float(tpir[k])))
    return results, edges, fpir, tpir, n_pos, n_neg


def plot_results(hist_pos, hist_neg, curve, outdir, prefix='v1'):
    os.makedirs(outdir, exist_ok=True)
    nbins = len(hist_pos)
    w = 2.0 / nbins
    centers = -1.0 + (np.arange(nbins) + 0.5) * w
    n_pos, n_neg = hist_pos.sum(), hist_neg.sum()

    # 重采样直方图便于绘图（2e6 bin 画不了）
    factor = nbins // 2000
    hp = hist_pos[:factor * 2000].reshape(2000, factor).sum(1)
    hn = hist_neg[:factor * 2000].reshape(2000, factor).sum(1)
    xc = centers[:factor * 2000:factor]
    pdf_pos = hp / (n_pos * w * factor)
    pdf_neg = hn / (n_neg * w * factor)

    # 图1：相似度分布
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.plot(xc, pdf_neg, color='#d62728', lw=1.2, label=f'负样本对 ({n_neg:,})')
    ax.plot(xc, pdf_pos, color='#1f77b4', lw=1.2, label=f'正样本对 ({n_pos:,})')
    ax.set_yscale('log')
    ax.set_xlabel('余弦相似度')
    ax.set_ylabel('概率密度 (log)')
    ax.set_title('正负样本相似度分布（全范围）')
    ax.legend()
    ax.grid(alpha=0.3)
    ax = axes[1]
    m = xc >= 0.0
    ax.plot(xc[m], pdf_neg[m], color='#d62728', lw=1.2, label='负样本对')
    ax.plot(xc[m], pdf_pos[m], color='#1f77b4', lw=1.2, label='正样本对')
    ax.set_xlabel('余弦相似度')
    ax.set_ylabel('概率密度')
    ax.set_title('正负样本相似度分布（sim≥0 放大）')
    ax.legend()
    ax.grid(alpha=0.3)
    p1 = os.path.join(outdir, f'{prefix}_sim_distribution.png')
    fig.tight_layout()
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    # 图2：TPIR@FPIR 曲线
    edges, fpir, tpir = curve['edges'], curve['fpir'], curve['tpir']
    # 抽稀
    step = max(1, len(fpir) // 5000)
    fx, ty = fpir[::step], tpir[::step]
    keep = fx > 0
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fx[keep], ty[keep], color='#2ca02c', lw=1.5, label='TPIR@FPIR')
    for r in curve['key_points']:
        if r['fpir'] > 0:
            ax.scatter([r['fpir']], [r['tpir']], zorder=5, s=40)
            ax.annotate(f"FPIR={r['target_fpir']:.0e}\nTPIR={r['tpir'] * 100:.2f}%\nthr={r['threshold']:.4f}",
                        (r['fpir'], r['tpir']), textcoords='offset points', xytext=(10, -10), fontsize=9)
    ax.set_xscale('log')
    ax.set_xlim(1e-6, 1.0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('FPIR（假正例率，对数轴）')
    ax.set_ylabel('TPIR（真正例率）')
    ax.set_title('TPIR@FPIR 曲线')
    ax.grid(alpha=0.3, which='both')
    ax.legend()
    p2 = os.path.join(outdir, f'{prefix}_tpir_fpir.png')
    fig.tight_layout()
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    return p1, p2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', default='s4_0618_enhance.pkl')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--nbins', type=int, default=2_000_000)
    ap.add_argument('--outdir', default='results')
    args = ap.parse_args()

    setup_chinese_font()
    t_all = time.time()

    t0 = time.time()
    with open(args.pkl, 'rb') as f:
        feats, _flip, ids, file_paths = pickle.load(f)
    print(f"[加载] pkl 读取耗时 {time.time() - t0:.2f}s, N={len(ids)}, dim={feats.shape[1]}")

    feats = torch.from_numpy(np.ascontiguousarray(feats, dtype=np.float32))
    ids = torch.from_numpy(np.asarray(ids))
    device = torch.device(f'cuda:{args.gpu}')

    hist_pos, hist_neg, t_compute = compute_histograms_single_gpu(feats, ids, device, nbins=args.nbins)
    hist_pos_np = hist_pos.numpy()
    hist_neg_np = hist_neg.numpy()
    print(f"[计算] 单卡({args.gpu}) 相似度+直方图耗时 {t_compute:.2f}s")

    targets = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    key_points, edges, fpir, tpir, n_pos, n_neg = tpir_at_fpir(hist_pos_np, hist_neg_np, targets)
    print(f"[统计] 正样本对 {int(n_pos):,}，负样本对 {int(n_neg):,}")
    print("[指标] TPIR@FPIR:")
    for r in key_points:
        print(f"  FPIR<={r['target_fpir']:.0e}: TPIR={r['tpir'] * 100:.3f}%  "
              f"(实际FPIR={r['fpir']:.3e}, 阈值={r['threshold']:.4f})")

    p1, p2 = plot_results(hist_pos_np, hist_neg_np,
                          dict(edges=edges, fpir=fpir, tpir=tpir, key_points=key_points),
                          args.outdir, prefix='v1')
    print(f"[绘图] {p1}\n[绘图] {p2}")
    print(f"[总耗时] {time.time() - t_all:.2f}s")


if __name__ == '__main__':
    main()
