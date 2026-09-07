#!/usr/bin/env python3
"""
大规模人脸特征相似度评估：多卡分块计算 NxN 上三角相似度，
用直方图统计负样本、精确收集正样本，计算 TPIR@FPIR 并绘图。
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
from matplotlib import font_manager
from matplotlib.ticker import LogLocator, NullFormatter

PKL_DEFAULT = "s4_0618_enhance.pkl"
FONT_PATH = "font/SourceHanSansSC-Normal.otf"
HIST_CHUNK = 8_388_608  # 2^23，float32 histc 计数精确
VMIN = -1.0
VMAX = 1.0
TARGET_FPIRS = (1e-5, 1e-4, 1e-3, 1e-2)
NEG_TOPK_PER_GPU = 100000


def parse_args():
    p = argparse.ArgumentParser(description="人脸特征 TPIR@FPIR 多卡评估")
    p.add_argument("--pkl", default=PKL_DEFAULT, help="特征 pkl 路径")
    p.add_argument("--gpus", default="0,1,2,3,4,5,6", help="GPU 列表，逗号分隔")
    p.add_argument("--block-size", type=int, default=8192, help="分块边长")
    p.add_argument("--bins", type=int, default=100000, help="相似度直方图 bin 数")
    p.add_argument("--out-dir", default=".", help="图表与结果输出目录")
    p.add_argument(
        "--extract-threshold",
        type=float,
        default=None,
        help="提取样本对的相似度阈值；默认使用 TPIR@FPIR=1e-3 对应阈值",
    )
    p.add_argument(
        "--extract-max",
        type=int,
        default=20000,
        help="每类（FP/FN）最多保存的样本对数",
    )
    p.add_argument("--skip-extract", action="store_true", help="跳过样本对提取")
    p.add_argument("--skip-plot", action="store_true", help="跳过绘图")
    return p.parse_args()


def setup_chinese_font():
    if not os.path.isfile(FONT_PATH):
        print(f"[警告] 未找到中文字体: {FONT_PATH}")
        return None
    font_manager.fontManager.addfont(FONT_PATH)
    prop = font_manager.FontProperties(fname=FONT_PATH)
    plt.rcParams["font.family"] = prop.get_name()
    plt.rcParams["axes.unicode_minus"] = False
    return prop


def load_data(pkl_path):
    with open(pkl_path, "rb") as f:
        feats, _flip, ids, file_paths = pickle.load(f)
    feats = np.ascontiguousarray(feats, dtype=np.float32)
    ids = np.ascontiguousarray(ids, dtype=np.int64)
    return feats, ids, file_paths


def pair_stats(ids):
    n = int(ids.shape[0])
    total_pairs = n * (n - 1) // 2
    _unique, counts = np.unique(ids, return_counts=True)
    counts = counts.astype(np.int64)
    n_pos = int(np.sum(counts * (counts - 1) // 2))
    n_neg = total_pairs - n_pos
    return n, int(_unique.shape[0]), total_pairs, n_pos, n_neg, counts


def row_splits(n, n_gpu):
    """按上三角对数均分行区间，使各 GPU 工作量接近。"""
    total = n * (n - 1) // 2
    splits = [0]
    for g in range(1, n_gpu):
        target = total * g // n_gpu
        lo, hi = splits[-1], n
        while lo < hi:
            mid = (lo + hi) // 2
            if mid * (2 * n - mid - 1) // 2 >= target:
                hi = mid
            else:
                lo = mid + 1
        splits.append(int(lo))
    splits.append(n)
    return splits


def pairs_in_row_range(r0, r1, n):
    def cum(r):
        return r * (2 * n - r - 1) // 2

    return cum(r1) - cum(r0)


def configure_device(device_id):
    torch.cuda.set_device(device_id)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return torch.device(f"cuda:{device_id}")


def accumulate_hist(values, hist_i64, n_bins, vmin, vmax):
    if values.numel() == 0:
        return
    flat = values.reshape(-1)
    n = flat.numel()
    for s in range(0, n, HIST_CHUNK):
        e = min(s + HIST_CHUNK, n)
        h = torch.histc(flat[s:e], bins=n_bins, min=vmin, max=vmax)
        hist_i64.add_(h.to(torch.int64))


def compact_topk(buf_i, buf_j, buf_s, cap, largest=True):
    if not buf_s:
        return [], [], [], None
    i_cat = torch.cat(buf_i)
    j_cat = torch.cat(buf_j)
    s_cat = torch.cat(buf_s)
    k = min(cap, s_cat.numel())
    if s_cat.numel() > k:
        topv, topi = torch.topk(s_cat, k, largest=largest)
        i_cat, j_cat, s_cat = i_cat[topi], j_cat[topi], topv
    kth = float(s_cat.min().item()) if largest else float(s_cat.max().item())
    return [i_cat], [j_cat], [s_cat], kth


def gpu_hist_worker(device_id, feats_np, ids_np, r0, r1, n, n_bins, block, neg_cap):
    t0 = time.time()
    device = configure_device(device_id)
    feats = torch.from_numpy(feats_np).to(device, non_blocking=True)
    ids = torch.from_numpy(ids_np).to(device, non_blocking=True)
    torch.cuda.synchronize(device)
    hist = torch.zeros(n_bins, dtype=torch.int64, device=device)
    n_done = 0
    buf_i, buf_j, buf_s = [], [], []
    n_buf = 0
    dyn_thr = 0.25 if neg_cap > 0 else 2.0

    row_bs = block
    col_bs = block
    n_row_blocks = (max(r1 - r0, 0) + row_bs - 1) // row_bs if r1 > r0 else 0
    done_row_blocks = 0

    for rs in range(r0, r1, row_bs):
        re = min(rs + row_bs, r1)
        feat_r = feats[rs:re]
        ids_r = ids[rs:re]
        br = re - rs
        for cs in range(rs, n, col_bs):
            ce = min(cs + col_bs, n)
            feat_c = feats[cs:ce]
            ids_c = ids[cs:ce]
            sim = torch.mm(feat_r, feat_c.t())
            sim.clamp_(-1.0, 1.0)

            if cs >= re:
                accumulate_hist(sim, hist, n_bins, VMIN, VMAX)
                n_done += br * (ce - cs)
                valid = None
            else:
                i_idx = torch.arange(rs, re, device=device).unsqueeze(1)
                j_idx = torch.arange(cs, ce, device=device).unsqueeze(0)
                valid = i_idx < j_idx
                accumulate_hist(sim[valid], hist, n_bins, VMIN, VMAX)
                n_done += int(valid.sum().item())

            if neg_cap > 0:
                high = sim > dyn_thr
                if valid is not None:
                    high = high & valid
                # 身份区间无重叠时全是负样本，避免构造大 mask
                id_overlap = not (ids_r[-1] < ids_c[0] or ids_c[-1] < ids_r[0])
                if id_overlap:
                    high = high & (ids_r.unsqueeze(1) != ids_c.unsqueeze(0))
                if high.any():
                    ii, jj = torch.nonzero(high, as_tuple=True)
                    ss = sim[ii, jj]
                    buf_i.append(rs + ii)
                    buf_j.append(cs + jj)
                    buf_s.append(ss)
                    n_buf += int(ss.numel())
                    if n_buf > neg_cap * 2:
                        buf_i, buf_j, buf_s, dyn_thr = compact_topk(
                            buf_i, buf_j, buf_s, neg_cap, largest=True
                        )
                        n_buf = 0 if not buf_s else int(buf_s[0].numel())
            del sim

        done_row_blocks += 1
        if done_row_blocks in (1, n_row_blocks) or done_row_blocks % 2 == 0:
            print(
                f"[GPU {device_id}] 行块 {done_row_blocks}/{n_row_blocks}  "
                f"rows=[{rs},{re})  已统计对数={n_done:,}  用时={time.time()-t0:.1f}s",
                flush=True,
            )

    if neg_cap > 0 and buf_s:
        buf_i, buf_j, buf_s, _ = compact_topk(buf_i, buf_j, buf_s, neg_cap, largest=True)

    torch.cuda.synchronize(device)
    hist_np = hist.detach().cpu().numpy().copy()
    if buf_s:
        top_i = buf_i[0].detach().cpu().numpy().astype(np.int64)
        top_j = buf_j[0].detach().cpu().numpy().astype(np.int64)
        top_s = buf_s[0].detach().cpu().numpy().astype(np.float32)
    else:
        top_i = np.zeros((0,), dtype=np.int64)
        top_j = np.zeros((0,), dtype=np.int64)
        top_s = np.zeros((0,), dtype=np.float32)

    dt = time.time() - t0
    print(
        f"[GPU {device_id}] 完成 rows=[{r0},{r1})  对数={n_done:,}  "
        f"高分负对={top_s.size:,}  用时={dt:.2f}s",
        flush=True,
    )
    return {
        "device_id": device_id,
        "hist": hist_np,
        "n_done": int(n_done),
        "time": dt,
        "r0": r0,
        "r1": r1,
        "neg_i": top_i,
        "neg_j": top_j,
        "neg_s": top_s,
    }


def compute_positive_pairs(feats, ids, device_id=0):
    """按身份分组，批量计算所有正样本对 (i<j) 的相似度及原始下标。"""
    t0 = time.time()
    order = np.argsort(ids, kind="mergesort")
    sorted_ids = ids[order]
    unique_ids, counts = np.unique(sorted_ids, return_counts=True)
    counts = counts.astype(np.int64)
    n_id = int(unique_ids.shape[0])
    max_c = int(counts.max())

    padded = np.zeros((n_id, max_c, feats.shape[1]), dtype=np.float32)
    orig_idx = np.full((n_id, max_c), -1, dtype=np.int64)
    valid = np.zeros((n_id, max_c), dtype=bool)
    feats_sorted = feats[order]
    off = 0
    for i, c in enumerate(counts):
        c = int(c)
        padded[i, :c] = feats_sorted[off : off + c]
        orig_idx[i, :c] = order[off : off + c]
        valid[i, :c] = True
        off += c

    device = configure_device(device_id)
    ft = torch.from_numpy(padded).to(device)
    vm = torch.from_numpy(valid).to(device)
    sim = torch.bmm(ft, ft.transpose(1, 2)).clamp_(-1.0, 1.0)
    pair_mask = vm.unsqueeze(2) & vm.unsqueeze(1)
    tri = torch.triu(torch.ones((max_c, max_c), dtype=torch.bool, device=device), diagonal=1)
    pair_mask = pair_mask & tri.unsqueeze(0)
    pos_scores = sim[pair_mask].detach().cpu().numpy().astype(np.float32)

    id_ix, a_ix, b_ix = torch.nonzero(pair_mask, as_tuple=True)
    id_ix = id_ix.cpu().numpy()
    a_ix = a_ix.cpu().numpy()
    b_ix = b_ix.cpu().numpy()
    pos_i = orig_idx[id_ix, a_ix]
    pos_j = orig_idx[id_ix, b_ix]
    swap = pos_i > pos_j
    pos_i[swap], pos_j[swap] = pos_j[swap], pos_i[swap]

    dt = time.time() - t0
    print(
        f"[正样本] 身份数={n_id:,}  最大张数/人={max_c}  "
        f"正对数={pos_scores.size:,}  用时={dt:.2f}s",
        flush=True,
    )
    return pos_scores, pos_i, pos_j, dt


def scores_to_hist_torch(scores, n_bins, vmin=VMIN, vmax=VMAX):
    if scores.size == 0:
        return np.zeros(n_bins, dtype=np.int64)
    hist = torch.zeros(n_bins, dtype=torch.int64)
    accumulate_hist(torch.from_numpy(np.ascontiguousarray(scores)), hist, n_bins, vmin, vmax)
    return hist.numpy().astype(np.int64)


def hist_edges(n_bins, vmin=VMIN, vmax=VMAX):
    return np.linspace(vmin, vmax, n_bins + 1)


def count_above_hist(hist, threshold, vmin=VMIN, vmax=VMAX):
    n_bins = int(hist.shape[0])
    width = (vmax - vmin) / n_bins
    if threshold >= vmax:
        return 0.0
    if threshold < vmin:
        return float(hist.sum())
    idx = int(np.floor((threshold - vmin) / width))
    idx = min(max(idx, 0), n_bins - 1)
    lo = vmin + idx * width
    hi = lo + width
    frac = (hi - threshold) / width if width > 0 else 0.0
    frac = min(max(frac, 0.0), 1.0)
    return float(hist[idx + 1 :].sum()) + frac * float(hist[idx])


def threshold_from_all_hist(all_hist, pos_scores, n_neg, target_fpir):
    """二分阈值，使 (all>t - pos>t) / n_neg ≈ target_fpir。"""
    target = float(target_fpir) * float(n_neg)
    pos_sorted = np.sort(pos_scores)
    lo, hi = -1.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        all_ab = count_above_hist(all_hist, mid)
        pos_ab = pos_sorted.size - int(np.searchsorted(pos_sorted, mid, side="right"))
        neg_ab = all_ab - pos_ab
        if neg_ab > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def tpir_at_threshold(pos_scores, threshold):
    if pos_scores.size == 0:
        return 0.0
    return float(np.count_nonzero(pos_scores > threshold) / pos_scores.size)


def fpir_at_threshold(all_hist, pos_scores, n_neg, threshold):
    all_ab = count_above_hist(all_hist, threshold)
    pos_ab = np.count_nonzero(pos_scores > threshold)
    neg_ab = max(all_ab - pos_ab, 0.0)
    return float(neg_ab / n_neg) if n_neg else 0.0


def roc_from_hists(pos_hist, neg_hist, n_pos, n_neg, vmin=VMIN, vmax=VMAX):
    n_bins = int(neg_hist.shape[0])
    width = (vmax - vmin) / n_bins
    neg_rev = np.cumsum(neg_hist[::-1])[::-1]
    pos_rev = np.cumsum(pos_hist[::-1])[::-1]
    thresholds = vmin + np.arange(n_bins) * width
    fpir = neg_rev.astype(np.float64) / max(n_neg, 1)
    tpir = pos_rev.astype(np.float64) / max(n_pos, 1)
    return thresholds, fpir, tpir


def hist_mean(hist, n_bins):
    edges = hist_edges(n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    s = hist.sum()
    if s == 0:
        return 0.0
    return float((centers * hist).sum() / s)


def hist_quantile(hist, q, n_total):
    target = q * n_total
    csum = np.cumsum(hist)
    idx = int(np.searchsorted(csum, target, side="left"))
    idx = min(max(idx, 0), hist.shape[0] - 1)
    edges = hist_edges(hist.shape[0])
    return float(edges[idx])


def plot_distribution(pos_scores, neg_hist, n_neg, font_prop, out_path):
    edges = hist_edges(int(neg_hist.shape[0]))
    centers = 0.5 * (edges[:-1] + edges[1:])
    neg_pdf = neg_hist.astype(np.float64) / max(n_neg, 1)
    pos_hist, pos_edges = np.histogram(pos_scores, bins=400, range=(VMIN, VMAX), density=True)
    pos_centers = 0.5 * (pos_edges[:-1] + pos_edges[1:])
    bin_w = edges[1] - edges[0]
    neg_density = neg_pdf / bin_w

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(pos_centers, pos_hist, color="#1f77b4", lw=1.8, label="正样本对")
    ax.fill_between(pos_centers, pos_hist, alpha=0.25, color="#1f77b4")
    step = max(len(centers) // 2000, 1)
    ax.plot(centers[::step], neg_density[::step], color="#d62728", lw=1.8, label="负样本对")
    ax.fill_between(centers[::step], neg_density[::step], alpha=0.20, color="#d62728")
    ax.set_xlabel("余弦相似度", fontproperties=font_prop, fontsize=12)
    ax.set_ylabel("密度", fontproperties=font_prop, fontsize=12)
    ax.set_title("正负样本对相似度分布", fontproperties=font_prop, fontsize=14)
    ax.legend(prop=font_prop, fontsize=11)
    ax.set_xlim(-0.4, 1.0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[绘图] 相似度分布 -> {out_path}")


def plot_tpir_fpir(fpir, tpir, eval_points, font_prop, out_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    order = np.argsort(fpir)
    ax.plot(fpir[order], tpir[order], color="#2ca02c", lw=2.0, label="TPIR@FPIR 曲线")
    ax.set_xscale("log")
    ax.set_xlim(1e-6, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("FPIR", fontproperties=font_prop, fontsize=12)
    ax.set_ylabel("TPIR", fontproperties=font_prop, fontsize=12)
    ax.set_title("TPIR @ FPIR 曲线", fontproperties=font_prop, fontsize=14)
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=8))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="both", alpha=0.3)

    colors = ["#c0392b", "#e67e22", "#2980b9", "#8e44ad"]
    offsets = [(8, 10), (8, -32), (8, 10), (8, -32)]
    for (fpir_t, tpir_v, _thr), c, off in zip(eval_points, colors, offsets):
        ax.axvline(fpir_t, color=c, ls="--", lw=1.0, alpha=0.7)
        ax.scatter([fpir_t], [tpir_v], color=c, s=40, zorder=5)
        ax.annotate(
            f"FPIR={fpir_t:.0e}\nTPIR={tpir_v * 100:.2f}%",
            xy=(fpir_t, tpir_v),
            xytext=off,
            textcoords="offset points",
            fontsize=8,
            color=c,
            fontproperties=font_prop,
        )
    ax.legend(prop=font_prop, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[绘图] TPIR@FPIR 曲线 -> {out_path}")


def save_pair_file(path, ij, scores, ids, file_paths, pair_kind):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# kind={pair_kind}  count={ij.shape[0]}\n")
        f.write("i\tj\tsim\tid_i\tid_j\tpath_i\tpath_j\n")
        for k in range(ij.shape[0]):
            i, j = int(ij[k, 0]), int(ij[k, 1])
            f.write(
                f"{i}\t{j}\t{float(scores[k]):.6f}\t{int(ids[i])}\t{int(ids[j])}\t"
                f"{file_paths[i]}\t{file_paths[j]}\n"
            )
    print(f"[提取] {pair_kind} {ij.shape[0]:,} 对 -> {path}")


def select_pairs(mask_idx, scores, i_arr, j_arr, max_n, prefer_high):
    if mask_idx.size == 0:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    if mask_idx.size > max_n:
        order = np.argsort(scores[mask_idx])
        if prefer_high:
            order = order[::-1]
        mask_idx = mask_idx[order[:max_n]]
    return np.stack([i_arr[mask_idx], j_arr[mask_idx]], axis=1), scores[mask_idx]


def print_banner(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    gpus = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    if not gpus:
        raise RuntimeError("至少需要一张 GPU")
    n_gpu = len(gpus)
    neg_cap = 0 if args.skip_extract else NEG_TOPK_PER_GPU

    print_banner("数据加载与分析")
    t_load = time.time()
    feats, ids, file_paths = load_data(args.pkl)
    t_load = time.time() - t_load
    n, n_ids, total_pairs, n_pos, n_neg, counts = pair_stats(ids)
    norms = np.linalg.norm(feats, axis=1)
    print(f"  特征矩阵形状 : {feats.shape}  dtype={feats.dtype}")
    print(f"  样本总数     : {n:,}")
    print(f"  身份总数     : {n_ids:,}")
    print(f"  每人张数     : min={int(counts.min())}  max={int(counts.max())}  mean={counts.mean():.2f}")
    print(f"  总样本对数   : {total_pairs:,}")
    print(f"  正样本对数   : {n_pos:,}  ({n_pos / total_pairs * 100:.6f}%)")
    print(f"  负样本对数   : {n_neg:,}  ({n_neg / total_pairs * 100:.6f}%)")
    print(f"  L2 范数      : mean={norms.mean():.6f}  std={norms.std():.6f}")
    print(f"  是否归一化   : {'是' if abs(float(norms.mean()) - 1.0) < 0.01 else '否'}")
    print(f"  读取耗时     : {t_load:.2f}s")
    print(f"  GPU 列表     : {gpus}")
    print(f"  分块大小     : {args.block_size}")
    print(f"  直方图 bins  : {args.bins}")

    splits = row_splits(n, n_gpu)
    print("\n  各 GPU 行划分（上三角负载均衡）:")
    for g, r0, r1 in zip(gpus, splits[:-1], splits[1:]):
        pc = pairs_in_row_range(r0, r1, n)
        print(f"    cuda:{g}  rows=[{r0:6d}, {r1:6d})  pairs={pc:,}  ({pc / total_pairs * 100:.2f}%)")

    # 预热 CUDA 上下文，避免首个 kernel 抖动
    for g in gpus:
        torch.zeros(1, device=f"cuda:{g}")

    print_banner("正样本对计算（按身份批量）")
    pos_scores, pos_i, pos_j, t_pos = compute_positive_pairs(feats, ids, device_id=gpus[0])
    if pos_scores.size != n_pos:
        print(f"[警告] 正样本数量不一致: 得到 {pos_scores.size:,}  期望 {n_pos:,}")

    print_banner(f"多卡上三角相似度直方图（{n_gpu} x GPU, 线程并行）")
    t_hist = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=n_gpu) as ex:
        futs = [
            ex.submit(
                gpu_hist_worker,
                dev,
                feats,
                ids,
                r0,
                r1,
                n,
                args.bins,
                args.block_size,
                neg_cap,
            )
            for dev, r0, r1 in zip(gpus, splits[:-1], splits[1:])
        ]
        for fut in as_completed(futs):
            results.append(fut.result())
    t_hist = time.time() - t_hist
    results.sort(key=lambda x: x["device_id"])

    all_hist = np.zeros(args.bins, dtype=np.int64)
    n_hist_pairs = 0
    print("\n  各卡统计:")
    for r in results:
        h = np.asarray(r["hist"], dtype=np.int64)
        all_hist += h
        n_hist_pairs += r["n_done"]
        print(
            f"    cuda:{r['device_id']}  pairs={r['n_done']:,}  "
            f"hist_sum={int(h.sum()):,}  time={r['time']:.2f}s"
        )

    pos_hist = scores_to_hist_torch(pos_scores, args.bins)
    neg_hist = all_hist - pos_hist
    if np.any(neg_hist < 0):
        deficit = int((-neg_hist[neg_hist < 0]).sum())
        print(f"[提示] 正负样本 bin 边界差 {deficit}，已裁剪（相对 206 亿对可忽略）")
        neg_hist = np.clip(neg_hist, 0, None)

    print(f"\n  直方图合计对数 : {int(all_hist.sum()):,}  (扫描计数 {n_hist_pairs:,})")
    print(f"  期望总对数     : {total_pairs:,}")
    print(f"  正样本 hist    : {int(pos_hist.sum()):,} / {n_pos:,}")
    print(f"  负样本 hist    : {int(neg_hist.sum()):,} / {n_neg:,}")
    print(f"  多卡直方图耗时 : {t_hist:.2f}s")

    print_banner("正负样本相似度统计")
    print(
        f"  正样本 sim : mean={pos_scores.mean():.4f}  std={pos_scores.std():.4f}  "
        f"min={pos_scores.min():.4f}  max={pos_scores.max():.4f}  "
        f"p50={np.median(pos_scores):.4f}"
    )
    print(
        f"  负样本 sim : mean={hist_mean(neg_hist, args.bins):.4f}  "
        f"p50≈{hist_quantile(neg_hist, 0.5, n_neg):.4f}  "
        f"p99≈{hist_quantile(neg_hist, 0.99, n_neg):.4f}  "
        f"p99.9≈{hist_quantile(neg_hist, 0.999, n_neg):.4f}"
    )

    print_banner("TPIR @ FPIR 评估")
    eval_points = []
    for fpir_t in TARGET_FPIRS:
        thr = threshold_from_all_hist(all_hist, pos_scores, n_neg, fpir_t)
        tpir = tpir_at_threshold(pos_scores, thr)
        fpir_hat = fpir_at_threshold(all_hist, pos_scores, n_neg, thr)
        eval_points.append((fpir_t, tpir, thr))
        print(
            f"  TPIR @ FPIR={fpir_t:.0e} : {tpir * 100:7.3f}%    "
            f"threshold={thr:.6f}    (反推 FPIR={fpir_hat:.3e})"
        )

    expected = {1e-5: (0.60, 0.65), 1e-4: (0.82, 0.85), 1e-3: (0.90, 0.93), 1e-2: (0.95, 0.97)}
    print("\n  与预期范围对照:")
    in_range = True
    for fpir_t, tpir, _thr in eval_points:
        lo, hi = expected[fpir_t]
        ok = (lo - 0.05) <= tpir <= (hi + 0.05)
        in_range = in_range and ok
        flag = "OK" if ok else "超出容差"
        print(f"    FPIR={fpir_t:.0e}  得到 {tpir * 100:.2f}%  预期 {lo * 100:.0f}-{hi * 100:.0f}%  [{flag}]")

    roc_thr, roc_fpir, roc_tpir = roc_from_hists(pos_hist, neg_hist, n_pos, n_neg)

    font_prop = setup_chinese_font()
    dist_path = os.path.join(args.out_dir, "similarity_distribution.png")
    curve_path = os.path.join(args.out_dir, "tpir_fpir_curve.png")
    if not args.skip_plot:
        print_banner("绘制评估图表")
        plot_distribution(pos_scores, neg_hist, n_neg, font_prop, dist_path)
        plot_tpir_fpir(roc_fpir, roc_tpir, eval_points, font_prop, curve_path)

    extract_thr = args.extract_threshold
    if extract_thr is None:
        extract_thr = eval_points[2][2]
    fn_path = os.path.join(args.out_dir, "pairs_fn_below_threshold.tsv")
    fp_path = os.path.join(args.out_dir, "pairs_fp_above_threshold.tsv")
    pos_above_path = os.path.join(args.out_dir, "pairs_pos_above_threshold.tsv")

    t_ex = 0.0
    if not args.skip_extract:
        print_banner(f"提取 above/below 阈值样本对  (threshold={extract_thr:.6f})")
        t_ex = time.time()
        pos_above_idx = np.nonzero(pos_scores > extract_thr)[0]
        pos_below_idx = np.nonzero(pos_scores <= extract_thr)[0]
        print(f"  正样本 above={pos_above_idx.size:,}  below={pos_below_idx.size:,}")

        pa_ij, pa_s = select_pairs(pos_above_idx, pos_scores, pos_i, pos_j, args.extract_max, prefer_high=False)
        fn_ij, fn_s = select_pairs(pos_below_idx, pos_scores, pos_i, pos_j, args.extract_max, prefer_high=True)
        save_pair_file(pos_above_path, pa_ij, pa_s, ids, file_paths, "pos_above")
        save_pair_file(fn_path, fn_ij, fn_s, ids, file_paths, "pos_below_FN")

        neg_i = np.concatenate([r["neg_i"] for r in results])
        neg_j = np.concatenate([r["neg_j"] for r in results])
        neg_s = np.concatenate([r["neg_s"] for r in results])
        fp_mask = neg_s > extract_thr
        fp_idx = np.nonzero(fp_mask)[0]
        print(f"  缓存高分负对 {neg_s.size:,}，其中 sim>threshold 的 FP={fp_idx.size:,}")
        fp_ij, fp_s = select_pairs(fp_idx, neg_s, neg_i, neg_j, args.extract_max, prefer_high=True)
        save_pair_file(fp_path, fp_ij, fp_s, ids, file_paths, "neg_above_FP")
        t_ex = time.time() - t_ex
        print(f"  样本对提取耗时: {t_ex:.2f}s")

    metrics = {
        "n": n,
        "n_ids": n_ids,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "total_pairs": total_pairs,
        "hist_pairs": int(all_hist.sum()),
        "pos_sim_mean": float(pos_scores.mean()),
        "pos_sim_std": float(pos_scores.std()),
        "neg_sim_mean": hist_mean(neg_hist, args.bins),
        "tpir_fpir": [{"fpir": f, "tpir": t, "threshold": thr} for f, t, thr in eval_points],
        "extract_threshold": float(extract_thr),
        "time": {
            "load": t_load,
            "pos": t_pos,
            "hist": t_hist,
            "extract": t_ex,
            "total": t_load + t_pos + t_hist + t_ex,
        },
        "in_expected_range": bool(in_range),
        "plots": {"distribution": dist_path, "curve": curve_path},
    }
    metrics_path = os.path.join(args.out_dir, "eval_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print_banner("完成")
    print(f"  指标文件 : {metrics_path}")
    print(f"  总耗时   : {metrics['time']['total']:.2f}s  (直方图 {t_hist:.2f}s)")
    print(f"  处理速度 : {total_pairs / max(t_hist, 1e-6) / 1e9:.2f} G pairs/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
