#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一评估公共实现 —— 所有模型的读取/校验/指标代码完全相同，只有 cores/ 中的核心计算不同。

统一约定:
    * 直方图网格: 200,000 bins 覆盖 [-1, 1]（bin 宽 1e-5），所有 core 输出同一网格
    * core 契约: compute(feats, ids, gpus, workdir) -> (pos_hist, neg_hist, meta)
      - feats:  (N,512) float32，已 L2 归一化
      - ids:    (N,) int64（原始身份标签，非连续编码亦可，内部统计用 unique）
      - pos_hist / neg_hist: (200000,) int64，边 k 覆盖 [lo + k*w, lo + (k+1)*w)
    * 校验: pos_hist 计数 == 理论正样本对数；neg_hist 计数 == 理论负样本对数
    * 指标: TPIR@FPIR (1e-5, 1e-4, 1e-3, 1e-2)，由直方图尾部累计求阈值
"""
import math
import pickle
import time

import numpy as np

# ---------------------------------------------------------------- 统一网格
BINS = 200_000
LO, HI = -1.0, 1.0
W = (HI - LO) / BINS                     # 1e-5
FPIRS = (1e-5, 1e-4, 1e-3, 1e-2)


def edge(k: int) -> float:
    """第 k 个 bin 的左边缘（k in [0, BINS]）"""
    return LO + k * W


# ---------------------------------------------------------------- 数据读取（全模型一致）
def load_data(pkl_path: str):
    """读取测试数据 pkl: (feats, _, ids, _) 四元组，返回 L2 归一化后的 (feats, ids)"""
    t0 = time.time()
    with open(pkl_path, 'rb') as f:
        feats, _flip, ids, _paths = pickle.load(f)
    feats = np.ascontiguousarray(feats, dtype=np.float32)
    ids = np.ascontiguousarray(ids, dtype=np.int64)
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    feats = feats / norms
    print(f"[加载] {pkl_path} N={len(ids):,} d={feats.shape[1]} 耗时 {time.time()-t0:.1f}s", flush=True)
    return feats, ids


def pair_stats(ids):
    """返回 (总样本对数, 正样本对数, 负样本对数)（i<j 且同身份为正）"""
    _, counts = np.unique(ids, return_counts=True)
    total_pos = int((counts.astype(np.int64) * (counts - 1) // 2).sum())
    total = len(ids) * (len(ids) - 1) // 2
    return total, total_pos, total - total_pos


# ---------------------------------------------------------------- 直方图工具（core 可调用）
def gpu_hist(values, device=None, bins=BINS):
    """GPU 直方图统计（torch.histc 语义），返回 int64 张量。供 core 内部使用。"""
    import torch
    t = values if not isinstance(values, torch.Tensor) else values
    if not isinstance(t, torch.Tensor):
        t = torch.from_numpy(np.ascontiguousarray(values)).to(device)
    h = torch.histc(t, bins=bins, min=LO, max=HI)
    return h.round().to(torch.int64)


def cpu_hist(values, bins=BINS):
    """CPU 直方图统计（np.histogram 语义），返回 int64 数组。供 core 内部收集正样本等使用。"""
    return np.histogram(values, bins=bins, range=(LO, HI))[0].astype(np.int64)


def rebin(hist, factor):
    """把高分辨率直方图按 factor 聚合到统一网格（factor 须整除）。"""
    n = len(hist)
    assert n % factor == 0, f"cannot rebin {n} bins by {factor}"
    return hist.reshape(n // factor, factor).sum(axis=1)


# ---------------------------------------------------------------- 校验（全模型一致）
def validate_hists(pos_hist, neg_hist, ids, rel_tol=1e-5, abs_min=10):
    """校验直方图计数与理论值一致。

    允许极小偏差（rel_tol 相对容差 / abs_min 绝对下限，取大者）——某些模型实现
    在分带/分块边界存在 ±几十对的固有微小误差（如 glm-pro），用户确认“误差不大即可”。
    ok=True 表示在容差内；detail['strict_ok'] 表示逐对精确一致；delta 记录偏差。
    """
    total, pos, neg = pair_stats(ids)
    got_pos, got_neg = int(pos_hist.sum()), int(neg_hist.sum())
    issues = []
    tol_pos = max(abs_min, rel_tol * pos)
    tol_neg = max(abs_min, rel_tol * neg)
    d_pos = got_pos - pos
    d_neg = got_neg - neg
    if len(pos_hist) != BINS or len(neg_hist) != BINS:
        issues.append(f"bins={len(pos_hist)}/{len(neg_hist)} != {BINS}")
    if abs(d_pos) > tol_pos:
        issues.append(f"pos {got_pos:,} != 理论 {pos:,} (差 {d_pos:,} > 容差 {tol_pos:.0f})")
    if abs(d_neg) > tol_neg:
        issues.append(f"neg {got_neg:,} != 理论 {neg:,} (差 {d_neg:,} > 容差 {tol_neg:.0f})")
    if got_pos + got_neg != total and abs(got_pos + got_neg - total) > max(abs_min, rel_tol * total):
        issues.append(f"total {got_pos+got_neg:,} != 理论 {total:,}")
    if int(neg_hist.min()) < 0 or int(pos_hist.min()) < 0:
        issues.append("直方图出现负计数")
    strict_ok = (not issues and d_pos == 0 and d_neg == 0)
    return (not issues), {'total': total, 'pos': pos, 'neg': neg,
                          'got_total': got_pos + got_neg, 'got_pos': got_pos,
                          'got_neg': got_neg, 'delta_pos': d_pos, 'delta_neg': d_neg,
                          'strict_ok': strict_ok, 'rel_tol': rel_tol, 'issues': issues}


# ---------------------------------------------------------------- 指标（全模型一致）
def _neg_tail(neg_counts):
    """S[k] = 相似度 >= edge(k) 的负样本计数（含 bin k），尾部累计"""
    return np.cumsum(neg_counts[::-1])[::-1]


def threshold_bin_at_fpir(neg_counts, total_neg, fpir):
    """求最小 bin 序号 k：S[k] <= ceil(fpir * total_neg)，阈值取 edge(k)"""
    S = _neg_tail(neg_counts)
    target = max(1, math.ceil(fpir * total_neg))
    idx = np.nonzero(S <= target)[0]
    return int(idx[0]) if len(idx) else BINS - 1


def compute_metrics(pos_hist, neg_hist):
    """由统一网格直方图计算 TPIR@FPIR 各档。所有模型调用同一实现。"""
    pos_hist = np.asarray(pos_hist, dtype=np.int64)
    neg_hist = np.asarray(neg_hist, dtype=np.int64)
    total_neg = int(neg_hist.sum())
    total_pos = int(pos_hist.sum())
    points = []
    for f in FPIRS:
        k = threshold_bin_at_fpir(neg_hist, total_neg, f)
        t = edge(k)
        fpir_hat = float(neg_hist[k:].sum()) / total_neg          # >= t 的负样本占比
        above_pos = int(pos_hist[k + 1:].sum())                   # > t 的正样本计数
        tpir = above_pos / total_pos if total_pos else 0.0
        points.append({'fpir': f, 'threshold': round(float(t), 6),
                       'actual_fpir': round(fpir_hat, 8), 'tpir': round(tpir, 10),
                       'pos_above': above_pos})
    return points


def fmt_tpir(points):
    lines = []
    for p in points:
        lines.append(f"  FPIR={p['fpir']:>7.0e} 阈值={p['threshold']:.6f} "
                     f"实际FPIR={p['actual_fpir']:.2e} TPIR={p['tpir']*100:.4f}%")
    return '\n'.join(lines)
