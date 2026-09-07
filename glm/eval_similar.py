#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人脸特征大规模相似度评估系统（多GPU并行）
==========================================================================
数据规模: N=203,234 → 总样本对 206.5亿（相似度矩阵全量需 ~77GB，不可存储）

核心设计（用统计代替全量保存）:
  1. 特征常驻各 GPU（fp32 精确计算，关闭 TF32），分块 block×block 计算
     上三角余弦相似度（特征已 L2 归一化，相似度 = 矩阵乘法）
  2. 相似度不全量保存: GPU 端实时统计进 2,000,000 bins 直方图（bin 宽 1e-6）
     负样本直方图 = 全体直方图 - 正样本直方图
  3. 正样本对仅 218 万 → 精确收集全部正样本相似度（~8MB）→ TPIR 精确计算
  4. FPIR 对应阈值由负样本直方图尾部累计计数求得（阈值分辨率 1e-6）
  5. 提取模式: 二次扫描，提取 above/below 阈值样本对（top-K + 精确计数），
     worker 仅回传全局索引 (i, j)，由主进程映射回文件路径，避免大体积传输
  6. 多卡并行: tile 任务队列动态分配（天然负载均衡），每 GPU 一个进程

用法:
  python eval_similar.py                          # 统计 + TPIR@FPIR + 绘图
  python eval_similar.py --extract-fpir 1e-3,1e-4 # 提取 FPIR 阈值之上负样本对
  python eval_similar.py --extract-tpir 0.95      # 提取 TPIR 阈值之下正样本对
  python eval_similar.py --extract-thresh 0.5     # 按原始阈值提取 above/below
  python eval_similar.py --gpus 0,1,2 --block 16384
"""
import argparse
import csv
import json
import math
import os
import pickle
import time

import numpy as np
import torch
import torch.multiprocessing as mp

PKL_PATH = 's4_0618_enhance.pkl'
FONT_PATH = 'font/SourceHanSansSC-Normal.otf'
LO, HI = -1.0, 1.0  # 余弦相似度直方图范围


# ---------------------------------------------------------------- 数据加载
def load_data(pkl_path):
    """读取 pkl，返回归一化特征、连续编码 id、原始 id、文件路径"""
    t0 = time.time()
    with open(pkl_path, 'rb') as f:
        feats, _flip, ids, paths = pickle.load(f)  # flip 特征为空，本任务不使用
    feats = np.ascontiguousarray(feats, dtype=np.float32)
    ids = np.asarray(ids)
    uniq, inv = np.unique(ids, return_inverse=True)
    ids_cont = inv.astype(np.int32)
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    feats /= norms  # 兜底归一化（数据已验证归一化，保证数值严格为 1）
    print(f"[加载] N={len(ids):,}  身份数={len(uniq):,}  耗时 {time.time()-t0:.1f}s")
    return feats, ids_cont, ids, paths


def pair_stats(ids_cont):
    """统计正负样本对数量（i<j，同 id 为正）"""
    _, counts = np.unique(ids_cont, return_counts=True)
    total_pos = int((counts.astype(np.int64) * (counts - 1) // 2).sum())
    total = len(ids_cont) * (len(ids_cont) - 1) // 2
    return total, total_pos, total - total_pos


# ---------------------------------------------------------------- GPU worker
def gpu_worker(cfg, feats_shm, ids_shm, tile_q):
    """单卡 worker: 从任务队列取 tile，计算相似度并实时统计直方图/提取样本对"""
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = f'cuda:{cfg["gpu"]}'
    torch.cuda.set_device(dev)
    B, BINS = cfg['block'], cfg['bins']
    N, mode = cfg['N'], cfg['mode']
    feats = feats_shm.to(dev)          # (N,512) fp32 常驻显存
    ids_gpu = ids_shm.to(dev)          # (N,) int32
    torch.cuda.synchronize()

    hist_all = torch.zeros(BINS, dtype=torch.int64, device=dev)
    hist_pos = torch.zeros(BINS, dtype=torch.int64, device=dev)
    pos_chunks = []                    # 精确正样本相似度
    extract = {s['name']: {'gi': [], 'gj': [], 'sim': [], 'count': 0}
               for s in cfg.get('extract_spec', [])}
    t_mm = t_hist = 0.0
    n_done = 0

    while True:
        tile = tile_q.get()
        if tile is None:
            break
        bi, bj = tile
        r0, r1 = bi * B, min((bi + 1) * B, N)
        c0, c1 = bj * B, min((bj + 1) * B, N)
        A, C = feats[r0:r1], feats[c0:c1]
        is_diag = bi == bj

        torch.cuda.synchronize(); t0 = time.time()
        simf = A @ C.t()                                   # (b1,b2) fp32
        simf.clamp_(LO, HI)
        if is_diag:  # 只保留严格上三角；下三角/对角填范围外值让 histc 丢弃
            tri = torch.ones(simf.shape[0], simf.shape[1],
                             dtype=torch.bool, device=dev).triu_(1)
            simf.masked_fill_(~tri, LO - 2.0)
        torch.cuda.synchronize(); t_mm += time.time() - t0

        t0 = time.time()
        h = torch.histc(simf, bins=BINS, min=LO, max=HI)   # 范围外值自动丢弃
        hist_all += h.round_().to(torch.int64)

        pos_mask = ids_gpu[r0:r1, None] == ids_gpu[None, c0:c1]
        if is_diag:
            pos_mask &= tri
        pv = simf[pos_mask]
        if pv.numel() > 0:
            hp = torch.histc(pv, bins=BINS, min=LO, max=HI)
            hist_pos += hp.round_().to(torch.int64)
            pos_chunks.append(pv.cpu().numpy())
        t_hist += time.time() - t0

        # ---------- 提取 above/below 阈值样本对 ----------
        if mode == 'extract':
            neg_mask = ~pos_mask
            for s in cfg['extract_spec']:
                t, kind, name = s['t'], s['kind'], s['name']
                m = (simf > t) & neg_mask if kind == 'above' else (simf <= t) & pos_mask
                cnt = int(m.sum())
                if cnt == 0:
                    continue
                extract[name]['count'] += cnt
                idx = m.nonzero()
                vals = simf[m]
                keep = cfg['extract_max']
                if cnt > keep:  # above 取相似度最高 / below 取最低的 top-K
                    v2, sel = torch.topk(vals if kind == 'above' else -vals, keep)
                    idx, vals = idx[sel], (v2 if kind == 'above' else -v2)
                extract[name]['gi'].append((idx[:, 0] + r0).cpu().numpy())
                extract[name]['gj'].append((idx[:, 1] + c0).cpu().numpy())
                extract[name]['sim'].append(vals.cpu().numpy())

        n_done += 1
        if n_done % 10 == 0 or n_done == cfg['nb_tiles']:
            print(f"  [gpu{cfg['gpu']}] tile {n_done}/{cfg['nb_tiles']} "
                  f"({bi},{bj}) matmul={t_mm/n_done*1000:.0f}ms hist={t_hist/n_done*1000:.0f}ms",
                  flush=True)

    out = os.path.join(cfg['outdir'], f"_gpu{cfg['gpu']}_{mode}.npz")
    payload = {
        'hist_all': hist_all.cpu().numpy(), 'hist_pos': hist_pos.cpu().numpy(),
        'pos_sims': (np.concatenate(pos_chunks) if pos_chunks else np.zeros(0, np.float32)),
        't_mm': t_mm, 't_hist': t_hist, 'n_tiles': n_done,
    }
    if mode == 'extract':
        for name, d in extract.items():
            payload[f'{name}::gi'] = np.concatenate(d['gi']) if d['gi'] else np.zeros(0, np.int64)
            payload[f'{name}::gj'] = np.concatenate(d['gj']) if d['gj'] else np.zeros(0, np.int64)
            payload[f'{name}::sim'] = np.concatenate(d['sim']) if d['sim'] else np.zeros(0, np.float32)
            payload[f'{name}::count'] = np.int64(d['count'])
    np.savez(out, **payload)
    print(f"  [gpu{cfg['gpu']}] 完成 {n_done} tiles | matmul {t_mm:.1f}s hist {t_hist:.1f}s", flush=True)


# ---------------------------------------------------------------- 并行调度
def run_pass(mode, feats_shm, ids_shm, cfg_base, extract_spec=None):
    """spawn 每 GPU 一个进程，tile 队列动态负载均衡，返回合并结果"""
    ctx = mp.get_context('spawn')
    gpus = cfg_base['gpus']
    nb = math.ceil(cfg_base['N'] / cfg_base['block'])
    tiles = [(bi, bj) for bi in range(nb) for bj in range(bi, nb)]
    tiles.sort(key=lambda t: -(t[1] - t[0]))  # 宽 tile（计算量大）先入队

    cfg = dict(cfg_base)
    cfg.update(mode=mode, nb=nb, nb_tiles=len(tiles), extract_spec=extract_spec or [])
    tile_q = ctx.Queue()
    for t in tiles:
        tile_q.put(t)
    for _ in gpus:
        tile_q.put(None)

    t0 = time.time()
    procs = []
    for gpu in gpus:
        c = dict(cfg); c['gpu'] = gpu
        p = ctx.Process(target=gpu_worker, args=(c, feats_shm, ids_shm, tile_q))
        p.start(); procs.append(p)
    for p in procs:
        p.join()
    dt = time.time() - t0

    merged = {'hist_all': None, 'hist_pos': None, 'pos_sims': [], 'extract': {}}
    for gpu in gpus:
        fp = os.path.join(cfg['outdir'], f"_gpu{gpu}_{mode}.npz")
        d = np.load(fp)
        merged['hist_all'] = d['hist_all'] if merged['hist_all'] is None \
            else merged['hist_all'] + d['hist_all']
        merged['hist_pos'] = d['hist_pos'] if merged['hist_pos'] is None \
            else merged['hist_pos'] + d['hist_pos']
        merged['pos_sims'].append(d['pos_sims'])
        if mode == 'extract':
            for key in d.files:
                if '::' in key:
                    name, field = key.split('::')
                    cur = merged['extract'].setdefault(name, {})
                    cur[field] = cur.get(field, 0) + d[key] if field == 'count' \
                        else cur.get(field, []) + [d[key]]
        d.close()
        os.remove(fp)
    merged['pos_sims'] = np.concatenate(merged['pos_sims'])
    merged['dt'] = dt
    return merged


# ---------------------------------------------------------------- 指标计算
def threshold_at_fpir(neg_counts, total_neg, fpir):
    """由负样本直方图尾部累计求 bin 序号: 使 count(sim>t)/total_neg <= fpir"""
    S = np.cumsum(neg_counts[::-1])[::-1]  # S[i] = bin i..末尾计数 (sim >= LO+i*w)
    k = max(1, math.ceil(fpir * total_neg))
    hits = np.nonzero(S <= k)[0]
    return int(hits[0]) if len(hits) else len(S) - 1


def threshold_at_tpir(pos_sorted, total_pos, tpir):
    """由精确正样本相似度分位数求阈值: 使 TPIR(t) ≈ tpir"""
    k = int(round((1.0 - tpir) * total_pos))
    k = min(max(k, 1), total_pos - 1)
    return float(pos_sorted[k])


def compute_metrics(neg_counts, pos_sims, bins):
    pos_sorted = np.sort(pos_sims)
    total_neg = int(neg_counts.sum())
    total_pos = len(pos_sims)
    w = (HI - LO) / bins

    def tp_at(t):
        return 1.0 - np.searchsorted(pos_sorted, t, side='right') / total_pos

    points = []
    for f in [1e-5, 1e-4, 1e-3, 1e-2]:
        t = LO + threshold_at_fpir(neg_counts, total_neg, f) * w
        points.append({'fpir': f, 'threshold': round(float(t), 6),
                       'tpir': float(tp_at(t))})

    grid = np.logspace(-6, 0, 100)
    curve_f, curve_t, curve_th = [], [], []
    for f in grid:
        t = LO + threshold_at_fpir(neg_counts, total_neg, f) * w
        curve_f.append(f); curve_t.append(tp_at(t)); curve_th.append(t)
    return points, (np.array(curve_f), np.array(curve_t), np.array(curve_th))


# ---------------------------------------------------------------- 可视化
def make_plots(neg_counts, pos_sims, points, curve, bins, outdir, totals, elapsed):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager
    import matplotlib.pyplot as plt

    if os.path.exists(FONT_PATH):
        font_manager.fontManager.addfont(FONT_PATH)
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=FONT_PATH).get_name()
    else:
        print(f"[警告] 未找到中文字体 {FONT_PATH}")
    plt.rcParams.update({'axes.unicode_minus': False,
        'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb',
        'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': '#52514e',
        'xtick.color': '#898781', 'ytick.color': '#898781',
        'text.color': '#0b0b0b', 'axes.grid': True, 'grid.color': '#e1e0d9',
        'grid.linewidth': 0.8, 'axes.axisbelow': True, 'font.size': 10})
    C_POS, C_NEG = '#2a78d6', '#eb6834'

    total_pairs, total_pos, total_neg, N = totals
    w = (HI - LO) / bins
    disp = 2000                          # 展示用降采样 bin 数
    grp = bins // disp
    neg_disp = neg_counts.reshape(disp, grp).sum(1).astype(np.float64)
    edges = np.linspace(LO, HI, disp + 1)
    pos_disp = np.histogram(pos_sims, bins=edges)[0].astype(np.float64)

    def draw(ax, counts, color, label):
        # 只绘制非零支撑区间，避免 log 轴下 0 计数产生误导性基线
        nz = np.nonzero(counts > 0)[0]
        i0, i1 = nz[0], nz[-1] + 1
        ax.stairs(counts[i0:i1], edges[i0:i1 + 1], color=color, lw=2,
                  alpha=0.8, label=label)

    # ---- 图1: 相似度分布 ----
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for ax, title in [(axes[0], '全范围'), (axes[1], '判决区域放大')]:
        draw(ax, neg_disp, C_NEG, f'负样本对 ({total_neg:.2e})')
        draw(ax, pos_disp, C_POS, f'正样本对 ({total_pos:.2e})')
        ax.set_yscale('log'); ax.set_xlabel('余弦相似度')
        ax.set_ylabel('样本对数（对数刻度）'); ax.set_title(title, fontsize=11)
    j_lo = max(0, int((points[3]['threshold'] - 0.15 - LO) / w / grp))
    j_hi = min(disp, int((min(1.0, points[0]['threshold'] + 0.1) - LO) / w / grp))
    axes[1].set_xlim(edges[j_lo], edges[j_hi])
    for p in points:
        axes[1].axvline(p['threshold'], color='#898781', lw=1, ls=':')
    axes[1].annotate(f"FPIR=1e-3 阈值 {points[2]['threshold']:.4f}",
                     xy=(points[2]['threshold'], 0.5), fontsize=8, color='#52514e',
                     xytext=(5, 0), textcoords='offset points')
    axes[0].legend(frameon=False, loc='upper left')
    fig.suptitle('正/负样本对相似度分布', fontsize=13)
    fig.tight_layout()
    p1 = os.path.join(outdir, 'sim_distribution.png')
    fig.savefig(p1, dpi=150, bbox_inches='tight'); plt.close(fig)

    # ---- 图2: TPIR@FPIR 曲线 ----
    curve_f, curve_t, curve_th = curve
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax = axes[0]
    ax.semilogx(curve_f, curve_t * 100, color=C_POS, lw=2, label='TPIR@FPIR')
    for p in points:
        ax.plot(p['fpir'], p['tpir'] * 100, 'o', ms=8, color=C_NEG, zorder=5)
        ax.annotate(f"{p['tpir']*100:.2f}%", xy=(p['fpir'], p['tpir'] * 100),
                    xytext=(7, -3), textcoords='offset points', fontsize=9,
                    color='#52514e')
    ax.set_xlabel('FPIR（对数刻度）'); ax.set_ylabel('TPIR (%)')
    ax.set_title('TPIR @ FPIR', fontsize=11); ax.set_ylim(0, 102)
    ax.legend(frameon=False, loc='lower right')
    ax = axes[1]
    ax.semilogx(curve_f, curve_th, color=C_POS, lw=2)
    for p in points:
        ax.plot(p['fpir'], p['threshold'], 'o', ms=8, color=C_NEG, zorder=5)
        ax.annotate(f"{p['threshold']:.4f}", xy=(p['fpir'], p['threshold']),
                    xytext=(7, 2), textcoords='offset points', fontsize=9,
                    color='#52514e')
    ax.set_xlabel('FPIR（对数刻度）'); ax.set_ylabel('判决阈值')
    ax.set_title('阈值 — FPIR 工作点', fontsize=11)
    ax.set_ylim(0, max(curve_th) * 1.1)  # 裁掉 FPIR→1 时阈值趋向 -1 的无意义段
    fig.suptitle(f'TPIR@FPIR 评估（N={N:,}，总对数 {total_pairs:.2e}，'
                 f'总耗时 {elapsed:.1f}s）', fontsize=13)
    fig.tight_layout()
    p2 = os.path.join(outdir, 'tpir_at_fpir.png')
    fig.savefig(p2, dpi=150, bbox_inches='tight'); plt.close(fig)
    return p1, p2


# ---------------------------------------------------------------- 样本对提取
def write_extract_csvs(extract, totals, orig_ids, paths, outdir, extract_max):
    """合并各 GPU 提取结果，映射回文件路径并写 CSV"""
    edir = os.path.join(outdir, 'extract')
    os.makedirs(edir, exist_ok=True)
    _, total_pos, total_neg, _ = totals
    summary = {}
    for name, d in extract.items():
        gi = np.concatenate(d['gi']); gj = np.concatenate(d['gj'])
        sim = np.concatenate(d['sim']); count = int(d['count'])
        if len(sim) > extract_max:
            sel = np.argsort(-sim)[:extract_max]
            gi, gj, sim = gi[sel], gj[sel], sim[sel]
        order = np.argsort(-sim)
        gi, gj, sim = gi[order], gj[order], sim[order]
        fp = os.path.join(edir, f'{name}.csv')
        with open(fp, 'w', newline='', encoding='utf-8') as f:
            wtr = csv.writer(f)
            wtr.writerow(['i', 'j', 'sim', 'id_i', 'id_j', 'path_i', 'path_j'])
            for a, b, s in zip(gi.tolist(), gj.tolist(), sim.tolist()):
                wtr.writerow([a, b, f'{s:.6f}',
                              orig_ids[a], orig_ids[b], paths[a], paths[b]])
        kind = 'above' if name.startswith('above') else 'below'
        rate = count / (total_neg if kind == 'above' else total_pos)
        summary[name] = {'count': count, 'saved': len(sim), 'csv': fp,
                         'rate': float(rate)}
        print(f"  [{name}] 命中 {count:,} 对 (占比 {rate:.3e})，"
              f"保存 top-{len(sim):,} → {fp}")
    return summary


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(description='人脸特征相似度评估（多GPU并行）')
    ap.add_argument('--pkl', default=PKL_PATH)
    ap.add_argument('--outdir', default='results')
    ap.add_argument('--gpus', default='0,1,2,3,4,5,6', help='用于计算的 GPU 列表')
    ap.add_argument('--block', type=int, default=16384, help='分块大小')
    ap.add_argument('--bins', type=int, default=2_000_000, help='相似度直方图 bin 数')
    ap.add_argument('--extract-fpir', default='', help='逗号分隔 FPIR 值，提取其阈值之上的负样本对')
    ap.add_argument('--extract-tpir', default='', help='逗号分隔 TPIR 值，提取其阈值之下的正样本对')
    ap.add_argument('--extract-thresh', default='', help='逗号分隔原始阈值，提取 above/below 样本对')
    ap.add_argument('--extract-max', type=int, default=100_000, help='每个阈值最多保存的样本对数')
    args = ap.parse_args()

    gpus = [int(g) for g in args.gpus.split(',') if g != '']
    os.makedirs(args.outdir, exist_ok=True)
    t_start = time.time()

    # 1. 数据加载（仅主进程，CPU）
    feats, ids_cont, orig_ids, paths = load_data(args.pkl)
    N = len(ids_cont)
    total, total_pos, total_neg = pair_stats(ids_cont)
    t_load = time.time() - t_start
    print(f"[样本对] 总 {total:,} | 正 {total_pos:,} ({total_pos/total*100:.4f}%) "
          f"| 负 {total_neg:,}")

    # 共享内存张量，避免向每个 worker 复制特征
    feats_shm = torch.from_numpy(feats).share_memory_()
    ids_shm = torch.from_numpy(ids_cont).share_memory_()
    cfg_base = {'N': N, 'block': args.block, 'bins': args.bins,
                'gpus': gpus, 'outdir': args.outdir,
                'extract_max': args.extract_max}

    # 2. 第一遍扫描: 直方图统计 + 精确正样本相似度
    print(f"\n[扫描1] 统计模式: {len(gpus)} GPU × block={args.block} × "
          f"{args.bins:,} bins ...")
    res = run_pass('stats', feats_shm, ids_shm, cfg_base)
    t_scan = res['dt']
    hist_all, hist_pos, pos_sims = res['hist_all'], res['hist_pos'], res['pos_sims']
    neg_counts = hist_all - hist_pos
    assert neg_counts.min() >= 0, '直方图计数出现负值！'

    # 3. 校验 + 指标
    got_all, got_pos = int(hist_all.sum()), int(hist_pos.sum())
    print(f"\n[校验] 直方图总计数 {got_all:,} (理论 {total:,}) "
          f"{'✓' if got_all == total else '✗ 异常!'}")
    print(f"[校验] 正样本计数 {got_pos:,} (理论 {total_pos:,}) "
          f"{'✓' if got_pos == total_pos else '✗ 异常!'} "
          f"| 精确收集正样本相似度 {len(pos_sims):,} 条")

    points, curve = compute_metrics(neg_counts, pos_sims, args.bins)
    print(f"[扫描1] GPU 耗时 {t_scan:.1f}s\n")
    print("=" * 56)
    print(f"{'FPIR':>8} | {'判决阈值':>12} | {'TPIR':>10}")
    print('-' * 56)
    for p in points:
        print(f"{p['fpir']:>8.0e} | {p['threshold']:>12.4f} | {p['tpir']*100:>9.2f}%")
    print("=" * 56)

    # 4. 可视化 + 持久化统计结果（后续分析无需重新计算）
    totals = (total, total_pos, total_neg, N)
    p1, p2 = make_plots(neg_counts, pos_sims, points, curve, args.bins,
                        args.outdir, totals, time.time() - t_start)
    np.save(os.path.join(args.outdir, 'pos_sims.npy'), pos_sims)
    np.save(os.path.join(args.outdir, 'neg_hist.npy'), neg_counts.astype(np.int64))
    print(f"\n[输出] {p1}\n[输出] {p2}")
    print(f"[输出] {args.outdir}/pos_sims.npy (精确正样本相似度), "
          f"{args.outdir}/neg_hist.npy (负样本直方图)")

    # 5. 第二遍扫描: 提取 above/below 阈值样本对
    spec = []
    for v in args.extract_fpir.split(','):
        if v.strip():
            f = float(v)
            t = LO + threshold_at_fpir(neg_counts, total_neg, f) / args.bins * (HI - LO)
            spec.append({'name': f'above_fpir{f:g}', 't': t, 'kind': 'above'})
    for v in args.extract_tpir.split(','):
        if v.strip():
            v_ = float(v)
            t = threshold_at_tpir(np.sort(pos_sims), total_pos, v_)
            spec.append({'name': f'below_tpir{v_:g}', 't': t, 'kind': 'below'})
    for v in args.extract_thresh.split(','):
        if v.strip():
            t = float(v)
            spec.append({'name': f'above_th{t:g}', 't': t, 'kind': 'above'})
            spec.append({'name': f'below_th{t:g}', 't': t, 'kind': 'below'})

    extract_summary = {}
    if spec:
        print(f"\n[扫描2] 提取模式: {len(spec)} 个阈值 × {len(gpus)} GPU ...")
        for s in spec:
            print(f"  - {s['name']}: t={s['t']:.4f} ({s['kind']})")
        res2 = run_pass('extract', feats_shm, ids_shm, cfg_base, extract_spec=spec)
        print(f"[扫描2] GPU 耗时 {res2['dt']:.1f}s")
        extract_summary = write_extract_csvs(
            res2['extract'], totals, orig_ids, paths, args.outdir, args.extract_max)

    # 6. 汇总
    metrics = {
        'N': N, 'total_pairs': total, 'pos_pairs': total_pos, 'neg_pairs': total_neg,
        'tpir_at_fpir': points,
        'timing': {'load_s': round(t_load, 1), 'scan1_gpu_s': round(t_scan, 1),
                   'total_s': round(time.time() - t_start, 1)},
        'config': {'gpus': gpus, 'block': args.block, 'bins': args.bins},
        'extract': extract_summary,
    }
    mpath = os.path.join(args.outdir, 'metrics.json')
    with open(mpath, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[输出] {mpath}")
    print(f"\n[完成] 总耗时 {time.time()-t_start:.1f}s")


if __name__ == '__main__':
    main()
