#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第三步(最终版): 多GPU并行人脸特征相似度评估系统

架构设计 (针对206亿样本对, 不保存全量相似度):
- 单进程 + 多GPU + 每GPU独立CUDA stream 异步执行
  (相比多进程: 无spawn启动开销、无进程间通信、结果合并零成本)
- 等面积行划分: 第k卡负责行[b_k, b_{k+1}), 各卡上三角面积相等
  b_k = N*(1-sqrt(1-k/G)); 第k卡只需加载特征切片F[b_k:]
- 特征传输多线程并行(各卡独立PCIe DMA); GPU上下文预热与数据加载并行
- 正样本对(218万): 按身份组分配给组起点所在卡, 同组大小分桶 gather+bmm, 全量精确保存
- 负样本对(206亿): 大块上三角matmul + 两级直方图统计(粗4096全域 + 细65536尾部)
                   + top-N最高相似度对增量提取(带索引, 用于错误分析)

关键性能要点 (实测迭代得出):
1. 全程只用异步CUDA op: boolean索引/bincount等需要读取GPU结果的op会强制CPU-GPU同步,
   曾导致每块24ms的CPU等待; 改用 masked_fill_打标记 + histc(输出大小固定,异步)
2. 排除位置(正样本对/下三角)填-2: histc(min>域内)自动忽略, topk中自然沉底,
   每卡结尾topk保序后按有效前缀切片, 全程无同步
3. histc直接在sim域[-1,1]统计, 无需量化; topk结果即精确sim(无需重算)
4. 大块(8192x65536)减少launch次数; 三角mask模板预分配

指标: TPIR@FPIR (精确top-N覆盖FPIR>=cap/n_neg时用精确值, 其余用直方图)
提取: 负样本对 sim>T (above/假阳性) / 正样本对 sim<T (below/假阴性), 导出原始索引
可视化: 相似度分布图 + TPIR@FPIR曲线 (中文字体)
"""
import argparse
import math
import os
import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

DEFAULT_FONT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'font', 'SourceHanSansSC-Normal.otf')

# 两级直方图 (sim域): 粗=全域, 细=尾部[0.2, 1]
COARSE_BINS = 4096            # bin宽 4.88e-4 (分布图/大FPIR)
FINE_BINS, FINE_LO, FINE_HI = 65536, 0.2, 1.0   # bin宽 1.22e-5 (FPIR<=~2e-2)
INVALID = -2.0                # 排除位置标记 (sim域外, histc忽略, topk沉底)


# ----------------------------- 参数 -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description='多GPU人脸特征相似度评估系统 (TPIR@FPIR)')
    p.add_argument('--pkl', default='s4_0618_enhance.pkl', help='特征文件路径')
    p.add_argument('--gpus', default='0,1,2,3,4,5,6', help='使用的GPU编号, 逗号分隔')
    p.add_argument('--block-rows', type=int, default=8192, help='行块大小')
    p.add_argument('--block-cols', type=int, default=65536, help='列块大小')
    p.add_argument('--extract-cap', type=int, default=1000000,
                   help='top-N最高相似度负样本对提取上限(精确覆盖FPIR>=cap/n_neg)')
    p.add_argument('--extract-above', default='auto',
                   help="负样本对提取阈值 sim>T ('auto'=FPIR 1e-5阈值, 'none'=关闭, 或数值)")
    p.add_argument('--extract-below', default='auto',
                   help="正样本对提取阈值 sim<T ('auto'=同above阈值, 'none'=关闭, 或数值)")
    p.add_argument('--fpir-targets', default='1e-2,1e-3,1e-4,1e-5', help='报告的FPIR评估点')
    p.add_argument('--out-dir', default='results', help='输出目录')
    p.add_argument('--save-npz', default='eval_stats.npz', help='统计结果保存文件名(空串关闭)')
    p.add_argument('--no-plot', action='store_true', help='不绘图')
    p.add_argument('--no-extract', action='store_true', help='不提取样本对(跳过topk, 最快)')
    p.add_argument('--with-paths', action='store_true', help='导出提取对时附带文件路径')
    p.add_argument('--tf32', action='store_true', help='允许TF32加速matmul(略降精度, 默认严格fp32)')
    return p.parse_args()


# ----------------------------- 数据 -----------------------------
def load_and_sort(path):
    """读取pkl, 按id稳定排序(同身份连续), 返回紧凑id与原始索引映射"""
    with open(path, 'rb') as f:
        feats, _, ids, file_paths = pickle.load(f)
    feats = np.asarray(feats, dtype=np.float32)
    ids = np.asarray(ids)
    order = np.argsort(ids, kind='stable')
    feats_s = feats[order]
    ids_s = ids[order]
    _, ids_c = np.unique(ids_s, return_inverse=True)
    counts = np.bincount(ids_c)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    ends = np.cumsum(counts).astype(np.int64)
    return feats_s, ids_c.astype(np.int32), counts, starts, ends, order, file_paths


def build_equal_area_bounds(N, G):
    """等面积行划分: 各卡上三角面积相等, 返回G+1个边界"""
    b = [0] * (G + 1)
    b[G] = N
    for k in range(1, G):
        b[k] = int(round(N * (1.0 - math.sqrt(1.0 - k / G))))
    for k in range(1, G + 1):
        b[k] = max(b[k], b[k - 1] + 1)
    b[G] = N
    return b


def transfer_to_gpu(dev, stream, F_cpu, ids_cpu, a):
    """在独立线程中把本卡需要的特征切片传到GPU (各卡PCIe DMA并行)"""
    with torch.cuda.device(dev), torch.cuda.stream(stream):
        return F_cpu[a:].to(dev), ids_cpu.to(dev)


# ----------------------------- GPU任务(异步排队) -----------------------------
def launch_gpu_tasks(dev, stream, Fk, ids_g, a, b, N, B, C, cap,
                     g_starts, g_ends, do_extract=True):
    """在指定GPU的stream上异步排队: 负样本大块扫描 + 正样本bmm(后置, 避免其分配split扫描大块的显存池)
    返回结果tensor引用(全部留在GPU, 调用方synchronize后回收)"""
    with torch.cuda.device(dev), torch.cuda.stream(stream):
        e1, e2, e3 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        e1.record()
        neg_c = torch.zeros(COARSE_BINS, dtype=torch.int64, device=dev)
        neg_f = torch.zeros(FINE_BINS, dtype=torch.int64, device=dev)

        # ---- 负样本: 行块×列块上三角大块扫描 (全程异步op) ----
        empty = lambda dt: torch.zeros(0, dtype=dt, device=dev)
        # 三角排除模板: 前B×B为下三角(含对角线), 其余列False
        tri_tpl = torch.ones(B, C, dtype=torch.bool, device=dev).tril_()
        buf_v = buf_i = buf_j = None
        for r0 in range(a, b, B):
            r1 = min(r0 + B, b)
            rows = Fk[r0 - a:r1 - a]
            ids_rows = ids_g[r0:r1]
            for c0 in range(r0, N, C):
                c1 = min(c0 + C, N)
                S = rows @ Fk[c0 - a:c1 - a].T               # (b,c) fp32 ∈[-1,1]
                eq = ids_rows[:, None] == ids_g[c0:c1][None, :]
                if c0 < r1:  # 列块与行块重叠: 排除下三角(含对角线)
                    excl = eq | tri_tpl[:r1 - r0, :c1 - c0]
                else:
                    excl = eq
                # clamp防fp32误差越出histc范围; 排除位置填INVALID(histc自动忽略)
                S.clamp_(-1.0, 1.0).masked_fill_(excl, INVALID)
                flatS = S.view(-1)
                # 两级直方图: histc输出大小固定→异步
                neg_c += torch.histc(flatS, bins=COARSE_BINS, min=-1.0, max=1.0).long()
                neg_f += torch.histc(flatS, bins=FINE_BINS, min=FINE_LO, max=FINE_HI).long()
                # top-N: 排除项(-2)沉底; 结果即精确sim, 合并阶段按有效前缀切片
                if do_extract:
                    k = min(cap, flatS.numel())
                    v, p = torch.topk(flatS, k)
                    wc = c1 - c0
                    if buf_v is None:
                        buf_v, buf_i, buf_j = v, p // wc + r0, p % wc + c0
                    else:
                        buf_v = torch.cat([buf_v, v])
                        buf_i = torch.cat([buf_i, p // wc + r0])
                        buf_j = torch.cat([buf_j, p % wc + c0])
                    if buf_v.numel() > 2 * cap:
                        sel = torch.topk(buf_v, cap).indices
                        buf_v, buf_i, buf_j = buf_v[sel], buf_i[sel], buf_j[sel]
        if buf_v is None:
            buf_v, buf_i, buf_j = empty(torch.float32), empty(torch.long), empty(torch.long)
        else:  # 统一topk保序(值降序→无效项连续沉底), 便于合并阶段前缀切片
            kk = min(cap, buf_v.numel())
            bv, order_ = torch.topk(buf_v, kk)
            buf_v, buf_i, buf_j = bv, buf_i[order_], buf_j[order_]
        e2.record()

        # ---- 正样本(后置): 组起点在[a,b)的身份组, 同组大小分桶 gather+bmm ----
        ps, pi, pj = [], [], []
        sizes = g_ends - g_starts
        buckets = []
        for c in np.unique(sizes):
            if c < 2:
                continue
            gsel = np.where(sizes == c)[0]
            idx_np = np.concatenate([np.arange(g_starts[k], g_ends[k]) for k in gsel]) - a
            buckets.append((int(c), len(gsel), g_starts[gsel], idx_np))
        if buckets:
            all_idx = torch.from_numpy(np.concatenate([x[3] for x in buckets])).to(dev)
            all_base = torch.from_numpy(np.concatenate([x[2] for x in buckets])).to(dev)
            offs, boffs = [0], [0]
            for c, nG, _, idx_np in buckets:
                offs.append(offs[-1] + len(idx_np))
                boffs.append(boffs[-1] + nG)
            for bi, (c, nG, _, _) in enumerate(buckets):
                X = Fk[all_idx[offs[bi]:offs[bi + 1]]].view(nG, c, 512)
                Gm = torch.bmm(X, X.transpose(1, 2))
                tri = torch.triu_indices(c, c, offset=1, device=dev)
                base = all_base[boffs[bi]:boffs[bi + 1]]
                ps.append(Gm[:, tri[0], tri[1]].reshape(-1))
                pi.append((base[:, None] + tri[0][None, :]).reshape(-1))
                pj.append((base[:, None] + tri[1][None, :]).reshape(-1))
        pos = (torch.cat(ps), torch.cat(pi), torch.cat(pj)) if ps else \
              (empty(torch.float32), empty(torch.long), empty(torch.long))
        e3.record()
        return {'dev': dev, 'Fk': Fk, 'neg_c': neg_c, 'neg_f': neg_f, 'pos': pos,
                'buf': (buf_v, buf_i, buf_j), 'ev': (e1, e2, e3)}


# ----------------------------- 直方图指标 -----------------------------
class Hist:
    """一级直方图: sim域[lo,hi]的nbins个bin"""
    def __init__(self, hist, nbins, lo, hi):
        self.hist = hist
        self.nbins, self.lo = nbins, lo
        self.w = (hi - lo) / nbins
        self.cum_above = np.cumsum(hist[::-1])[::-1]       # #{sim >= bin左边界}

    def bin_left_sim(self, b):
        return self.lo + b * self.w

    def threshold_for_count(self, target):
        """最大阈值(bin左边界)使 count(sim>=t) >= target"""
        below = self.cum_above < target
        if below.any():
            b = max(int(np.argmax(below)) - 1, 0)
        else:
            b = self.nbins - 1
        return self.bin_left_sim(b)


def find_threshold(fine, coarse, n_neg, f):
    """FPIR=f 对应阈值: 优先细直方图(阈值>=0.2), 否则粗直方图"""
    target = f * n_neg
    if fine is not None and fine.cum_above[0] >= target:
        return fine.threshold_for_count(target), 'fine'
    return coarse.threshold_for_count(target), 'coarse'


def eval_metrics(pos_sims, neg_c, neg_f, top_v_sorted_desc, n_neg, targets):
    pos_sorted = np.sort(pos_sims)
    n_pos = len(pos_sorted)
    coarse = Hist(neg_c, COARSE_BINS, -1.0, 1.0)
    fine = Hist(neg_f, FINE_BINS, FINE_LO, FINE_HI) if neg_f is not None else None
    rows = []
    for f in targets:
        r = int(round(f * n_neg))
        if top_v_sorted_desc is not None and 1 <= r <= len(top_v_sorted_desc):
            t = float(top_v_sorted_desc[r - 1])
            method = 'exact'
        else:
            t, lvl = find_threshold(fine, coarse, n_neg, f)
            method = f'hist-{lvl}'
        tpir = 1.0 - np.searchsorted(pos_sorted, t, side='right') / n_pos
        rows.append({'fpir': f, 'tpir': tpir, 'threshold': t, 'method': method})
    return rows, pos_sorted, coarse, fine


def tpir_fpir_curve(pos_sorted, coarse, fine, top_v_sorted_desc, n_neg,
                    f_min=1e-6, f_max=1.0, npts=300):
    fs = np.logspace(math.log10(f_min), math.log10(f_max), npts)
    n_pos = len(pos_sorted)
    ts = np.empty(len(fs))
    for i, f in enumerate(fs):
        r = int(round(f * n_neg))
        if top_v_sorted_desc is not None and 1 <= r <= len(top_v_sorted_desc):
            ts[i] = top_v_sorted_desc[r - 1]
        else:
            ts[i], _ = find_threshold(fine, coarse, n_neg, f)
    tpirs = 1.0 - np.searchsorted(pos_sorted, ts, side='right') / n_pos
    return fs, tpirs, ts


# ----------------------------- 提取 -----------------------------
def export_pairs(out_path, i_orig, j_orig, sims, file_paths, with_paths, tag):
    data = {'i': i_orig.astype(np.int64), 'j': j_orig.astype(np.int64),
            'sim': sims.astype(np.float32), 'tag': tag}
    if with_paths:
        data['path_i'] = [file_paths[k] for k in i_orig]
        data['path_j'] = [file_paths[k] for k in j_orig]
    with open(out_path, 'wb') as f:
        pickle.dump(data, f)
    return data


# ----------------------------- 可视化 -----------------------------
def plot_all(out_dir, coarse, pos_sims, curve, key_rows, font_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = prop.get_name()
    plt.rcParams['axes.unicode_minus'] = False

    n_neg, n_pos = int(coarse.hist.sum()), len(pos_sims)

    # ---------- 图1: 相似度分布 ----------
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    lo, hi, M = -0.4, 1.0, 280
    edges = np.linspace(lo, hi, M + 1)
    centers = coarse.bin_left_sim(np.arange(coarse.nbins)) + coarse.w / 2
    neg_d = np.histogram(centers, bins=edges, weights=coarse.hist)[0] / (n_neg * (edges[1] - edges[0]))
    pos_d = np.histogram(pos_sims, bins=edges)[0] / (n_pos * (edges[1] - edges[0]))
    mids = (edges[:-1] + edges[1:]) / 2
    ax.fill_between(mids, neg_d, step='mid', alpha=0.55, color='#4C72B0', label=f'负样本对 ({n_neg:,})')
    ax.fill_between(mids, pos_d, step='mid', alpha=0.55, color='#DD8452', label=f'正样本对 ({n_pos:,})')
    t5 = [r['threshold'] for r in key_rows if abs(r['fpir'] - 1e-5) < 1e-12]
    if t5:
        ax.axvline(t5[0], color='#C44E52', ls='--', lw=1.2)
        ax.text(t5[0], ax.get_ylim()[1] * 0.5, f' FPIR=1e-5\n 阈值={t5[0]:.4f}',
                fontsize=9, color='#C44E52', va='top')
    ax.set_yscale('log')
    ax.set_xlim(lo, hi)
    ax.set_xlabel('余弦相似度', fontsize=12)
    ax.set_ylabel('概率密度 (对数轴)', fontsize=12)
    ax.set_title('正/负样本对相似度分布', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, which='both')
    p1 = os.path.join(out_dir, 'sim_distribution.png')
    fig.tight_layout()
    fig.savefig(p1)
    plt.close(fig)

    # ---------- 图2: TPIR@FPIR曲线 ----------
    fs, tpirs, _ = curve
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.semilogx(fs, tpirs * 100, color='#4C72B0', lw=2, label='TPIR@FPIR')
    ax.scatter([r['fpir'] for r in key_rows], [r['tpir'] * 100 for r in key_rows],
               color='#C44E52', zorder=5, s=40)
    for r in key_rows:
        ax.annotate(f"FPIR={r['fpir']:.0e}\nTPIR={r['tpir']*100:.2f}% (阈值{r['threshold']:.3f})",
                    (r['fpir'], r['tpir'] * 100), textcoords='offset points',
                    xytext=(8, -14), fontsize=9, color='#C44E52')
    ax.set_xlabel('FPIR (对数轴)', fontsize=12)
    ax.set_ylabel('TPIR (%)', fontsize=12)
    ax.set_title('TPIR@FPIR 曲线', fontsize=14)
    ax.set_xlim(fs[0], 1.0)
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=11)
    p2 = os.path.join(out_dir, 'tpir_fpir_curve.png')
    fig.tight_layout()
    fig.savefig(p2)
    plt.close(fig)
    return p1, p2


# ----------------------------- 主流程 -----------------------------
def main():
    args = parse_args()
    t_total = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    gpus = [int(x) for x in args.gpus.split(',')]
    G, cap = len(gpus), args.extract_cap
    do_extract = not args.no_extract

    print("=" * 88)
    print(f"多GPU人脸特征相似度评估系统 | GPU: {gpus} | 提取: {do_extract} | TF32: {args.tf32}")
    print("=" * 88)

    # 预热线程 (与数据加载并行):
    # - 每GPU一个线程: 创建CUDA上下文 + 预分配显存池(消除首块cudaMalloc阻塞, 其会同步等待设备空闲)
    # - matplotlib预导入 (绘图耗时项, 与加载并行)
    def _warm_gpu(g):
        dev = torch.device(g)
        with torch.cuda.device(dev):
            torch.zeros(8192, 8192, device=dev).tril_()          # 上下文+kernel
            t = [torch.empty(args.block_rows, args.block_cols, device=dev),   # S
                 torch.empty(args.block_rows, args.block_cols, dtype=torch.bool, device=dev),  # eq
                 torch.empty(args.block_rows, args.block_cols, dtype=torch.bool, device=dev)]  # excl
            del t                                                # 归还caching allocator池, 后续复用免cudaMalloc

    def _warm_mpl():
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot  # noqa: F401

    warm_threads = [threading.Thread(target=_warm_gpu, args=(g,), daemon=True) for g in gpus]
    warm_threads.append(threading.Thread(target=_warm_mpl, daemon=True))
    for t in warm_threads:
        t.start()

    # ---------- 1. 加载与排序 ----------
    t0 = time.time()
    feats_s, ids_c, counts, starts, ends, order, file_paths = load_and_sort(args.pkl)
    N = len(ids_c)
    n_pos = int(sum(int(c) * (c - 1) // 2 for c in counts))
    n_neg = N * (N - 1) // 2 - n_pos
    F_cpu = torch.from_numpy(np.ascontiguousarray(feats_s))
    ids_cpu = torch.from_numpy(ids_c)
    print(f"\n[1] 加载+排序: {time.time()-t0:.2f}s | N={N:,} 身份={len(counts):,} "
          f"| 正对={n_pos:,} 负对={n_neg:,}")

    # ---------- 2. 等面积划分 ----------
    bounds = build_equal_area_bounds(N, G)
    print(f"[2] 等面积行划分 (每卡上三角~{n_neg/G/1e9:.2f}G对):")
    for k in range(G):
        print(f"    GPU{gpus[k]}: 行[{bounds[k]:>7,}, {bounds[k+1]:>7,}) "
              f"({bounds[k+1]-bounds[k]:,}行, 特征{(N-bounds[k])/N*100:.0f}%)")
    g_assign = np.clip(np.searchsorted(bounds, starts, side='right') - 1, 0, G - 1)
    for t in warm_threads:
        t.join()  # 等预热完成

    # ---------- 3. 多GPU异步执行 ----------
    t0 = time.time()
    streams = [torch.cuda.Stream(device=torch.device(g)) for g in gpus]
    devs = [torch.device(g) for g in gpus]
    # 特征切片多线程并行传输 (各卡独立PCIe DMA)
    with ThreadPoolExecutor(G) as ex:
        transfers = list(ex.map(
            lambda k: transfer_to_gpu(devs[k], streams[k], F_cpu, ids_cpu, bounds[k]),
            range(G)))
    t_transfer = time.time() - t0
    results = []
    for gi in range(G):
        gsel = np.where(g_assign == gi)[0]
        results.append(launch_gpu_tasks(
            devs[gi], streams[gi], transfers[gi][0], transfers[gi][1],
            bounds[gi], bounds[gi + 1], N, args.block_rows, args.block_cols, cap,
            starts[gsel], ends[gsel], do_extract))
    for s in streams:
        s.synchronize()
    t_compute = time.time() - t0
    ev_lines = [f"    GPU{gpus[k]}: 正样本 {r['ev'][0].elapsed_time(r['ev'][1]):4.0f}ms | "
                f"扫描 {r['ev'][1].elapsed_time(r['ev'][2]):5.0f}ms"
                for k, r in enumerate(results)]
    print(f"\n[3] 多GPU执行完成: {t_compute:.2f}s (含并行传输{t_transfer:.2f}s) | "
          f"吞吐 {n_neg/t_compute/1e9:.1f} G对/s")
    print("\n".join(ev_lines))

    # ---------- 4. 合并 (各卡结果并行D2H) ----------
    t0 = time.time()

    def fetch(r):
        p = r['pos']
        bv, bi, bj = r['buf']
        if bv.numel():  # buf值降序, 无效项(-2)连续沉底 → 有效前缀切片(GPU空闲, item()无流水线代价)
            n_valid = int((bv > INVALID / 2).sum().item())
            bv, bi, bj = bv[:n_valid], bi[:n_valid], bj[:n_valid]
        return (r['neg_c'].cpu().numpy(), r['neg_f'].cpu().numpy(),
                p[0].cpu().numpy(), p[1].cpu().numpy(), p[2].cpu().numpy(),
                bv.cpu().numpy(), bi.cpu().numpy(), bj.cpu().numpy())

    with ThreadPoolExecutor(G) as ex:
        fetched = list(ex.map(fetch, results))
    neg_c = sum(f[0] for f in fetched)
    neg_f = sum(f[1] for f in fetched)
    pos_sims = np.concatenate([f[2] for f in fetched])
    pos_i = np.concatenate([f[3] for f in fetched])
    pos_j = np.concatenate([f[4] for f in fetched])

    # 校验
    assert int(neg_c.sum()) == n_neg, f"负样本对数不符: {int(neg_c.sum())} != {n_neg}"
    assert len(pos_sims) == n_pos, f"正样本对数不符: {len(pos_sims)} != {n_pos}"
    print(f"[4] 合并+校验: {time.time()-t0:.2f}s | 负对={int(neg_c.sum()):,} ✓ 正对={len(pos_sims):,} ✓ "
          f"| 细直方图覆盖 {int(neg_f.sum()):,} 对 (sim>=0.2)")

    # top-N全局合并 (值即精确sim)
    top_v = top_i = top_j = None
    t0 = time.time()
    if do_extract:
        top_v = np.concatenate([f[5] for f in fetched])
        top_i = np.concatenate([f[6] for f in fetched])
        top_j = np.concatenate([f[7] for f in fetched])
        if len(top_v) > cap:
            sel = np.argpartition(-top_v, cap - 1)[:cap]
            top_v, top_i, top_j = top_v[sel], top_i[sel], top_j[sel]
    top_sorted_desc = np.sort(top_v)[::-1] if top_v is not None else None
    print(f"[5] top-N合并: {time.time()-t0:.2f}s"
          + (f" ({len(top_v):,}对, 精确覆盖FPIR>={len(top_v)/n_neg:.1e})" if top_v is not None else ""))

    # ---------- 6. TPIR@FPIR ----------
    targets = [float(x) for x in args.fpir_targets.split(',')]
    key_rows, pos_sorted, coarse, fine = eval_metrics(
        pos_sims, neg_c, neg_f, top_sorted_desc, n_neg, targets)
    print(f"\n[6] TPIR@FPIR 评估结果:")
    print(f"    {'FPIR':>8} | {'TPIR':>9} | {'阈值':>8} | 方法")
    print(f"    {'-'*8}-+-{'-'*9}-+-{'-'*8}-|---------")
    for r in key_rows:
        print(f"    {r['fpir']:>8.0e} | {r['tpir']*100:>8.3f}% | {r['threshold']:>8.4f} | {r['method']}")

    # ---------- 7. 样本对提取 ----------
    print(f"\n[7] 样本对提取:")
    t5 = next((r['threshold'] for r in key_rows if abs(r['fpir'] - 1e-5) < 1e-12),
              key_rows[-1]['threshold'])
    thr_above = t5 if args.extract_above == 'auto' else args.extract_above
    thr_below = thr_above if args.extract_below == 'auto' else args.extract_below

    if do_extract and args.extract_above != 'none':
        m = top_v > float(thr_above)
        b_idx = min(max(int((float(thr_above) - coarse.lo) / coarse.w), 0), coarse.nbins - 1)
        n_fp_total = int(coarse.cum_above[b_idx])
        i_o, j_o = order[top_i[m]], order[top_j[m]]
        sim_o = top_v[m]
        o = np.argsort(-sim_o)
        path = os.path.join(args.out_dir, f'neg_above_{thr_above}.pkl')
        export_pairs(path, i_o[o], j_o[o], sim_o[o], file_paths, args.with_paths,
                     f'neg_sim>{thr_above}')
        note = f' (受cap截断, 总数~{n_fp_total:,})' if int(m.sum()) >= cap else ''
        print(f"    负样本对 sim>{float(thr_above):.4f} (above/假阳性): {int(m.sum()):,}对{note} -> {path}")
        for k in range(min(3, len(sim_o))):
            print(f"      示例: sim={sim_o[o[k]]:.4f} | {file_paths[i_o[o[k]]]}")
            print(f"              vs {file_paths[j_o[o[k]]]}")

    if args.extract_below != 'none':
        m = pos_sims < float(thr_below)
        i_o, j_o = order[pos_i[m]], order[pos_j[m]]
        sim_o = pos_sims[m]
        o = np.argsort(sim_o)
        path = os.path.join(args.out_dir, f'pos_below_{thr_below}.pkl')
        export_pairs(path, i_o[o], j_o[o], sim_o[o], file_paths, args.with_paths,
                     f'pos_sim<{thr_below}')
        print(f"    正样本对 sim<{float(thr_below):.4f} (below/假阴性): {int(m.sum()):,}对 -> {path}")
        for k in range(min(3, len(sim_o))):
            print(f"      示例: sim={sim_o[o[k]]:.4f} | {file_paths[i_o[o[k]]]}")
            print(f"              vs {file_paths[j_o[o[k]]]}")

    # ---------- 8. 可视化 (后台线程, 与保存并行) ----------
    plot_thread = None
    if not args.no_plot:
        curve = tpir_fpir_curve(pos_sorted, coarse, fine, top_sorted_desc, n_neg)
        plot_thread = threading.Thread(
            target=plot_all,
            args=(args.out_dir, coarse, pos_sims, curve, key_rows, DEFAULT_FONT))
        plot_thread.start()

    # ---------- 9. 保存统计 ----------
    if args.save_npz:
        t0 = time.time()
        np.savez(os.path.join(args.out_dir, args.save_npz),
                 neg_coarse=neg_c, neg_fine=neg_f, pos_sims=pos_sims,
                 pos_i=pos_i, pos_j=pos_j,
                 **({} if top_v is None else
                    {'top_v': top_v, 'top_i': top_i, 'top_j': top_j}),
                 order=order)
        print(f"[9] 统计保存: {time.time()-t0:.2f}s -> {os.path.join(args.out_dir, args.save_npz)}")

    print(f"\n{'='*88}\n总耗时: {time.time()-t_total:.2f}s "
          f"(计算 {t_compute:.2f}s + 合并/指标/提取/绘图)\n{'='*88}")
    if plot_thread is not None:
        plot_thread.join()
        print(f"图表已生成: {args.out_dir}/sim_distribution.png, {args.out_dir}/tpir_fpir_curve.png")


if __name__ == '__main__':
    main()
