#!/usr/bin/env python3
"""第四步：性能优化版 v3（大块 GEMM + 对级负载均衡 + 多线程多卡）

相对 step3 的关键优化（依据实测剖析）：
1. 块大小 8192 → 16384：块数 325 → 91，Python/内核启动开销摊销 ~4 倍；
   此前剖析显示每块 ~5ms 的 Python 调度开销（GIL 串行化），是线程版多卡慢的元凶；
2. 对级贪心负载均衡（91 个 (i,j) 块按元素数分配），比行级更均匀；
3. 转置块先 contiguous 再 GEMM（cuBLAS 快路径，fp32 ~45 TFLOPS）；
4. int32 bincount（torch 2.12 CUDA 支持，比 int64 快 ~35%）；
5. 正样本对在父进程串行计算（~1s），与 GPU 直方图并发重叠；
6. worker 预热：特征加载后先跑一次小 GEMM/bincount，避免首块懒加载计入耗时；
7. --core 模式：只跑核心计算（直方图+正样本+指标），用于单卡/多卡公平对比。

用法：
  python step4_eval_optimized.py --core --gpus 0        # 单卡核心
  python step4_eval_optimized.py --core                 # 7 卡核心
  python step4_eval_optimized.py                        # 完整流程（含提取+绘图）
"""
import argparse
import concurrent.futures as cf
import json
import os
import pickle
import time

import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)

_GLOBAL_F = None
_GLOBAL_IDS = None
_GLOBAL_F_PIN = None
_ORDER_PAD = None
_C_G = None
_N_GROUPS = None


def load_data(path, use_cache):
    t0 = time.time()
    if use_cache and os.path.exists('feats_f32.npy') and os.path.exists('ids.npy'):
        F = np.load('feats_f32.npy')
        ids = np.load('ids.npy')
        paths = open('paths.txt', encoding='utf-8').read().splitlines()
    else:
        with open(path, 'rb') as f:
            F, _, ids, paths = pickle.load(f)
        F = np.asarray(F, dtype=np.float32)
        ids = np.asarray(ids, dtype=np.int64)
        paths = list(paths)
    norms = np.linalg.norm(F, axis=1)
    if np.abs(norms.mean() - 1.0) > 1e-3 or norms.std() > 1e-3:
        print('[warn] 特征未严格 L2 归一化，进行归一化处理')
        F = F / norms[:, None]
    return F, ids, paths, time.time() - t0


def block_rows(N, B):
    return [(s, min(s + B, N)) for s in range(0, N, B)]


def block_pairs(rows):
    """上三角块对列表 (i, j, r0, r1, c0, c1, weight)。"""
    pairs = []
    for i, (r0, r1) in enumerate(rows):
        for j in range(i, len(rows)):
            c0, c1 = rows[j]
            if i == j:
                w = (r1 - r0) * (r1 - r0 + 1) // 2
            else:
                w = (r1 - r0) * (c1 - c0)
            pairs.append((i, j, r0, r1, c0, c1, w))
    return pairs


def assign_pairs_to_gpus(pairs, ngpu):
    """对级贪心负载均衡。"""
    order = sorted(range(len(pairs)), key=lambda k: -pairs[k][6])
    loads = [0] * ngpu
    assign = [[] for _ in range(ngpu)]
    for k in order:
        g = int(np.argmin(loads))
        loads[g] += pairs[k][6]
        assign[g].append(pairs[k])
    return assign, loads


# ----------------------------------------------------------------------------
# GPU worker（线程版；每个 worker 持有自己 GPU 上的特征缓存）
# ----------------------------------------------------------------------------
_FEATS_CACHE = {}
_BT_CACHE = {}
_BLK_CACHE = {}      # (gpu, c0, c1) -> 该行块的设备张量（流式传输缓存）
_BLK_STREAMS = {}    # (gpu, c0, c1) -> 传输流


def _ensure_blocks(gpu, ranges):
    """异步流式传输若干行块（每个块独立 CUDA stream，与计算重叠）。"""
    dev = f'cuda:{gpu}'
    for c0, c1 in ranges:
        key = (gpu, c0, c1)
        if key not in _BLK_CACHE:
            s = torch.cuda.Stream(device=dev)
            with torch.cuda.stream(s):
                blk = _GLOBAL_F_PIN[c0:c1].to(dev, non_blocking=True)
            _BLK_CACHE[key] = blk
            _BLK_STREAMS[key] = s


def _wait_block(gpu, c0, c1):
    key = (gpu, c0, c1)
    torch.cuda.current_stream(f'cuda:{gpu}').wait_stream(_BLK_STREAMS[key])
    return _BLK_CACHE[key]


def _get_bt(gpu, c0, c1):
    """列块转置缓存（基于流式传输的行块缓存）。"""
    key = (gpu, c0, c1)
    if key not in _BT_CACHE:
        _BT_CACHE[key] = _BLK_CACHE[key].t().contiguous()
    return _BT_CACHE[key]


def gpu_hist_worker(gpu, my_pairs, scale):
    torch.set_num_threads(1)
    dev = f'cuda:{gpu}'
    torch.cuda.set_device(gpu)
    comp = torch.cuda.Stream()               # 专用计算流：避开 legacy 默认流的全流隐式同步
    ranges = sorted({(r0, r1) for (_, _, r0, r1, _, _, _) in my_pairs}
                    | {(c0, c1) for (_, _, _, _, c0, c1, _) in my_pairs})
    _ensure_blocks(gpu, ranges)              # 提交全部异步拷贝（传输与计算重叠）
    dbg = 'STEP4_DEBUG' in os.environ
    if dbg:
        ts = [time.time()]
    nbins = 2 * scale + 1
    with torch.cuda.stream(comp):
        # 预热：首个完整块上跑一次全尺寸 GEMM+量化+bincount，触发懒加载
        c0, c1 = ranges[0]
        comp.wait_stream(_BLK_STREAMS[(gpu, c0, c1)])
        W = _BLK_CACHE[(gpu, c0, c1)]
        C = W @ W.t()
        q = (C * scale).round().clamp(-scale, scale).to(torch.int32) + scale
        torch.bincount(q.reshape(-1), minlength=nbins)
        if dbg:
            ts.append(time.time())
        hist = torch.zeros(nbins, dtype=torch.int64, device=dev)
        t0 = time.time()
        for (i, j, r0, r1, c0, c1, w) in my_pairs:
            comp.wait_stream(_BLK_STREAMS[(gpu, r0, r1)])
            comp.wait_stream(_BLK_STREAMS[(gpu, c0, c1)])
            A = _BLK_CACHE[(gpu, r0, r1)]
            Bt = _get_bt(gpu, c0, c1)
            C = A @ Bt                          # (nr, nc)
            if i == j:
                m = torch.triu(torch.ones(r1 - r0, c1 - c0, dtype=torch.bool, device=dev), 1)
                v = C[m].reshape(-1)
            else:
                v = C.reshape(-1)
            q = (v * scale).round().clamp(-scale, scale).to(torch.int32) + scale
            hist += torch.bincount(q, minlength=nbins)
        if dbg:
            ts.append(time.time())
    torch.cuda.synchronize(dev)
    if dbg:
        print(f'[dbg] gpu{gpu} warm{ts[1]-ts[0]:.2f}s loop{ts[2]-ts[1]:.2f}s '
              f'sync{time.time()-ts[2]:.2f}s', flush=True)
    return hist.cpu().numpy(), len(my_pairs), time.time() - t0


def gpu_extract_worker(gpu, my_pairs, scale, t_above, max_above, t_below, max_below, p_below):
    torch.set_num_threads(1)
    dev = f'cuda:{gpu}'
    torch.cuda.set_device(gpu)
    comp = torch.cuda.Stream()
    ranges = sorted({(r0, r1) for (_, _, r0, r1, _, _, _) in my_pairs}
                    | {(c0, c1) for (_, _, _, _, c0, c1, _) in my_pairs})
    _ensure_blocks(gpu, ranges)              # 直方图 pass 已缓存大部分块
    ids_g = torch.from_numpy(_GLOBAL_IDS).to(dev)
    above_i, above_j, above_s = [], [], []
    below_i, below_j, below_s = [], [], []
    n_above = n_below = 0
    t0 = time.time()
    with torch.cuda.stream(comp):
        for (i, j, r0, r1, c0, c1, w) in my_pairs:
            comp.wait_stream(_BLK_STREAMS[(gpu, r0, r1)])
            comp.wait_stream(_BLK_STREAMS[(gpu, c0, c1)])
            A = _BLK_CACHE[(gpu, r0, r1)]
            Bt = _get_bt(gpu, c0, c1)
            C = A @ Bt
            nr, nc = C.shape
            id_neq = ids_g[r0:r1][:, None] != ids_g[c0:c1][None, :]
            if i == j:
                mask = torch.triu(torch.ones(nr, nc, dtype=torch.bool, device=dev), 1)
            else:
                mask = torch.ones(nr, nc, dtype=torch.bool, device=dev)
            if n_above < max_above:
                m = mask & id_neq & (C > t_above)
                idx = m.nonzero()
                if idx.numel() > 0:
                    gi = r0 + idx[:, 0].to(torch.int32)
                    gj = c0 + idx[:, 1].to(torch.int32)
                    take = min(idx.shape[0], max_above - n_above)
                    above_i.append(gi[:take].cpu())
                    above_j.append(gj[:take].cpu())
                    above_s.append(C[m][:take].cpu())
                    n_above += take
            if n_below < max_below:
                rnd = torch.rand(nr, nc, device=dev) < p_below
                m2 = mask & id_neq & (C < t_below) & rnd
                idx2 = m2.nonzero()
                if idx2.numel() > 0:
                    gi = r0 + idx2[:, 0].to(torch.int32)
                    gj = c0 + idx2[:, 1].to(torch.int32)
                    take = min(idx2.shape[0], max_below - n_below)
                    below_i.append(gi[:take].cpu())
                    below_j.append(gj[:take].cpu())
                    below_s.append(C[m2][:take].cpu())
                    n_below += take
    torch.cuda.synchronize(dev)
    res = {}
    if above_i:
        res['ai'] = torch.cat(above_i).numpy()
        res['aj'] = torch.cat(above_j).numpy()
        res['as'] = torch.cat(above_s).numpy()
    if below_i:
        res['bi'] = torch.cat(below_i).numpy()
        res['bj'] = torch.cat(below_j).numpy()
        res['bs'] = torch.cat(below_s).numpy()
    return res, n_above, n_below, time.time() - t0


# ----------------------------------------------------------------------------
# 正样本对计算：GPU padded batched-GEMM（作为线程池的附加任务，避免主线程 GIL 干扰）
# ----------------------------------------------------------------------------
def gpu_pos_worker(gpu, scale):
    torch.set_num_threads(1)
    dev = f'cuda:{gpu}'
    if ('full', gpu) not in _FEATS_CACHE:
        _FEATS_CACHE[('full', gpu)] = _GLOBAL_F_PIN.to(dev, non_blocking=True)
    Fg = _FEATS_CACHE[('full', gpu)]
    torch.cuda.synchronize(dev)
    order_pad = torch.from_numpy(_ORDER_PAD).to(dev)
    c_g = torch.from_numpy(_C_G).to(dev)
    Fp = Fg[order_pad]                                   # (G*38, 512)
    Fp3 = Fp.view(_N_GROUPS, 38, 512)
    S = torch.bmm(Fp3, Fp3.transpose(1, 2))              # (G, 38, 38)
    ll = torch.arange(38, device=dev)
    valid = ll[None, :] < c_g[:, None]                   # (G, 38)
    mask3 = valid[:, :, None] & valid[:, None, :] & (ll[None, None, :] > ll[None, :, None])
    idx = mask3.nonzero()                                # (total_pos, 3)
    sims = S[mask3]
    gi = order_pad[idx[:, 0] * 38 + idx[:, 1]].to(torch.int32)
    gj = order_pad[idx[:, 0] * 38 + idx[:, 2]].to(torch.int32)
    sims, perm = torch.sort(sims)                        # 按相似度排序，索引同步
    gi = gi[perm]
    gj = gj[perm]
    q = (sims * scale).round().clamp(-scale, scale).to(torch.int32) + scale
    hist = torch.bincount(q, minlength=2 * scale + 1)
    total_pos = int(idx.shape[0])
    return (hist.cpu().numpy(), gi.cpu().numpy(), gj.cpu().numpy(),
            sims.cpu().numpy(), total_pos)


def prepare_pos_indices(ids):
    """构造按身份分组的 padded 索引（pad 位置指向 0 号样本，其相似度会被 mask 剔除）。"""
    order = np.argsort(ids, kind='stable')
    starts = np.r_[0, np.flatnonzero(np.diff(ids[order])) + 1, len(ids)]
    c_g = (starts[1:] - starts[:-1]).astype(np.int32)
    ng = len(c_g)
    off = np.tile(np.arange(38, dtype=np.int64), ng)
    base = np.repeat(starts[:-1].astype(np.int64), 38)
    ok = off < np.repeat(c_g.astype(np.int64), 38)
    pad_idx = np.where(ok, base + off, 0)
    order_pad = order[pad_idx].astype(np.int64)
    return order_pad, c_g, ng


# ----------------------------------------------------------------------------
# 指标
# ----------------------------------------------------------------------------
def compute_metrics(full_hist, pos_hist, pos_sims, targets, scale):
    nbins = 2 * scale + 1
    total_pairs = int(full_hist.sum())
    total_pos = len(pos_sims)
    total_neg = total_pairs - total_pos
    neg_hist = np.maximum(full_hist - pos_hist, 0)
    cneg = neg_hist[::-1].cumsum()[::-1]
    pos_sorted = np.sort(pos_sims)
    edges = (np.arange(nbins, dtype=np.float64) - scale) / scale
    cnt_above = total_pos - np.searchsorted(pos_sorted, edges, side='right')
    results = []
    for tgt in targets:
        need = tgt * total_neg
        k = int(np.searchsorted(-cneg, -need, side='left'))
        k = min(max(k, 0), nbins - 1)
        th = float(edges[k])
        tpir = float(cnt_above[k] / total_pos)
        fpir = float(cneg[k] / total_neg)
        results.append(dict(target=tgt, threshold=th, fpir_actual=fpir, tpir=tpir))
    curve = dict(fpir=(cneg / total_neg)[::-1].astype(np.float64),
                 tpir=(cnt_above / total_pos)[::-1].astype(np.float64))
    return results, neg_hist, cneg, cnt_above, curve, total_pairs, total_pos, total_neg


# ----------------------------------------------------------------------------
# 绘图（中文字体）
# ----------------------------------------------------------------------------
def setup_font():
    os.environ.setdefault('MPLCONFIGDIR', os.path.abspath('.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.font_manager as fm
    for f in ['font/SourceHanSansSC-Normal.otf', 'SourceHanSansSC-Normal.otf']:
        if os.path.exists(f):
            fm.fontManager.addfont(f)
            break
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Source Han Sans SC'
    plt.rcParams['axes.unicode_minus'] = False
    return plt


def plot_figures(neg_hist, pos_hist, cneg, curve, results, scale, total_pos, total_neg, outdir):
    plt = setup_font()
    nbins = 2 * scale + 1
    bw = 1.0 / scale
    centers = (np.arange(nbins, dtype=np.float64) - scale) / scale + bw / 2

    fig, ax = plt.subplots(figsize=(10, 6))
    pos_d = pos_hist / (total_pos * bw)
    neg_d = neg_hist / (total_neg * bw)
    ax.semilogy(centers, pos_d, color='#2ca02c', lw=1.6, label=f'正样本 (n={total_pos:,})')
    ax.semilogy(centers, neg_d, color='#d62728', lw=1.6, label=f'负样本 (n={total_neg:,})')
    for r in results:
        ax.axvline(r['threshold'], color='gray', ls='--', lw=0.7, alpha=0.7)
        ax.text(r['threshold'], ax.get_ylim()[1] * 0.25, f'FPIR={r["target"]:.0e}',
                rotation=90, fontsize=8, va='top', ha='right', color='gray')
    ax.set_xlim(-0.6, 1.0)
    ax.set_xlabel('相似度 (cosine similarity)')
    ax.set_ylabel('概率密度（对数轴）')
    ax.set_title('正负样本相似度分布图')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    p1 = os.path.join(outdir, 'fig_sim_distribution.png')
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    xs = curve['fpir']
    ys = curve['tpir']
    keep = xs > 0
    idx = np.unique(np.linspace(0, keep.sum() - 1, 3000).astype(int))
    ax.plot(xs[keep][idx], ys[keep][idx], color='#1f77b4', lw=1.8)
    ax.set_xscale('log')
    for r in results:
        ax.scatter([r['target']], [r['tpir']], color='red', zorder=5, s=30)
        ax.annotate(f'FPIR={r["target"]:.0e}\nTPIR={r["tpir"]*100:.2f}%',
                    (r['target'], r['tpir']), xytext=(12, 8), textcoords='offset points',
                    fontsize=8, color='red')
    ax.set_xlim(1e-7, 1.0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('误识率 FPIR（对数轴）')
    ax.set_ylabel('通过率 TPIR')
    ax.set_title('TPIR@FPIR 评估曲线')
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    p2 = os.path.join(outdir, 'fig_tpir_fpir_curve.png')
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    return p1, p2


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def preinit_contexts(gpus):
    """预初始化各 GPU 主上下文（驱动全局锁串行化 ~0.1-0.2s/卡），
    与数据加载并行执行以隐藏该成本。"""
    for g in gpus:
        torch.cuda.set_device(g)
        _ = torch.zeros(1, device=f'cuda:{g}')
        torch.cuda.synchronize(f'cuda:{g}')


def main():
    global _GLOBAL_F, _GLOBAL_IDS, _GLOBAL_F_PIN, _ORDER_PAD, _C_G, _N_GROUPS
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='s4_0618_enhance.pkl')
    ap.add_argument('--use-cache', action='store_true')
    ap.add_argument('--gpus', default='0,1,2,3,4,5,6')
    ap.add_argument('--block', type=int, default=16384)
    ap.add_argument('--bin-scale', type=int, default=20000)
    ap.add_argument('--fpir-targets', default='1e-5,1e-4,1e-3,1e-2')
    ap.add_argument('--core', action='store_true', help='只跑核心计算（公平对比单卡/多卡）')
    ap.add_argument('--above', type=float, default=None)
    ap.add_argument('--below', type=float, default=None)
    ap.add_argument('--max-above', type=int, default=20000000)
    ap.add_argument('--max-below', type=int, default=2000)
    ap.add_argument('--outdir', default='results')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    gpus = [int(x) for x in args.gpus.split(',') if x.strip() != '']
    targets = [float(x) for x in args.fpir_targets.split(',')]
    timing = {}
    t_all = time.time()

    # 限制父进程 BLAS 线程（正样本为大量小 GEMM，多线程无益且干扰 GPU worker 调度）
    os.environ['OPENBLAS_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['OMP_NUM_THREADS'] = '4'

    print('=' * 78)
    print(f'大规模人脸特征相似度评估系统（优化版v5 | {len(gpus)} GPU 大块并行' +
          (' | core 模式' if args.core else '') + f' | block={args.block}）')
    print('=' * 78)

    # 1. 数据加载（与 CUDA 上下文预初始化并行，互相隐藏延迟）
    import threading
    ctx_t = threading.Thread(target=preinit_contexts, args=(gpus,))
    ctx_t.start()
    F, ids, paths, t_load = load_data(args.data, args.use_cache)
    timing['load'] = t_load
    N = len(ids)
    total_pairs = N * (N - 1) // 2
    _GLOBAL_F, _GLOBAL_IDS = F, ids
    # 原位注册 pinned 内存（cudaHostRegister，无拷贝）：多卡并发 H2D 走 DMA 快路径
    t0 = time.time()
    try:
        torch.cuda.cudart().cudaHostRegister(F.ctypes.data, F.nbytes, 0)
        _GLOBAL_F_PIN = torch.from_numpy(F)
        timing['pin'] = time.time() - t0
        print(f'[1] 数据加载 {t_load:.2f}s | N={N:,} | 总样本对数 {total_pairs:,} '
              f'| pinned注册 {timing["pin"]:.2f}s')
    except Exception as e:
        _GLOBAL_F_PIN = torch.from_numpy(F)
        print(f'[1] 数据加载 {t_load:.2f}s | N={N:,} | 总样本对数 {total_pairs:,} '
              f'| pinned注册失败({e}), 退回可分页拷贝')

    # 2. 块对规划 + 对级负载均衡 + 正样本分组索引
    rows = block_rows(N, args.block)
    pairs = block_pairs(rows)
    assign, loads = assign_pairs_to_gpus(pairs, len(gpus))
    _ORDER_PAD, _C_G, _N_GROUPS = prepare_pos_indices(ids)
    ctx_t.join()
    print(f'[2] 块规划: {len(rows)} 行块 → {len(pairs)} 个上三角块对 | '
          f'每卡块数={[len(a) for a in assign]} | 负载={[l // 10**6 for l in loads]}M元素')

    # 3+4. GPU 直方图（多线程多卡）+ GPU 正样本（附加任务）——主线程全程空闲，无 GIL 干扰
    full_hist = np.zeros(2 * args.bin_scale + 1, dtype=np.int64)
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        hist_futs = [ex.submit(gpu_hist_worker, gpus[g], assign[g], args.bin_scale)
                     for g in range(len(gpus))]
        pos_fut = ex.submit(gpu_pos_worker, gpus[0], args.bin_scale)
        wtimes = []
        for g, f in enumerate(hist_futs):
            h, nb, t = f.result()
            full_hist += h
            wtimes.append(t)
            print(f'    GPU {gpus[g]}: {nb} 块, {t:.2f}s')
        t0p = time.time()
        pos_hist, gi, gj, gs, total_pos = pos_fut.result()
        timing['pos_pass'] = time.time() - t0p
        print(f'    正样本对(GPU baddbmm)完成 {total_pos:,} 对, {timing["pos_pass"]:.2f}s')
        timing['hist_pass'] = time.time() - t0
        timing['hist_worker_max'] = max(wtimes)

    # 5. 指标
    results, neg_hist, cneg, cnt_above, curve, tp_pairs, tp_pos, tp_neg = compute_metrics(
        full_hist, pos_hist, gs, targets, args.bin_scale)
    print(f'[5] TPIR@FPIR 结果 (正样本对 {total_pos:,} | 负样本对 {tp_neg:,})')
    for r in results:
        print(f'    TPIR @ FPIR={r["target"]:.0e} : {r["tpir"]*100:.2f}% '
              f'(阈值 {r["threshold"]:.4f}, 实际FPIR {r["fpir_actual"]:.2e})')
    expect = {1e-5: (0.60, 0.65), 1e-4: (0.82, 0.85), 1e-3: (0.90, 0.93), 1e-2: (0.95, 0.97)}
    check = {}
    for r in results:
        lo, hi = expect.get(r['target'], (None, None))
        if lo is not None:
            ok = lo - 0.05 <= r['tpir'] <= hi + 0.05
            check[f'{r["target"]:.0e}'] = dict(expected=[lo, hi], got=r['tpir'], ok=bool(ok))
    print(f'    预期范围校验: {"全部 PASS" if all(v["ok"] for v in check.values()) else "存在 FAIL"}')

    # 6. 提取 pass（复用线程池 worker 的 GPU 特征缓存）
    extraction = {}
    t_above = args.above if args.above is not None else results[0]['threshold']
    t_below = args.below if args.below is not None else results[0]['threshold']
    figs = None
    if not args.core:
        print(f'[6] 样本对提取 (above>{t_above:.4f} | below<{t_below:.4f}) ...')
        p_below = min(max(args.max_below / max(tp_neg, 1), 0.0), 0.5)
        t0 = time.time()
        with cf.ThreadPoolExecutor(max_workers=len(gpus)) as ex:
            futs = [ex.submit(gpu_extract_worker, gpus[g], assign[g], args.bin_scale,
                              t_above, args.max_above, t_below, args.max_below, p_below)
                    for g in range(len(gpus))]
            agg = {}
            n_above = n_below = 0
            for f in futs:
                r, na, nb, t = f.result()
                n_above += na
                n_below += nb
                for k, v in r.items():
                    agg.setdefault(k, []).append(v)
        timing['extract'] = time.time() - t0
        truncated_above = n_above >= args.max_above
        truncated_below = n_below >= args.max_below
        n_above_saved = 0
        if 'ai' in agg:
            ai = np.concatenate(agg['ai'])[:args.max_above]
            aj = np.concatenate(agg['aj'])[:args.max_above]
            as_ = np.concatenate(agg['as'])[:args.max_above]
            np.savez(os.path.join(args.outdir, 'above_neg_pairs.npz'),
                     i=ai, j=aj, sim=as_, id_i=ids[ai], id_j=ids[aj])
            n_above_saved = len(ai)
            with open(os.path.join(args.outdir, 'above_neg_pairs_sample.csv'), 'w') as f:
                f.write('i,j,id_i,id_j,sim,path_i,path_j\n')
                for a, b, s in zip(ai[:200], aj[:200], as_[:200]):
                    f.write(f'{a},{b},{ids[a]},{ids[b]},{s:.6f},{paths[a]},{paths[b]}\n')
        mask_fr = gs < t_below
        fr_i = gi[mask_fr]
        fr_j = gj[mask_fr]
        fr_s = gs[mask_fr]
        np.savez(os.path.join(args.outdir, 'below_pos_pairs.npz'),
                 i=fr_i, j=fr_j, sim=fr_s, id_i=ids[fr_i], id_j=ids[fr_j])
        with open(os.path.join(args.outdir, 'below_pos_pairs_sample.csv'), 'w') as f:
            f.write('i,j,id_i,id_j,sim,path_i,path_j\n')
            for a, b, s in zip(fr_i[:200], fr_j[:200], fr_s[:200]):
                f.write(f'{a},{b},{ids[a]},{ids[b]},{s:.6f},{paths[a]},{paths[b]}\n')
        n_below_saved = 0
        if 'bi' in agg:
            bi = np.concatenate(agg['bi'])[:args.max_below]
            bj = np.concatenate(agg['bj'])[:args.max_below]
            bs = np.concatenate(agg['bs'])[:args.max_below]
            np.savez(os.path.join(args.outdir, 'below_neg_sample.npz'),
                     i=bi, j=bj, sim=bs, id_i=ids[bi], id_j=ids[bj])
            n_below_saved = len(bi)
        extraction = dict(
            above_threshold=float(t_above), above_neg_pairs_saved=n_above_saved,
            above_truncated=bool(truncated_above),
            below_threshold=float(t_below), below_pos_rejects_saved=int(mask_fr.sum()),
            below_neg_sampled=n_below_saved, below_truncated=bool(truncated_below))
        print(f'    提取耗时 {timing["extract"]:.2f}s | above负样本对 {n_above_saved:,} '
              f'| below正样本对(漏识) {int(mask_fr.sum()):,} | below负样本抽样 {n_below_saved:,}')

        # 7. 绘图
        t0 = time.time()
        figs = plot_figures(neg_hist, pos_hist, cneg, curve, results, args.bin_scale,
                            total_pos, tp_neg, args.outdir)
        timing['plot'] = time.time() - t0
        print(f'[7] 图表已保存 ({timing["plot"]:.2f}s)')

    # 8. 保存
    np.savez_compressed(os.path.join(args.outdir, 'histograms.npz'),
                        full=full_hist, pos=pos_hist, neg=neg_hist, bin_scale=args.bin_scale)
    np.save(os.path.join(args.outdir, 'pos_pairs_all.npy'), np.stack([gi, gj, gs], axis=1))
    np.save(os.path.join(args.outdir, 'curve_fpir.npy'), curve['fpir'])
    np.save(os.path.join(args.outdir, 'curve_tpir.npy'), curve['tpir'])
    timing['total'] = time.time() - t_all
    metrics = dict(
        code='step4_eval_optimized.py', version='v5', data=args.data, N=int(N),
        total_pairs=int(total_pairs), total_pos=int(total_pos), total_neg=int(tp_neg),
        bin_scale=args.bin_scale, block=args.block, gpus=gpus, core=args.core,
        eval_points=results, expected_range_check=check, extraction=extraction,
        timings={k: round(v, 3) for k, v in timing.items()}, figures=figs)
    with open(os.path.join(args.outdir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print('=' * 78)
    print(f'完成！总耗时 {timing["total"]:.2f}s | 直方图 {timing["hist_pass"]:.2f}s '
          f'(worker最慢 {timing["hist_worker_max"]:.2f}s) | 正样本 {timing["pos_pass"]:.2f}s')
    print(f'指标: {os.path.join(args.outdir, "metrics.json")}')


if __name__ == '__main__':
    main()
