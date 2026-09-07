# -*- coding: utf-8 -*-
"""
faireval_lib.py — 人脸特征相似度评估核心库（自研，仅用环境已有库）

针对亿级样本对的设计:
1. 特征已L2归一化 -> 相似度 = 特征矩阵乘法 (fp32精确, 不用TF32)
2. 不保存 NxN 全量相似度矩阵: 按 block x block 分块只算上三角,
   每块实时量化为直方图(bin)计数 -> 内存 O(nbins), 与样本数N无关
3. 对角线块利用 S=A@A^T 精确对称性: 全块计数 - 对角线 - 对称除2,
   只需2次 bincount, 免去上三角掩码开销
4. 多卡并行: 行块轮询(round-robin)分给各GPU -> 三角工作区天然负载均衡;
   每个GPU跑独立子进程(gpu_worker.py), 无fork死锁风险
5. 样本对提取(错误分析): 二次遍历 + 流式top-K, 提取above/below阈值样本对

进程模型: 主进程只加载numpy数据; GPU计算全部发生在 gpu_worker.py 子进程中,
各子进程自行 import torch 并写结果npz, 主进程合并 —— 避免fork+cuda/线程死锁。
"""

import os
import sys
import json
import time
import pickle
import uuid
import subprocess
import numpy as np

# ------------------- 全局常量 -------------------
DATA_FILE = 's4_0618_enhance.pkl'
FONT_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'font', 'SourceHanSansSC-Normal.otf')
OUT_DIR = 'outputs'
GPU_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gpu_worker.py')
SIM_LO, SIM_HI = -1.0, 1.0          # 相似度范围
DEFAULT_NBINS = 200000              # 直方图bin数, 每个bin宽1e-5
DEFAULT_BLOCK = 8192                # 分块大小(行/列)
MAX_EXTRACT_DEFAULT = 300000        # 默认每方向最多保存样本对数


def load_data(path=DATA_FILE):
    """读取pkl特征文件, 返回 (feats, ids, paths, t_load)"""
    t0 = time.perf_counter()
    with open(path, 'rb') as f:
        feats_list, feats_flip, ids, paths = pickle.load(f)
    feats = np.ascontiguousarray(feats_list, dtype=np.float32)
    ids = np.ascontiguousarray(np.asarray(ids), dtype=np.int64)
    assert len(feats) == len(ids) == len(paths), '数据长度不一致!'
    assert feats.ndim == 2 and feats.shape[1] >= 64, '特征维度异常: %s' % (feats.shape,)
    return feats, ids, list(paths), time.perf_counter() - t0


def pair_counts(ids):
    """统计正/负样本对数: 正=同身份i<j, 负=异身份i<j"""
    N = len(ids)
    total_pairs = N * (N - 1) // 2
    counts = np.bincount(ids)
    pos = int(np.dot(counts.astype(np.int64), counts - 1) // 2)
    return total_pairs, pos, total_pairs - pos


def _mk_jobs(N, block):
    """上三角块任务: [(块行i, r0, r1, 块列j, c0, c1)], 仅 i<=j"""
    nblocks = (N + block - 1) // block
    bounds = [(k * block, min((k + 1) * block, N)) for k in range(nblocks)]
    jobs = []
    for i in range(nblocks):
        r0, r1 = bounds[i]
        for j in range(i, nblocks):
            c0, c1 = bounds[j]
            jobs.append((i, r0, r1, j, c0, c1))
    return jobs, bounds


def assign_row_blocks(N, block, gpus):
    """行块轮询分配(上三角负载均衡): 返回每gpu的行块索引列表"""
    nblk = (N + block - 1) // block
    return {g: list(range(g_i, nblk, len(gpus))) for g_i, g in enumerate(gpus)}


# =====================================================================
# GPU 核心计算 (在 gpu_worker 子进程内执行)
# =====================================================================
def _worker_hist_core(gpu_id, feats_np, ids_np, row_list, nbins):
    """单GPU: 上三角分块matmul -> 正/负相似度直方图计数"""
    import torch
    torch.set_grad_enabled(False)
    torch.cuda.set_device(gpu_id)
    F = torch.from_numpy(feats_np).cuda()
    IDS = torch.from_numpy(ids_np).cuda()
    pos_hist = torch.zeros(nbins, dtype=torch.int64, device='cuda')
    neg_hist = torch.zeros(nbins, dtype=torch.int64, device='cuda')
    scale = torch.tensor(0.5 * nbins, dtype=torch.float32, device='cuda')
    t_calc = 0.0
    t0 = time.perf_counter()
    for (i, r0, r1, j, c0, c1) in row_list:
        A = F[r0:r1]
        Bc = F[c0:c1]
        tS = time.perf_counter()
        S = A @ Bc.T                       # (m,n) fp32 精确
        S.add_(1.0).mul_(scale).clamp_(0, nbins - 1)      # S原地->bin索引
        bins = S.reshape(-1).to(torch.int64)              # 全部元素bin
        allh = torch.bincount(bins, minlength=nbins)
        eq = (IDS[r0:r1, None] == IDS[c0:c1][None, :]).reshape(-1)
        posfull = torch.bincount(bins[eq], minlength=nbins)  # 同身份计数(含对称/对角)
        if i == j:                          # 对角块: 去对角线项后对称除2
            dh = torch.bincount(S.diagonal().to(torch.int64), minlength=nbins)
            pos_hist += (posfull - dh) // 2
            neg_hist += (allh - posfull) // 2
        else:                               # 非对角块 r<块c: 全部元素都是合法对
            pos_hist += posfull
            neg_hist += allh - posfull
        t_calc += time.perf_counter() - tS
    torch.cuda.synchronize()
    return dict(gpu=gpu_id,
                pos=pos_hist.cpu().numpy(), neg=neg_hist.cpu().numpy(),
                t_calc=t_calc, t_wall=time.perf_counter() - t0)


def _worker_extract_core(gpu_id, feats_np, ids_np, row_list, threshold, dirs, max_keep):
    """单GPU: 提取 above(负对sim>t)/below(正对sim<t) 样本对, 流式局部top-K"""
    import torch
    torch.set_grad_enabled(False)
    torch.cuda.set_device(gpu_id)
    F = torch.from_numpy(feats_np).cuda()
    IDS = torch.from_numpy(ids_np).cuda()
    t0 = time.perf_counter()
    buf = {d: ([], [], []) for d in dirs}
    for (i, r0, r1, j, c0, c1) in row_list:
        A = F[r0:r1]
        Bc = F[c0:c1]
        S = A @ Bc.T
        eq = IDS[r0:r1, None] == IDS[c0:c1][None, :]
        if i == j:
            m, n = S.shape
            valid = torch.arange(m, device='cuda')[:, None] < \
                    torch.arange(n, device='cuda')[None, :]   # 上三角(不含对角)
        else:
            valid = None
        for d in dirs:
            if d == 'above':                # 负样本对 sim>threshold
                sel = (~eq) if valid is None else (valid & (~eq))
                sel = sel & (S > threshold)
            else:                           # 正样本对 sim<threshold
                sel = eq if valid is None else (valid & eq)
                sel = sel & (S < threshold)
            cnt = int(sel.sum().item())
            if cnt == 0:
                continue
            coords = sel.nonzero()
            vr = (coords[:, 0].to(torch.int32) + r0).cpu()
            vc = (coords[:, 1].to(torch.int32) + c0).cpu()
            vs = S[sel].to(torch.float32).cpu()
            rb, cb, sb = buf[d]
            rb.append(vr); cb.append(vc); sb.append(vs)
            if sum(x.numel() for x in rb) > max_keep:   # 流式裁剪保持全局top-K
                rows = torch.cat(rb); cols = torch.cat(cb); sims = torch.cat(sb)
                _, idx = torch.sort(sims, descending=(d == 'above'))
                idx = idx[:max_keep]
                buf[d] = ([rows[idx]], [cols[idx]], [sims[idx]])
    out = {}
    for d in dirs:
        if buf[d][0]:
            out[d] = (torch.cat(buf[d][0]).numpy(), torch.cat(buf[d][1]).numpy(),
                      torch.cat(buf[d][2]).numpy())
        else:
            out[d] = None
    return dict(gpu=gpu_id, extract=out, t_wall=time.perf_counter() - t0)


# =====================================================================
# 子进程编排 (gpu_worker.py)
# =====================================================================
def _launch(gpu, mode, out, extra):
    """启动单个 gpu_worker 子进程, 返回 (process, logpath)"""
    cmd = [sys.executable, GPU_WORKER, '--mode', mode,
           '--gpu', str(gpu), '--out', out] + extra
    log = out + '.log'
    with open(log, 'w') as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    return p, log


def run_hist_pass(gpus, nbins, block, N, feat_npz=None, data=None):
    """多GPU直方图统计主入口: 启动子进程->合并结果
    返回 dict(pos, neg, per_gpu, wall)"""
    if feat_npz is None and data is None:
        data = DATA_FILE
    rows_map = assign_row_blocks(N, block, gpus)
    t0 = time.perf_counter()
    procs, logs, outs = [], [], []
    for g in gpus:
        out = os.path.join(OUT_DIR, '_worker_hist_%d_%s.npz' % (g, uuid.uuid4().hex[:8]))
        extra = ['--nbins', str(nbins), '--block', str(block),
                 '--row-blocks', ','.join(map(str, rows_map[g]))]
        if feat_npz is not None:
            extra += ['--feat-npz', feat_npz]
        else:
            extra += ['--data-pkl', os.path.abspath(data)]
        p, log = _launch(g, 'hist', out, extra)
        procs.append(p); logs.append(log); outs.append(out)
    for p in procs:
        if p.wait() != 0:
            tail = open(logs[procs.index(p)]).read().strip().splitlines()[-15:]
            raise RuntimeError('GPU worker失败:\n' + '\n'.join(tail))
    wall = time.perf_counter() - t0
    pos = np.zeros(nbins, dtype=np.int64)
    neg = np.zeros(nbins, dtype=np.int64)
    per_gpu = []
    for g, out in zip(gpus, outs):
        d = np.load(out)
        pos += d['pos_hist']; neg += d['neg_hist']
        per_gpu.append((g, float(d['t_calc']), float(d['t_wall'])))
        os.remove(out)
    return dict(pos=pos, neg=neg, per_gpu=per_gpu, wall=wall)


def run_extract_pass(gpus, block, N, threshold, dirs, max_keep):
    """多GPU样本对提取主入口; 返回 [(gpu, dict方向->(rows,cols,sims)或None), ...]"""
    rows_map = assign_row_blocks(N, block, gpus)
    t0 = time.perf_counter()
    procs, logs, outs = [], [], []
    for g in gpus:
        out = os.path.join(OUT_DIR, '_worker_ext_%d_%s.npz' % (g, uuid.uuid4().hex[:8]))
        extra = ['--block', str(block), '--row-blocks', ','.join(map(str, rows_map[g])),
                 '--threshold', str(threshold), '--dirs', ','.join(dirs),
                 '--max', str(max_keep), '--data-pkl', os.path.abspath(DATA_FILE)]
        p, log = _launch(g, 'extract', out, extra)
        procs.append(p); logs.append(log); outs.append(out)
    for p in procs:
        if p.wait() != 0:
            tail = open(logs[procs.index(p)]).read().strip().splitlines()[-15:]
            raise RuntimeError('GPU worker失败:\n' + '\n'.join(tail))
    wall = time.perf_counter() - t0
    res = []
    for g, out in zip(gpus, outs):
        d = np.load(out, allow_pickle=False)
        gd = {}
        for dirn in dirs:
            key = dirn + '_rows'
            if key in d:
                gd[dirn] = (d[dirn + '_rows'], d[dirn + '_cols'], d[dirn + '_sims'])
            else:
                gd[dirn] = None
        res.append((g, gd, wall))
        os.remove(out)
    return res


# =====================================================================
# 指标计算 (直方图 -> TPIR@FPIR)
# =====================================================================
def bin_edges(nbins, lo=SIM_LO, hi=SIM_HI):
    return np.linspace(lo, hi, nbins + 1)


def threshold_at_fpir(neg_hist, total_neg, fpir_target, nbins):
    """反解阈值 t: FPIR(t)=P(neg sim>t)=fpir_target (bin内线性插值)"""
    edges = bin_edges(nbins)
    frac = np.concatenate([[0], np.cumsum(neg_hist)]).astype(np.float64) / total_neg
    y = 1.0 - float(fpir_target)               # 求 P(sim < t) = y
    k = min(max(int(np.searchsorted(frac, y, side='right')) - 1, 0), nbins - 1)
    f0, f1 = frac[k], frac[k + 1]
    if f1 - f0 <= 1e-15:
        return float(edges[k + 1])
    return float(min(max(edges[k] + (y - f0) / (f1 - f0) * (edges[k + 1] - edges[k]),
                         SIM_LO), SIM_HI))


def frac_above(hist, total, t, nbins):
    """P(sim>t): 分段线性CDF插值"""
    edges = bin_edges(nbins)
    cum = np.concatenate([[0], np.cumsum(hist)]).astype(np.float64) / total
    t = min(max(t, edges[0]), edges[-1])
    k = min(max(int(np.searchsorted(edges, t, side='right')) - 1, 0), nbins - 1)
    f = cum[k] + (cum[k + 1] - cum[k]) * (t - edges[k]) / max(edges[k + 1] - edges[k], 1e-12)
    return float(1.0 - f)


def compute_metrics(pos_hist, neg_hist, total_pos, total_neg, fpir_points,
                    nbins=DEFAULT_NBINS):
    """TPIR@FPIR 点表 + 稠密曲线"""
    points = []
    for p in fpir_points:
        thr = threshold_at_fpir(neg_hist, total_neg, p, nbins)
        points.append(dict(fpir=p, threshold=thr,
                           tpir=frac_above(pos_hist, total_pos, thr, nbins)))
    fpir_curve = np.geomspace(1e-6, 0.5, 500)
    thr_curve = np.array([threshold_at_fpir(neg_hist, total_neg, p, nbins)
                          for p in fpir_curve])
    tpir_curve = np.array([frac_above(pos_hist, total_pos, t, nbins)
                           for t in thr_curve])
    return dict(points=points, fpir_curve=fpir_curve, tpir_curve=tpir_curve,
                thr_curve=thr_curve)


# =====================================================================
# 数值自检: GPU直方图管线 vs CPU float64 全量直接计算
# =====================================================================
def selfcheck(gpu_id=0, n_subset=2000, fpir_points=(1e-2, 1e-3, 1e-4), nbins=200000):
    feats, ids, paths, _ = load_data()
    Fs, Is = np.ascontiguousarray(feats[:n_subset]), ids[:n_subset]
    tot, pos, neg = pair_counts(Is)
    print('  [selfcheck] 子集N=%d 正对=%s 负对=%s' % (n_subset, fmt_pair(pos), fmt_pair(neg)))

    # --- CPU float64 精确参考 ---
    sims64 = Fs.astype(np.float64) @ Fs.astype(np.float64).T
    iu = np.triu_indices(n_subset, 1)
    s_all = sims64[iu]
    eq = Is[iu[0]] == Is[iu[1]]
    s_pos, s_neg = s_all[eq], s_all[~eq]
    assert len(s_pos) == pos and len(s_neg) == neg, '子集正负计数不符!'
    ref = {}
    for p in fpir_points:
        order = np.sort(s_neg)[::-1]
        k = max(1, min(int(round(p * neg)), len(order) - 1))
        t = (float(order[k - 1]) + float(order[k])) / 2
        ref[p] = (t, float((s_pos > t).mean()))

    # --- GPU 直方图管线 (独立子进程, 传入子集npz) ---
    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = os.path.join(OUT_DIR, '_selfcheck_feats_%s.npz' % uuid.uuid4().hex[:8])
    np.savez(tmp, feats=Fs, ids=Is)
    try:
        res = run_hist_pass([gpu_id], nbins, DEFAULT_BLOCK, n_subset, feat_npz=os.path.abspath(tmp))
    finally:
        os.remove(tmp)
    ph, nh = res['pos'], res['neg']
    ok = int(ph.sum()) == pos and int(nh.sum()) == neg
    print('  [selfcheck] GPU直方图计数一致: 正=%d/%d 负=%d/%d  %s' %
          (int(ph.sum()), pos, int(nh.sum()), neg, 'OK' if ok else 'FAIL'))
    worst = 0.0
    for p in fpir_points:
        thr = threshold_at_fpir(nh, neg, p, nbins)
        tpir = frac_above(ph, pos, thr, nbins)
        r_t, r_tpir = ref[p]
        d1, d2 = abs(thr - r_t), abs(tpir - r_tpir)
        worst = max(worst, d1, d2)
        print('  [selfcheck] FPIR=%.0e 阈值 GPU=%.6f/CPU=%.6f |Δ|=%.1e | '
              'TPIR GPU=%.6f/CPU=%.6f |Δ|=%.1e' % (p, thr, r_t, d1, tpir, r_tpir, d2))
    passed = ok and worst < 2e-4
    print('  [selfcheck] 结论: %s (最大偏差 %.1e < 2e-4)' % ('PASS' if passed else 'FAIL', worst))
    return passed


# =====================================================================
# 可视化 (中文字体)
# =====================================================================
def setup_chinese_font():
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager, rcParams
    font_manager.fontManager.addfont(FONT_FILE)
    name = font_manager.FontProperties(fname=FONT_FILE).get_name()
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = [name, 'DejaVu Sans']
    rcParams['axes.unicode_minus'] = False
    return name


def plot_distributions(pos_hist, neg_hist, total_pos, total_neg, metrics, nbins, out_png):
    """图1: 正/负样本相似度分布 + 阈值标注"""
    import matplotlib.pyplot as plt
    edges = bin_edges(nbins)
    centers = (edges[:-1] + edges[1:]) / 2
    w = edges[1] - edges[0]
    dp = pos_hist / max(total_pos * w, 1.0)
    dn = neg_hist / max(total_neg * w, 1.0)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.6))
    ax = axes[0]
    ax.plot(centers, dp, color='#d62728', lw=1.4, label='正样本对（同身份）')
    ax.plot(centers, dn, color='#1f77b4', lw=1.1, alpha=0.9, label='负样本对（不同身份）')
    ax.set_xlabel('相似度（余弦）'); ax.set_ylabel('概率密度')
    ax.set_title('正/负样本对相似度分布')
    ax.legend(); ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(centers, dn, color='#1f77b4', lw=1.1, label='负样本对')
    for pt in metrics['points']:
        ax.axvline(pt['threshold'], color='#2ca02c', ls='--', lw=1.1,
                   label='FPIR=%.0e 阈值(%.3f)' % (pt['fpir'], pt['threshold']))
    ax.set_xlabel('相似度（余弦）'); ax.set_ylabel('概率密度')
    ax.set_title('负样本分布右尾与评估阈值')
    ax.legend(fontsize=8.5); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def plot_curve(metrics, out_png):
    """图2: TPIR@FPIR 曲线 (log-x)"""
    import matplotlib.pyplot as plt
    fp, tp = metrics['fpir_curve'], metrics['tpir_curve']
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    ax.semilogx(fp, tp * 100, color='#d62728', lw=2, label='TPIR@FPIR')
    ax.semilogx(fp, fp * 100, ':', color='gray', lw=1.2, label='随机基线（TPIR=FPIR）')
    for pt in metrics['points']:
        ax.plot(pt['fpir'], pt['tpir'] * 100, 'o', ms=7, mfc='#2ca02c', mec='k')
        ax.annotate('FPIR=%.0e\nTPIR=%.2f%%' % (pt['fpir'], pt['tpir'] * 100),
                    (pt['fpir'], pt['tpir'] * 100), textcoords='offset points',
                    xytext=(6, -18 if pt['fpir'] < 1e-3 else 8), fontsize=8.5)
    ax.set_xscale('log')
    ax.set_xlabel('FPIR（负样本误接受率）')
    ax.set_ylabel('TPIR（正样本通过率, %）')
    ax.set_title('TPIR @ FPIR 评估曲线')
    ax.set_ylim(-2, 105)
    ax.grid(which='both', alpha=0.3)
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


# =====================================================================
# 通用辅助
# =====================================================================
def save_json(path, obj):
    def _c(o):
        if isinstance(o, dict):
            return {str(k): _c(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_c(x) for x in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o
    with open(path, 'w') as f:
        json.dump(_c(obj), f, ensure_ascii=False, indent=2)
    return path


def fmt_pair(n):
    return '{:,}'.format(int(n))
