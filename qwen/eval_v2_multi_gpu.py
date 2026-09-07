#!/usr/bin/env python3
"""
第三步：v2 多卡并行最终版
特性：
  1. 多卡并行（默认7卡，torch.multiprocessing spawn），按每行工作量(N-1-i)负载均衡划分行块
  2. 分块矩阵乘法计算上三角余弦相似度（fp32全精度），直方图统计正/负样本分布
     —— 用 2M-bin 直方图（bin宽1e-6）代替全量相似度保存，亿级样本对也只需 O(bins) 内存
  3. TPIR@FPIR 指标计算（1e-1 ~ 1e-6），结果与任务文档预期范围自动校验
  4. 支持提取 above/below 阈值的样本对（用于错误分析）：count 全量统计、
     明细带数量上限保护（超限截断），输出 CSV + NPZ
  5. 绘图：相似度分布图 + TPIR@FPIR 曲线（中文字体）
  6. 全程计时，输出各阶段耗时

用法示例（在已激活的 conda 环境中运行）：
  LD_LIBRARY_PATH=$CONDA_PREFIX/lib \
  python eval_v2_multi_gpu.py --gpus 0,1,2,3,4,5,6 \
      --extract-threshold 0.4 --extract-mode both
"""
import argparse
import json
import os
import pickle
import time

import numpy as np
import torch
import torch.multiprocessing as mp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = os.path.join(HERE, 'font', 'SourceHanSansSC-Normal.otf')
SIM_FP_SCALE = 1_000_000  # 提取样本对时相似度定点化倍率（精度1e-6）


def setup_chinese_font():
    font_manager.fontManager.addfont(FONT_PATH)
    prop = font_manager.FontProperties(fname=FONT_PATH)
    plt.rcParams['font.family'] = prop.get_name()
    plt.rcParams['axes.unicode_minus'] = False


# --------------------------------------------------------------------------
# 负载均衡划分：行 i 的有效工作量为 (N-1-i) 对，前缀和 W(r)=r*(N-1)-r*(r-1)/2
# 解 W(r) = k/G * total，使每张卡分到的样本对数量尽量相等
# --------------------------------------------------------------------------
def balanced_row_splits(N, G):
    total = N * (N - 1) // 2
    splits = [0]
    for k in range(1, G):
        target = total * k / G
        lo, hi = 0, N
        while lo < hi:
            mid = (lo + hi) // 2
            w = mid * (N - 1) - mid * (mid - 1) // 2
            if w < target:
                lo = mid + 1
            else:
                hi = mid
        splits.append(lo)
    splits.append(N)
    return splits


# --------------------------------------------------------------------------
# GPU worker：计算 [r0, r1) 行的上三角相似度直方图 + 可选样本对提取
# --------------------------------------------------------------------------
def worker(rank, gpu_id, feats_sh, ids_sh, r0, r1, nbins, extract_cfg, queue):
    try:
        _worker_impl(rank, gpu_id, feats_sh, ids_sh, r0, r1, nbins, extract_cfg, queue)
    except Exception:
        import traceback
        queue.put(dict(rank=rank, gpu=gpu_id, error=traceback.format_exc()))


def _worker_impl(rank, gpu_id, feats_sh, ids_sh, r0, r1, nbins, extract_cfg, queue):
    torch.backends.cuda.matmul.allow_tf32 = False  # fp32 全精度
    torch.set_num_threads(1)
    device = torch.device(f'cuda:{gpu_id}')
    N = feats_sh.shape[0]
    feats = feats_sh.to(device)                 # 全量特征驻留本卡 (N*512*4 B)
    ids_dev = ids_sh.to(device)

    hist_pos = torch.zeros(nbins, dtype=torch.long, device=device)
    hist_neg = torch.zeros(nbins, dtype=torch.long, device=device)
    scale = nbins / 2.0

    do_extract = extract_cfg is not None
    if do_extract:
        thr = extract_cfg['threshold']
        mode = extract_cfg['mode']
        # 四类样本对各自独立配额：错误分析最关心的 false_accept / false_reject
        # 不会被数量占优的 true_* 类别挤掉配额
        cap = extract_cfg['cap_per_cat']
        cats = [('above', True), ('above', False), ('below', True), ('below', False)]
        parts = {c: [] for c in cats}
        counts = {c: 0 for c in cats}
        keeps = {c: 0 for c in cats}

    total_rows = r1 - r0   # 块高在循环内按剩余列数自适应（每块 ~1e8 元素，峰值显存约3GB）
    t0 = time.time()
    last_report = t0

    start = r0
    while start < r1:
        # 对角块列跳过：行块 [start,end) 只需与列 [start+1, N) 计算，
        # 总计算量减半，且各卡元素数 == 各卡样本对数，负载均衡真正成立
        c0 = start + 1
        if c0 >= N:
            break
        # 块高按剩余列数自适应，保持每块 ~1e8 元素（尾部行块可以更高）
        chunk = max(16, min(8192, int(1e8 // max(N - start, 1))))
        end = min(start + chunk, r1)
        sim = feats[start:end] @ feats[c0:N].t()                 # (c, N-c0) fp32
        sim.clamp_(-1.0, 1.0)
        rows = torch.arange(start, end, device=device).unsqueeze(1)
        cols = torch.arange(c0, N, device=device).unsqueeze(0)
        valid = cols > rows                                       # 上三角 i<j
        same = ids_dev[start:end].unsqueeze(1) == ids_dev[c0:N].unsqueeze(0)
        pos_mask = valid & same
        neg_mask = valid & ~same

        # 先按掩码取相似度（数量约为块元素一半），再映射bin，省显存
        pos_idx = ((sim[pos_mask] + 1.0) * scale).long().clamp_(0, nbins - 1)
        neg_idx = ((sim[neg_mask] + 1.0) * scale).long().clamp_(0, nbins - 1)
        hist_pos += torch.bincount(pos_idx, minlength=nbins)
        hist_neg += torch.bincount(neg_idx, minlength=nbins)

        if do_extract:
            hi = sim > thr
            tags = ('above', 'below') if mode == 'both' else (mode,)
            for tag in tags:
                m_base = valid & (hi if tag == 'above' else ~hi)
                for is_pos in (True, False):
                    cat = (tag, is_pos)
                    m = m_base & (same if is_pos else ~same)
                    n_sel = int(m.sum())
                    counts[cat] += n_sel
                    if n_sel > 0 and keeps[cat] < cap:
                        ri, cj = m.nonzero(as_tuple=True)
                        take = min(len(ri), cap - keeps[cat])
                        ri, cj = ri[:take], cj[:take]
                        rec = torch.stack([
                            (start + ri), (c0 + cj),
                            same[ri, cj].long(),
                            (sim[ri, cj] * SIM_FP_SCALE).round().long(),
                        ], 1).cpu()
                        parts[cat].append(rec)
                        keeps[cat] += rec.shape[0]
                    del m
                del m_base
            del hi

        now = time.time()
        if now - last_report > 5 or end == r1:
            last_report = now
            print(f"  [GPU{gpu_id}] 进度 {end - r0}/{total_rows} 行 "
                  f"({(end - r0) / max(total_rows, 1) * 100:.0f}%)  {now - t0:.1f}s",
                  flush=True)
        del sim, valid, same, pos_mask, neg_mask, pos_idx, neg_idx
        start = end

    t_compute = time.time() - t0
    # 结果一律转成 numpy 再入队：torch张量经Queue传递会走fd共享存储还原路径，
    # worker先退出时主进程detach fd会EOFError；numpy按原始字节pickle，无此问题
    result = dict(
        rank=rank, gpu=gpu_id, rows=(r0, r1), t_compute=t_compute,
        hist_pos=hist_pos.cpu().numpy(), hist_neg=hist_neg.cpu().numpy(),
    )
    if do_extract:
        def _cat(tag):
            lst = parts[(tag, True)] + parts[(tag, False)]
            return (torch.cat(lst).numpy() if lst
                    else np.zeros((0, 4), dtype=np.int64))
        result.update(
            counts={f'{t}_{"pos" if p else "neg"}': counts[(t, p)] for (t, p) in cats},
            truncs={f'{t}_{"pos" if p else "neg"}': keeps[(t, p)] >= cap for (t, p) in cats},
            above=_cat('above'), below=_cat('below'),
        )
    queue.put(result)
    print(f"  [GPU{gpu_id}] 完成: 行[{r0},{r1}) 共{total_rows}行 耗时 {t_compute:.2f}s", flush=True)


# --------------------------------------------------------------------------
# 指标计算
# --------------------------------------------------------------------------
def tpir_at_fpir(hist_pos, hist_neg, target_fpirs):
    """直方图右端累积: cum[k] = #(sim 落在 bin>=k) ≈ #(sim >= 左缘 edges[k])
    FPIR(edges[k]) = cum_neg[k]/Nneg, TPIR(edges[k]) = cum_pos[k]/Npos
    """
    nbins = len(hist_pos)
    w = 2.0 / nbins
    edges = -1.0 + np.arange(nbins) * w
    cum_pos = np.cumsum(hist_pos[::-1])[::-1].astype(np.float64)
    cum_neg = np.cumsum(hist_neg[::-1])[::-1].astype(np.float64)
    n_pos, n_neg = float(hist_pos.sum()), float(hist_neg.sum())
    fpir = cum_neg / n_neg
    tpir = cum_pos / n_pos
    results = []
    for t in target_fpirs:
        k = min(int(np.searchsorted(-fpir, -t)), nbins - 1)  # fpir 单调不增
        results.append(dict(target_fpir=t, threshold=float(edges[k]),
                            fpir=float(fpir[k]), tpir=float(tpir[k])))
    return results, edges, fpir, tpir, n_pos, n_neg


# --------------------------------------------------------------------------
# 绘图
# --------------------------------------------------------------------------
def plot_results(hist_pos, hist_neg, curve, outdir, prefix='v2', ngpu=7):
    os.makedirs(outdir, exist_ok=True)
    nbins = len(hist_pos)
    w = 2.0 / nbins
    n_pos, n_neg = hist_pos.sum(), hist_neg.sum()
    nb_plot = 2000
    factor = nbins // nb_plot
    hp = hist_pos[:factor * nb_plot].reshape(nb_plot, factor).sum(1)
    hn = hist_neg[:factor * nb_plot].reshape(nb_plot, factor).sum(1)
    xc = -1.0 + (np.arange(nb_plot) + 0.5) * w * factor
    pdf_pos = hp / (n_pos * w * factor)
    pdf_neg = hn / (n_neg * w * factor)

    # 图1：相似度分布（log密度 + sim>=0放大）
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.plot(xc, pdf_neg, color='#d62728', lw=1.2, label=f'负样本对 ({int(n_neg):,})')
    ax.plot(xc, pdf_pos, color='#1f77b4', lw=1.2, label=f'正样本对 ({int(n_pos):,})')
    ax.set_yscale('log')
    ax.set_xlabel('余弦相似度')
    ax.set_ylabel('概率密度 (log)')
    ax.set_title('正负样本相似度分布（全范围）')
    ax.legend(); ax.grid(alpha=0.3)
    ax = axes[1]
    m = xc >= 0.0
    ax.plot(xc[m], pdf_neg[m], color='#d62728', lw=1.2, label='负样本对')
    ax.plot(xc[m], pdf_pos[m], color='#1f77b4', lw=1.2, label='正样本对')
    ax.set_xlabel('余弦相似度')
    ax.set_ylabel('概率密度')
    ax.set_title('正负样本相似度分布（sim≥0 放大）')
    ax.legend(); ax.grid(alpha=0.3)
    p1 = os.path.join(outdir, f'{prefix}_sim_distribution.png')
    fig.tight_layout(); fig.savefig(p1, dpi=150); plt.close(fig)

    # 图2：TPIR@FPIR 曲线
    fpir, tpir = curve['fpir'], curve['tpir']
    step = max(1, len(fpir) // 5000)
    fx, ty = fpir[::step], tpir[::step]
    keep = fx > 0
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(fx[keep], ty[keep], color='#2ca02c', lw=1.6, label='TPIR@FPIR')
    for r in curve['key_points']:
        if r['fpir'] > 0:
            ax.scatter([r['fpir']], [r['tpir']], zorder=5, s=45)
            ax.annotate(f"FPIR={r['target_fpir']:.0e}\nTPIR={r['tpir'] * 100:.2f}%\nthr={r['threshold']:.4f}",
                        (r['fpir'], r['tpir']), textcoords='offset points', xytext=(12, -8), fontsize=9)
    ax.set_xscale('log')
    ax.set_xlim(1e-6, 1.0); ax.set_ylim(0, 1.02)
    ax.set_xlabel('FPIR（假正例率，对数轴）')
    ax.set_ylabel('TPIR（真正例率）')
    ax.set_title(f'TPIR@FPIR 曲线（{ngpu}卡并行计算）')
    ax.grid(alpha=0.3, which='both'); ax.legend()
    p2 = os.path.join(outdir, f'{prefix}_tpir_fpir.png')
    fig.tight_layout(); fig.savefig(p2, dpi=150); plt.close(fig)
    return p1, p2


# --------------------------------------------------------------------------
# 样本对提取输出
# --------------------------------------------------------------------------
def pair_type_name(is_pos, is_above):
    if is_pos and is_above:
        return 'true_accept'
    if is_pos and not is_above:
        return 'false_reject'
    if not is_pos and is_above:
        return 'false_accept'
    return 'true_reject'


def save_extracted(recs, tag, thr, file_paths, outdir, prefix, max_csv_rows):
    """recs: (M,4) int64 [i, j, is_pos, sim*1e6]。保存 NPZ 全量 + CSV(限行数)。"""
    recs = np.asarray(recs)
    if recs.shape[0] == 0:
        return None
    i_arr = recs[:, 0]
    j_arr = recs[:, 1]
    is_pos = recs[:, 2].astype(bool)
    sim = recs[:, 3] / SIM_FP_SCALE

    npz_path = os.path.join(outdir, f'{prefix}_extract_{tag}.npz')
    np.savez_compressed(npz_path, i=i_arr, j=j_arr, is_pos=is_pos, sim=sim,
                        threshold=np.float64(thr))

    n_csv = min(len(i_arr), max_csv_rows)
    order = np.argsort(-sim[:n_csv]) if tag == 'above' else np.argsort(sim[:n_csv])
    csv_path = os.path.join(outdir, f'{prefix}_extract_{tag}.csv')
    with open(csv_path, 'w') as f:
        f.write('rank,i,j,is_pos,pair_type,similarity,path_i,path_j\n')
        for r, o in enumerate(order):
            ip, jp = int(i_arr[o]), int(j_arr[o])
            f.write(f"{r},{ip},{jp},{int(is_pos[o])},"
                    f"{pair_type_name(bool(is_pos[o]), tag == 'above')},"
                    f"{sim[o]:.6f},{file_paths[ip]},{file_paths[jp]}\n")
    return dict(npz=npz_path, csv=csv_path, rows_saved=int(n_csv),
                total_kept=int(len(i_arr)))


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='多卡并行人脸特征相似度评估 (TPIR@FPIR)')
    ap.add_argument('--pkl', default=os.path.join(HERE, 's4_0618_enhance.pkl'))
    ap.add_argument('--gpus', default='0,1,2,3,4,5,6', help='逗号分隔的GPU编号')
    ap.add_argument('--nbins', type=int, default=2_000_000, help='直方图bin数（覆盖[-1,1]）')
    ap.add_argument('--outdir', default=os.path.join(HERE, 'results'))
    ap.add_argument('--extract-threshold', type=float, default=None,
                    help='样本对提取阈值（如 0.4）；不指定则不提取')
    ap.add_argument('--extract-mode', choices=['above', 'below', 'both'], default='both')
    ap.add_argument('--max-extract', type=int, default=2_000_000,
                    help='每类样本对(true/false_accept/reject)的明细上限，全卡合计')
    ap.add_argument('--max-csv-rows', type=int, default=500_000, help='CSV输出行数上限')
    ap.add_argument('--mp-method', choices=['spawn', 'fork'], default='spawn',
                    help='fork可省去子进程重复import（父进程未初始化CUDA时安全），更快')
    args = ap.parse_args()

    setup_chinese_font()
    timings = {}
    t_all = time.time()

    gpu_ids = [int(g) for g in args.gpus.split(',')]
    G = len(gpu_ids)
    os.makedirs(args.outdir, exist_ok=True)

    # 1) 读取数据（主进程读一次，共享内存传给各worker）
    t0 = time.time()
    with open(args.pkl, 'rb') as f:
        feats_np, _flip, ids_np, file_paths = pickle.load(f)
    feats = torch.from_numpy(np.ascontiguousarray(feats_np, dtype=np.float32)).share_memory_()
    ids = torch.from_numpy(np.ascontiguousarray(ids_np)).share_memory_()
    del feats_np, ids_np
    N = feats.shape[0]
    timings['load'] = time.time() - t0
    print(f"[加载] pkl 读取+共享内存 {timings['load']:.2f}s  N={N}  dim={feats.shape[1]}")

    uniq, counts = np.unique(ids.numpy(), return_counts=True)
    n_ids = len(uniq)
    total_pairs = N * (N - 1) // 2
    expect_pos = int((counts * (counts - 1) // 2).sum())
    print(f"[统计] 身份数 {n_ids}, 总样本对 {total_pairs:,}, "
          f"理论正样本对 {expect_pos:,}, 理论负样本对 {total_pairs - expect_pos:,}")

    # 2) 负载均衡划分 + 启动多卡worker
    splits = balanced_row_splits(N, G)

    def prefix_work(r):  # 前 r 行的有效样本对数
        return r * (N - 1) - r * (r - 1) // 2

    works = [prefix_work(splits[k + 1]) - prefix_work(splits[k]) for k in range(G)]
    print(f"[划分] 行块: {list(zip(splits[:-1], splits[1:]))}")
    print(f"[划分] 各卡样本对数: {[f'{w:,}' for w in works]}  "
          f"(最大/平均={max(works) / (sum(works) / G):.3f})")

    extract_cfg = None
    if args.extract_threshold is not None:
        extract_cfg = dict(threshold=args.extract_threshold, mode=args.extract_mode,
                           cap_per_cat=max(1, args.max_extract // G))
        print(f"[提取] 阈值={args.extract_threshold}, 模式={args.extract_mode}, "
              f"每类样本对每卡明细上限={extract_cfg['cap_per_cat']:,}")

    t0 = time.time()
    ctx = mp.get_context(args.mp_method)
    queue = ctx.Queue()
    procs = []
    for rank, gid in enumerate(gpu_ids):
        p = ctx.Process(target=worker,
                        args=(rank, gid, feats, ids, splits[rank], splits[rank + 1],
                              args.nbins, extract_cfg, queue))
        p.start()
        procs.append(p)

    results = [queue.get() for _ in range(G)]   # 先取完结果再join，避免管道死锁
    for p in procs:
        p.join()
    for r in results:
        if 'error' in r:
            print(f"[错误] GPU{r['gpu']} worker 失败:\n{r['error']}")
            raise SystemExit(1)
    timings['compute'] = time.time() - t0
    wtime = sum(r['t_compute'] for r in results) / G
    print(f"[计算] 多卡并行总墙钟 {timings['compute']:.2f}s（单卡平均 {wtime:.2f}s）")

    # 3) 合并直方图
    t0 = time.time()
    results.sort(key=lambda r: r['rank'])
    hist_pos = np.zeros(args.nbins, dtype=np.int64)
    hist_neg = np.zeros(args.nbins, dtype=np.int64)
    for r in results:
        hist_pos += r['hist_pos']
        hist_neg += r['hist_neg']
    timings['merge'] = time.time() - t0
    n_pos, n_neg = int(hist_pos.sum()), int(hist_neg.sum())
    print(f"[合并] {timings['merge']:.2f}s  正样本对 {n_pos:,}（理论 {expect_pos:,}，"
          f"{'一致' if n_pos == expect_pos else '不一致!'}），负样本对 {n_neg:,}")

    # 保存直方图（后续可直接复用，无需重算）
    np.save(os.path.join(args.outdir, 'v2_hist_pos.npy'), hist_pos)
    np.save(os.path.join(args.outdir, 'v2_hist_neg.npy'), hist_neg)

    # 4) TPIR@FPIR 指标
    t0 = time.time()
    targets = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    key_points, edges, fpir, tpir, _, _ = tpir_at_fpir(hist_pos, hist_neg, targets)
    timings['metrics'] = time.time() - t0
    print(f"[指标] 计算耗时 {timings['metrics']:.2f}s")
    print("       TPIR@FPIR 结果:")
    expected_range = {1e-2: (95, 97), 1e-3: (90, 93), 1e-4: (82, 85), 1e-5: (60, 65)}
    for r in key_points:
        t = r['target_fpir']
        mark = ''
        if t in expected_range:
            lo, hi = expected_range[t]
            pct = r['tpir'] * 100
            mark = '  [在预期范围内 ✓]' if lo - 5 <= pct <= hi + 5 else '  [偏离预期! 请检查]'
        print(f"  FPIR<={t:.0e}: TPIR={r['tpir'] * 100:.3f}%  "
              f"(实际FPIR={r['fpir']:.3e}, 阈值={r['threshold']:.4f}){mark}")

    # 5) 样本对提取输出
    extract_info = None
    if extract_cfg is not None:
        t0 = time.time()
        above = np.concatenate([r['above'] for r in results])
        below = np.concatenate([r['below'] for r in results])
        agg_counts, agg_truncs = {}, {}
        for r in results:
            for k, v in r['counts'].items():
                agg_counts[k] = agg_counts.get(k, 0) + v
            for k, v in r['truncs'].items():
                agg_truncs[k] = agg_truncs.get(k, False) or v
        thr = args.extract_threshold
        named = [('true_accept  (正样本, sim> thr)', agg_counts.get('above_pos', 0),
                  len(above[above[:, 2] == 1]) if len(above) else 0, agg_truncs.get('above_pos', False)),
                 ('false_accept (负样本, sim> thr)', agg_counts.get('above_neg', 0),
                  len(above[above[:, 2] == 0]) if len(above) else 0, agg_truncs.get('above_neg', False)),
                 ('false_reject (正样本, sim<=thr)', agg_counts.get('below_pos', 0),
                  len(below[below[:, 2] == 1]) if len(below) else 0, agg_truncs.get('below_pos', False)),
                 ('true_reject  (负样本, sim<=thr)', agg_counts.get('below_neg', 0),
                  len(below[below[:, 2] == 0]) if len(below) else 0, agg_truncs.get('below_neg', False))]
        print(f"[提取] 阈值={thr} 四类样本对统计（count 全量 / 明细保存 / 是否截断）:")
        for name, cnt, kept, trunc in named:
            print(f"  {name}: {cnt:>16,} / {kept:>10,} / {'是' if trunc else '否'}")
        info = dict(threshold=thr, mode=args.extract_mode,
                    counts={k: int(v) for k, v in agg_counts.items()},
                    truncated={k: bool(v) for k, v in agg_truncs.items()})
        info['above'] = save_extracted(above, 'above', thr,
                                       file_paths, args.outdir, 'v2', args.max_csv_rows)
        info['below'] = save_extracted(below, 'below', thr,
                                       file_paths, args.outdir, 'v2', args.max_csv_rows)
        for tag in ('above', 'below'):
            if info[tag]:
                print(f"  [{tag}] 明细 → {info[tag]['npz']} + {info[tag]['csv']}"
                      f"（CSV {info[tag]['rows_saved']:,} 行）")
        extract_info = info
        timings['extract'] = time.time() - t0

    # 6) 绘图
    t0 = time.time()
    p1, p2 = plot_results(hist_pos, hist_neg,
                          dict(edges=edges, fpir=fpir, tpir=tpir, key_points=key_points),
                          args.outdir, prefix='v2', ngpu=G)
    timings['plot'] = time.time() - t0
    print(f"[绘图] {timings['plot']:.2f}s → {p1}\n              → {p2}")

    # 7) 汇总
    timings['total'] = time.time() - t_all
    summary = dict(N=int(N), n_ids=int(n_ids), total_pairs=int(total_pairs),
                   n_pos=n_pos, n_neg=n_neg, gpus=gpu_ids, nbins=args.nbins,
                   key_points=key_points, timings=timings, extract=extract_info)
    with open(os.path.join(args.outdir, 'v2_summary.json'), 'w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print("\n========== 耗时汇总 ==========")
    for k, v in timings.items():
        print(f"  {k:>10s}: {v:8.2f} s")
    print(f"吞吐: {total_pairs / timings['compute'] / 1e8:.2f} 亿样本对/秒（仅计算阶段）")


if __name__ == '__main__':
    main()
