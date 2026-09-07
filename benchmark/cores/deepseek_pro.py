#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deepseek-pro core —— 提炼自 deepseek-pro/step4_eval_optimized.py（v5，同目录 任务日志.md）

原实现要点（step4 文件头注释 / gpu_hist_worker / gpu_pos_worker / assign_pairs_to_gpus）:
    * 只算上三角，大块 B=16384；块对 (bi,bj) 按元素数降序、对级贪心分给"最闲"卡
      （静态负载均衡，实测各卡负载差 <0.01%）
    * fp32 GEMM（cuBLAS）；相似度量化链 + CUDA bincount（torch 2.12 支持 int32）：
        q = round(v*scale).clamp(-scale, scale) -> int32 + scale
    * 热路径不做逐元素身份比对：全量上三角直方图 pass 不算 id，
      负样本直方图 = 全体直方图 - 正样本直方图
    * 正样本原在独立 pass 用 GPU padded-bmm（按身份分组补零一次算完）
    * 流式按块异步传输（多 CUDA stream 与计算重叠）+ worker 预热

统一网格适配（200,000 bins 覆盖 [-1,1)，宽 1e-5）:
    scale = BINS/2 = 100000 => 量化槽 0..200000（共 200001 槽）。原 40001 槽语义
    （round 到最近中心 + 两侧 clamp）保留，仅分辨率提至 1e-5；末槽（值 >= 1-5e-6
    或越界被 clamp 到 1.0 的样本对）折叠进 bin 199999，保证每对恰统计一次、不丢计数。

2M（200 万样本）内存安全改造（线程版原实现 2M 曾系统内存暴涨 + CUDA OOM）:
    * 统一框架用 spawn 每 GPU 一进程 + torch 共享内存张量只读（子进程不再各自
      复制全量 4GB 特征与大中间张量；物理内存一份，按块流式 H2D，无整卡突发拷贝）
    * 块内量化链原地（mul_/round_/clamp_ 复用 C 存储），不物化大 fp32 中间张量
    * 正样本不做整矩阵 bmm：父进程先静态判定"哪些 tile 含同身份对"（排序分组 +
      块内计数），worker 仅对这些 tile 做等值掩码、从同一 C 中取子集计数 ——
      与全量直方图同一批值、同一量化链 => pos 逐 bin 是 full 的子集，
      neg = full - pos 逐 bin 非负（bmm 重算在 fp32 末位噪声下可能产生负 bin），
      热路径 ~99.9% 的 tile 保持免身份比对（正样本对仅 ~千万级 / 总对数 ~2e12）
    * 显存占用：2M 时每卡 ≈ 4GB 特征缓存 + 1GB C + 1GB int32 q + 掩码瞬态 < 20GB

已去掉数据读取/样本对提取/绘图/argparse；仅 numpy/torch/torch.multiprocessing。
"""
import os
import time

import numpy as np
import torch
import torch.multiprocessing as mp

from .. import common

MODEL_NAME = 'deepseek-pro'
MODEL_DESC = ('DeepSeek-Pro v5 方法：spawn每GPU一进程 + 对级贪心静态负载均衡 + 大块fp32 GEMM '
              '+ int32量化bincount(200001槽折叠200000) + 负=全体-正(同C子集提取,逐bin非负)')
ORIGIN = ('deepseek-pro/step4_eval_optimized.py v5 (gpu_hist_worker/gpu_pos_worker/'
          'assign_pairs_to_gpus；bins 40001 -> 统一 200000；共享内存按块流式改造防 2M OOM)')

BLOCK = 16384                  # 原 v5 默认块大小
SCALE = common.BINS // 2       # 100000：统一 bin 宽 1e-5 的倒数（原 bin_scale=20000/5e-5）
NBINS = 2 * SCALE + 1          # 200001 个量化槽（末槽折叠进 bin 199999）
FOLD = 2 * SCALE               # 需要折叠的槽号（=200000）


def _block_rows(N, B):
    """每行块的 (r0, r1)（原 block_rows）"""
    return [(s, min(s + B, N)) for s in range(0, N, B)]


def _pos_tile_flags(ids, rows):
    """静态判定哪些上三角块对含 >=1 条同身份样本对。

    对角块 (b,b)：块内同身份行数 >= 2（逐块 np.unique 计数）。
    非对角：某身份任意两个成员所在行块为 (b1,b2), b1<b2，则 tile (b1,b2) 必含
    该身份跨块样本对（成员行号 < 列块行号 => 全局 i<j），故把该身份成员出现过的
    行块区间 [bf,bl] 内全部 (b1<b2) 块对标记上 —— 可能多标（成员并未落在中间
    块时该 tile 等值掩码为空，多一次无用比对），但绝不漏标。
    返回 (flags, pos_theory)：flags 为块对 (bi,bj) 集合，pos_theory 为正样本对理论数。
    """
    N = len(ids)
    flags = set()
    if N < 2:
        return flags, 0
    B = rows[0][1] - rows[0][0]
    # 对角块：块内同身份计数
    for i, (b0, b1) in enumerate(rows):
        _, cnt = np.unique(ids[b0:b1], return_counts=True)
        if cnt.size and cnt.max() >= 2:
            flags.add((i, i))
    # 非对角块：按身份分组，组跨度 [bf, bl] 内全部 (b1<b2) 块对
    order = np.argsort(ids, kind='stable')
    sid = ids[order]
    starts = np.r_[0, np.flatnonzero(sid[1:] != sid[:-1]) + 1, N]
    counts = np.diff(starts)
    pos_theory = int((counts.astype(np.int64) * (counts - 1) // 2).sum())
    for s, e in zip(starts[:-1], starts[1:]):
        if e - s < 2:
            continue
        bf = int(order[s]) // B
        bl = int(order[e - 1]) // B
        if bl > bf:
            for b1 in range(bf, bl):
                for b2 in range(b1 + 1, bl + 1):
                    flags.add((b1, b2))
    return flags, pos_theory


def _quant(v, scale):
    """step4 量化链（v 原地改写为整数值 fp32）：q = int32(round(clamp(v*scale))) + scale。
    v*scale 的 fp32 中间量在 round 后为整数（|q|<1e5 可精确表示），转 int32 无损。"""
    v.mul_(scale).round_().clamp_(-scale, scale)
    return v.to(torch.int32).add_(scale)


def _fold_slot(h):
    """把 200001 槽的 GPU int64 直方图折叠成 200000 bins 的 np.int64：
    末槽（值 >= 1-5e-6 / clamp 到 1.0 的对）并入 bin 199999（覆盖 [1-1e-5, 1]）。"""
    h[FOLD - 1] += h[FOLD]
    return h[:FOLD].cpu().numpy()


def _hist_worker(cfg, feats_shm, ids_shm, res_q):
    """单卡直方图 worker（spawn）：按父进程贪心分片的 tile 列表，
    一次扫描得到全体上三角直方图；含同身份对的 tile 顺带产出正样本直方图。"""
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = f'cuda:{cfg["gpu"]}'
    torch.cuda.set_device(dev)
    scale, nbins = cfg['scale'], cfg['nbins']
    tiles = cfg['tiles']
    F, I = feats_shm, ids_shm          # 只读共享内存（CPU 侧，不复制）
    ids_g = I.to(dev)                  # (N,) int64（正样本等值掩码用）

    # 本卡需要的行块（行+列两侧）全部异步提交到独立传输流，计算流按块 event 等待
    need = sorted({(r0, r1) for (r0, r1, c0, c1, _f) in tiles}
                  | {(c0, c1) for (r0, r1, c0, c1, _f) in tiles})
    cache, ev = {}, {}
    if need:
        upl = torch.cuda.Stream(device=dev)
        with torch.cuda.stream(upl):
            for (b0, b1) in need:
                cache[(b0, b1)] = F[b0:b1].to(dev, non_blocking=True)
                e = torch.cuda.Event()
                upl.record_event(e)
                ev[(b0, b1)] = e
    comp = torch.cuda.Stream(device=dev)
    with torch.cuda.stream(comp):
        # worker 预热：首个块上跑一次 GEMM+量化+bincount（原实现预热，不计入计时）
        if tiles:
            a0, a1, c0, c1, _f = tiles[0]
            comp.wait_event(ev[(a0, a1)])
            comp.wait_event(ev[(c0, c1)])
            w0 = cache[(a0, a1)][:1024] @ cache[(c0, c1)][:1024].t()
            torch.bincount(_quant(w0, scale).reshape(-1), minlength=nbins)

        hist_all = torch.zeros(nbins, dtype=torch.int64, device=dev)
        hist_pos = torch.zeros(nbins, dtype=torch.int64, device=dev)
        t_mm = t_hist = 0.0
        n = len(tiles)
        pe = max(1, n // 20)
        for k, (r0, r1, c0, c1, has_pos) in enumerate(tiles):
            comp.wait_event(ev[(r0, r1)])
            comp.wait_event(ev[(c0, c1)])
            A = cache[(r0, r1)]
            B = cache[(c0, c1)]
            is_diag = r0 == c0
            nr, nc = r1 - r0, c1 - c0

            t0 = time.perf_counter()
            C = A @ B.t()                              # (nr,nc) fp32
            comp.synchronize()                         # 只等计算流（传输流独立继续）
            t_mm += time.perf_counter() - t0

            t0 = time.perf_counter()
            m = None
            if has_pos:
                # 正样本：仅对含同身份对的 tile 做等值掩码，从同一 C 取子集
                # （同一批值 + 同一量化链 => 逐 bin 是全体直方图的子集）
                if is_diag:
                    m = torch.ones(nr, nc, dtype=torch.bool, device=dev).triu_(1)
                    pm = (ids_g[r0:r1, None] == ids_g[None, c0:c1]) & m
                else:
                    pm = ids_g[r0:r1, None] == ids_g[None, c0:c1]
                pv = C[pm]
                if pv.numel() > 0:
                    hist_pos += torch.bincount(_quant(pv, scale), minlength=nbins)
            if is_diag:
                # 对角块：严格上三角掩码后计数（每对 i<j 恰一次）
                if m is None:
                    m = torch.ones(nr, nc, dtype=torch.bool, device=dev).triu_(1)
                v = C[m]
                if v.numel() > 0:
                    hist_all += torch.bincount(_quant(v, scale), minlength=nbins)
            else:
                # 非对角块：行块全部索引 < 列块全部索引，块内每个元素一条合法样本对
                hist_all += torch.bincount(_quant(C.reshape(-1), scale), minlength=nbins)
            comp.synchronize()                         # 只等计算流（传输流独立继续）
            t_hist += time.perf_counter() - t0

            if (k + 1) % pe == 0 or k + 1 == n:
                print(f"  [gpu{cfg['gpu']}] tile {k + 1}/{n} "
                      f"matmul={t_mm / max(k + 1, 1) * 1000:.0f}ms "
                      f"hist={t_hist / max(k + 1, 1) * 1000:.0f}ms", flush=True)

    torch.cuda.synchronize(dev)
    res_q.put((_fold_slot(hist_all), _fold_slot(hist_pos),
               t_mm, t_hist, len(tiles)))
    print(f"  [gpu{cfg['gpu']}] 完成 {len(tiles)} tiles | matmul {t_mm:.1f}s "
          f"hist {t_hist:.1f}s", flush=True)


def compute(feats, ids, gpus, workdir):
    feats = np.ascontiguousarray(feats, dtype=np.float32)
    ids = np.ascontiguousarray(ids, dtype=np.int64)
    N = len(ids)
    os.makedirs(workdir, exist_ok=True)
    if not gpus:
        raise ValueError('gpus 列表为空')

    # ---- 块对规划 + 含正样本 tile 判定 + 对级贪心负载均衡（父进程，numpy）----
    rows = _block_rows(N, BLOCK)
    nb = len(rows)
    pairs = []                                   # (bi, bj, r0, r1, c0, c1, weight)
    for i in range(nb):
        r0, r1 = rows[i]
        for j in range(i, nb):
            c0, c1 = rows[j]
            if i == j:
                w = (r1 - r0) * (r1 - r0 + 1) // 2
            else:
                w = (r1 - r0) * (c1 - c0)
            pairs.append((i, j, r0, r1, c0, c1, w))
    flags, pos_theory = _pos_tile_flags(ids, rows)

    order = sorted(range(len(pairs)), key=lambda k: -pairs[k][6])
    loads = [0] * len(gpus)
    assign = [[] for _ in gpus]
    for k in order:
        i, j, r0, r1, c0, c1, w = pairs[k]
        g = int(np.argmin(loads))
        loads[g] += w
        assign[g].append((r0, r1, c0, c1, (i, j) in flags))
    total_pairs = N * (N - 1) // 2
    print(f'[deepseek-pro] N={N:,} 行块={nb} 上三角tile={len(pairs)} '
          f'含正样本tile={len(flags)} 理论正样本对={pos_theory:,} '
          f'负载={[l // 10 ** 6 for l in loads]}M元素', flush=True)

    # ---- spawn 前在主进程建立共享内存张量（不触碰 CUDA；2M 时物理内存仅一份）----
    feats_shm = torch.from_numpy(feats).share_memory_()
    ids_shm = torch.from_numpy(ids).share_memory_()

    ctx = mp.get_context('spawn')
    res_q = ctx.Queue()
    procs = []
    for g, gpu in enumerate(gpus):
        cfg = {'gpu': gpu, 'scale': SCALE, 'nbins': NBINS, 'tiles': assign[g]}
        p = ctx.Process(target=_hist_worker, args=(cfg, feats_shm, ids_shm, res_q))
        p.start()
        procs.append(p)

    t0 = time.perf_counter()
    # 先读后 join：每份结果 ~3.3MB，若先 join 等子进程退出，其 QueueFeederThread
    # 会因管道写满阻塞，而子进程退出 finalizer 又要 join 该线程 => 经典死锁。
    results = []
    while len(results) < len(procs):
        if not res_q.empty():
            results.append(res_q.get())
            continue
        if not any(p.is_alive() for p in procs):      # 无存活 worker 且无新结果 => 异常
            break
        time.sleep(0.05)
    for p in procs:
        p.join()
    core_s = time.perf_counter() - t0
    if len(results) != len(procs):
        raise RuntimeError(f'GPU worker 结果缺失 {len(results)}/{len(procs)} '
                           f'exitcode={[p.exitcode for p in procs]}')
    bad = [p.exitcode for p in procs if p.exitcode != 0]
    if bad:
        raise RuntimeError(f'GPU worker 异常退出 exitcode={bad}')

    hist_all = None
    hist_pos = None
    t_mm = t_hist = 0.0
    n_tiles = 0
    for ha, hp, mm, hs, nt in results:
        hist_all = ha if hist_all is None else hist_all + ha
        hist_pos = hp if hist_pos is None else hist_pos + hp
        t_mm += mm
        t_hist += hs
        n_tiles += nt
    hist_all = np.asarray(hist_all, dtype=np.int64)
    hist_pos = np.asarray(hist_pos, dtype=np.int64)
    assert hist_all.shape == (common.BINS,) and hist_pos.shape == (common.BINS,), \
        f'bins={hist_all.shape} != {common.BINS}'

    neg_hist = hist_all - hist_pos
    assert int(neg_hist.min()) >= 0, '负样本直方图出现负计数（正样本非全体子集？）'
    got_all, got_pos = int(hist_all.sum()), int(hist_pos.sum())
    assert got_all == total_pairs, f'全体直方图 {got_all:,} != 理论 {total_pairs:,}'
    assert got_pos == pos_theory, f'正样本直方图 {got_pos:,} != 理论 {pos_theory:,}'

    meta = {
        'model': MODEL_NAME, 'desc': MODEL_DESC, 'origin': ORIGIN,
        'parallel': ('spawn 每GPU一进程 + 静态对级贪心负载均衡 + 流式按块异步传输(CUDA stream)'
                     ' + 仅含正样本tile做等值掩码(热路径免身份比对)'),
        'block': BLOCK,
        'precision': 'fp32 (allow_tf32=False) + int32量化 round/clamp + CUDA bincount',
        'native_bins': 40_001,           # 原 40001 槽 / 5e-5 分辨率
        'core_s': round(core_s, 3),
        'matmul_s': round(t_mm, 3),
        'hist_s': round(t_hist, 3),
        'tiles_done': n_tiles,
    }
    return hist_pos, neg_hist, meta
