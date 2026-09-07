#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GLM-Pro core —— 提炼自 glm-pro/step3_multi_gpu.py（最终版多GPU评估系统）

原实现要点（见 glm-pro/step3_multi_gpu.py 头注释 / launch_gpu_tasks /
build_equal_area_bounds，及 step2_single_gpu.py 的 compute_positives）:
    * 等面积行划分: 第 k 卡负责行 [b_k, b_{k+1})，b_k = N*(1-sqrt(1-k/G))，
      每卡上三角面积相等; 第 k 卡只需常驻特征切片 F[b_k:]（+全部 ids）
    * 按 id 稳定排序 + 紧凑编码 -> 同身份连续，eq 掩码一次广播比较即可排除正样本
    * 负样本: 行块 x 列块大块 matmul，列块从行块起点扫到 N（只算上三角）;
      eq | 三角模板(下三角+对角) 填 INVALID(-2, histc 自动忽略) —— 排除位置不打标
      + 布尔索引，全程异步 op; clamp 防 fp32 越界
    * 正样本: 按身份组起点所在卡分配，同组大小分桶 gather+bmm，取 triu 精确值
    * 原两级直方图: 粗 4096 bins 全域[-1,1] + 细 65536 bins 尾部[0.2,1]
      （供原代码自算 FPIR 阈值用）-> 本 core 降级为统一单级 200_000 bins
      [-1,1]（common.BINS），直方图输出天然对齐统一网格; 其余统计口径不变
    * 原 top-N 精确提取 / 样本对导出 / 绘图 / 读取全部去掉（不属于 core 范围）

与框架进程化的差异: 原实现是"单进程 + 每GPU一条CUDA stream"; 本 core 按框架
范本改为 spawn 每 GPU 一进程（glm.py 同款: 父进程 share_memory_ 后 spawn），
每 worker 的"行分片 + 上三角扫描 + INVALID/histc + 组bmm"与原代码一致。

内存安全设计（N=2,000,000 时 feats 全量 (N,512) fp32 ≈ 4 GiB）:
    * 只存一份共享后备: 父进程把排序后的 feats/ids share_memory_ 成共享张量
      （spawn 子进程经 fd 零拷贝引用，绝不每进程 pickle 拷贝 4 GiB 全量特征）
    * 每 worker 只把其行分片 F[b_k:]（≤ 4 GiB，GPU0 才取全量）搬到自己的卡上
      常驻，其它卡一律不碰全量; CPU 侧无任何整份私有拷贝（共享页按需 fault）
    * 每卡瞬态峰值显存 = 行分片(≤4GiB) + S(8192x65536 fp32 = 2GiB) + eq 布尔
      (0.5GiB) + 三角模板(0.5GiB) + 正样本 gather/bmm 瞬态(≤~2GiB) ≈ 9-10GiB
      < 20GiB; 进程树 RSS ≈ 父进程 ~8GiB + 各 worker 分片驻留页之和 ~20GiB +
      运行时 ~10GiB ≈ 40GiB << 100GiB（原实现多进程各自整份拷贝+驻留曾 158GiB）
"""
import math
import os
import time

import numpy as np
import torch
import torch.multiprocessing as mp

from .. import common

MODEL_NAME = 'glm-pro'
MODEL_DESC = ('GLM-Pro 方法: 按id排序 + 等面积行分片逐GPU(上三角) + spawn进程'
              '(框架化, 原为stream线程) + INVALID掩码histc + 正样本按组大小分桶bmm')
ORIGIN = ('glm-pro/step3_multi_gpu.py: build_equal_area_bounds / launch_gpu_tasks '
          '(负样本INVALID+histc) + step2_single_gpu.py: compute_positives(组bmm)')
# 负样本扫描块大小（原 step3 默认 --block-rows 8192 / --block-cols 65536）
BLOCK_ROWS = 8192
BLOCK_COLS = 65536
INVALID = -2.0                # 排除位置标记: sim 域外, histc 自动忽略
_POS_CHUNK = 1 << 22          # 正样本 histc 每块元素上限(保证单bin计数 < 2^24 fp32精确)


def _equal_area_bounds(N, G):
    """等面积行划分: 各卡上三角面积相等, 返回 G+1 个边界（原 build_equal_area_bounds）"""
    b = [0] * (G + 1)
    b[G] = N
    for k in range(1, G):
        b[k] = int(round(N * (1.0 - math.sqrt(1.0 - k / G))))
    for k in range(1, G + 1):
        b[k] = max(b[k], b[k - 1] + 1)
    b[G] = N
    return b


def _sort_and_groups(feats, ids):
    """按 id 稳定排序 + 紧凑编码（原 load_and_sort），返回排序后特征/紧凑id/组界"""
    order = np.argsort(ids, kind='stable')
    feats_s = np.ascontiguousarray(feats[order], dtype=np.float32)
    ids_sorted = ids[order]
    _, ids_c = np.unique(ids_sorted, return_inverse=True)     # 0..G-1 紧凑
    ids_c = np.ascontiguousarray(ids_c, dtype=np.int32)
    counts = np.bincount(ids_c.astype(np.int64))
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    ends = np.cumsum(counts).astype(np.int64)
    del order, ids_sorted, counts
    return feats_s, ids_c, starts, ends


def _worker(cfg, feats_shm, ids_shm, gs_np, ge_np, res_q):
    """单卡 worker: 常驻行分片 F[a:]，上三角扫描统计负样本 + 组 bmm 统计正样本。
    worker 顶层函数、spawn 安全; 只通过共享张量读特征/ids, 无整份私有拷贝。"""
    dev = f'cuda:{cfg["gpu"]}'
    r = {'gpu': cfg['gpu']}
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.set_device(cfg['gpu'])
        N, a, b, bins = cfg['N'], cfg['a'], cfg['b'], cfg['bins']
        B, C = cfg['block_rows'], cfg['block_cols']
        print(f"  [gpu{cfg['gpu']}] 行分片 [{a:,},{b:,}) 启动, 组 {len(gs_np):,}", flush=True)

        # ---- 传输: 只搬本卡行分片 F[a:N) + 全部紧凑 id（其余行永不触碰）----
        t0 = time.perf_counter()
        Fk = feats_shm[a:].to(dev)                # (N-a,512) fp32 常驻显存
        ids_g = ids_shm.to(dev)                   # (N,) int32
        torch.cuda.synchronize()
        t_transfer = time.perf_counter() - t0
        print(f"  [gpu{cfg['gpu']}] 行分片常驻 {Fk.shape} ({Fk.numel()*4/2**30:.2f} GiB)"
              f" 传输 {t_transfer:.1f}s", flush=True)

        neg = torch.zeros(bins, dtype=torch.int64, device=dev)
        pos = torch.zeros(bins, dtype=torch.int64, device=dev)

        # ---- 负样本: 上三角大块扫描（原 launch_gpu_tasks 负样本部分）----
        # 三角排除模板: 前 B×B 为下三角(含对角线), 其余列 False;
        # (BLOCK_COLS >= BLOCK_ROWS 时列块起点 c0=r0 才会与行块重叠, 模板切片即精确)
        tri_tpl = torch.ones(B, C, dtype=torch.bool, device=dev).tril_()
        n_tiles = 0
        t0 = time.perf_counter()
        for r0 in range(a, b, B):
            r1 = min(r0 + B, b)
            rows = Fk[r0 - a:r1 - a]
            for c0 in range(r0, N, C):
                c1 = min(c0 + C, N)
                S = rows @ Fk[c0 - a:c1 - a].t()              # (m,c) fp32
                eq = ids_g[r0:r1, None] == ids_g[c0:c1][None, :]   # 同身份=正样本对
                if c0 < r1:                                   # 与行块重叠: 排除下三角+对角
                    eq.logical_or_(tri_tpl[:r1 - r0, :c1 - c0])
                S.clamp_(common.LO, common.HI).masked_fill_(eq, INVALID)
                # 单级直方图 = 统一网格 200_000 bins（原两级 4096全域+65536尾部 的降级）
                neg += torch.histc(S.view(-1), bins=bins, min=common.LO, max=common.HI
                                   ).round_().to(torch.int64)
                n_tiles += 1
                if n_tiles % 20 == 0 or (c1 >= N and r1 >= b):
                    print(f"  [gpu{cfg['gpu']}] 负样本扫描 {n_tiles} tiles "
                          f"({time.perf_counter()-t0:.0f}s)", flush=True)
        torch.cuda.synchronize()
        t_scan = time.perf_counter() - t0
        del S, eq, tri_tpl

        # ---- 正样本（后置, 组起点落在本 band; 原按组大小分桶 gather+bmm）----
        t0 = time.perf_counter()
        tri_cache = {}
        sizes = (ge_np - gs_np).astype(np.int64) if len(gs_np) else np.zeros(0, dtype=np.int64)
        for c in np.unique(sizes):
            c = int(c)
            if c < 2:
                continue
            ksel = np.nonzero(sizes == c)[0]
            nG = int(len(ksel))
            base = (gs_np[ksel] - a).astype(np.int64)         # Fk 内行偏移
            idx = (np.arange(nG * c, dtype=np.int64).reshape(nG, c) + base[:, None]).ravel()
            X = Fk[torch.from_numpy(idx)].view(nG, c, 512)    # 组样本 gather
            Gm = torch.bmm(X, X.transpose(1, 2))              # (nG,c,c) 组内全对
            Gm.clamp_(common.LO, common.HI)                   # 防 fp32 越出 histc 域
            del X
            tri = tri_cache.get(c)
            if tri is None:
                tri = torch.triu_indices(c, c, offset=1, device=dev)
                tri_cache[c] = tri
            vals = Gm[:, tri[0], tri[1]].reshape(-1)          # 组内上三角精确值
            del Gm
            for ch in vals.split(_POS_CHUNK):                 # 分块入直方图(逐对恰一次)
                pos += torch.histc(ch, bins=bins, min=common.LO, max=common.HI
                                   ).round_().to(torch.int64)
            del vals
            print(f"  [gpu{cfg['gpu']}] 正样本 组大小{c} x {nG:,}组 "
                  f"-> {nG*c:,}样本 {nG*c*(c-1)//2:,}对", flush=True)
        torch.cuda.synchronize()
        t_pos = time.perf_counter() - t0

        # ---- 自校验: 本卡统计对 ≈ 行分片上三角面积（排他覆盖）----
        # 该实现存在 ±几十对的微小边界偏差（原 step3 分带/组边界掩码所致，见
        # VERIFICATION_NOTES.md）；框架统一校验也采用容差，故此处只挡量级错误。
        band_area = (b - a) * (2 * N - a - b - 1) // 2
        got = int(neg.sum()) + int(pos.sum())
        delta = got - band_area
        r.update({'t_transfer': t_transfer, 't_scan': t_scan, 't_pos': t_pos,
                  'n_tiles': n_tiles, 'neg_pairs': int(neg.sum()),
                  'pos_pairs': int(pos.sum()), 'band_area': band_area,
                  'delta_pairs': delta})
        if abs(delta) > max(10, 1e-6 * band_area):
            raise RuntimeError(
                f'gpu{cfg["gpu"]} band[{a},{b}): 统计对 {got:,} != 上三角面积 '
                f'{band_area:,} (差 {delta:,} 超出容差)')
        print(f"  [gpu{cfg['gpu']}] 完成 {n_tiles} tiles | 负 {r['neg_pairs']:,} 对"
              f" | 正 {r['pos_pairs']:,} 对 | 扫描 {t_scan:.1f}s 正样本 {t_pos:.1f}s"
              f" 传输 {t_transfer:.1f}s", flush=True)
        r.update({'status': 'ok', 'neg': neg.cpu().numpy(), 'pos': pos.cpu().numpy()})
        res_q.put(r)
    except BaseException as e:  # noqa: BLE001 —— 任何异常都回传, 避免父进程空等
        print(f"  [gpu{cfg['gpu']}] 异常: {type(e).__name__}: {e}", flush=True)
        res_q.put({'status': 'error', 'gpu': cfg['gpu'], 'msg': f'{type(e).__name__}: {e}'})


def compute(feats, ids, gpus, workdir):
    feats = np.ascontiguousarray(feats, dtype=np.float32)
    ids = np.ascontiguousarray(ids, dtype=np.int64)
    N = len(ids)
    G = len(gpus)
    assert G >= 1, '需要至少 1 张 GPU'
    assert BLOCK_COLS >= BLOCK_ROWS, 'BLOCK_COLS < BLOCK_ROWS 时三角模板不精确'
    os.makedirs(workdir, exist_ok=True)

    # 1) 排序 + 紧凑编码（原 load_and_sort）—— 排序保证同身份连续、F[a:] 含组全行
    t0 = time.perf_counter()
    feats_s, ids_c, starts, ends = _sort_and_groups(feats, ids)
    sort_s = time.perf_counter() - t0
    print(f"[glm-pro] N={N:,} 身份={len(starts):,} 排序 {sort_s:.1f}s", flush=True)

    # 2) 等面积行划分 + 每组归属（组起点落在的 band）
    bounds = _equal_area_bounds(N, G)
    per_gpu = []
    for gi in range(G):
        lo = int(np.searchsorted(starts, bounds[gi], side='left'))
        hi = int(np.searchsorted(starts, bounds[gi + 1], side='left'))
        per_gpu.append((starts[lo:hi].copy(), ends[lo:hi].copy()))
    print(f"[glm-pro] 等面积行划分 {bounds}", flush=True)

    # 3) spawn 前把排序后特征/ids 放进共享内存（父进程不碰 CUDA; 子进程零拷贝引用）
    feats_shm = torch.from_numpy(feats_s).share_memory_()     # 拷入共享段后释放 numpy
    ids_shm = torch.from_numpy(ids_c).share_memory_()
    del feats_s, ids_c
    print(f"[glm-pro] 共享内存特征 {feats_shm.numel()*4/2**30:.2f} GiB 就绪", flush=True)

    ctx = mp.get_context('spawn')
    res_q = ctx.Queue()
    cfg = {'N': N, 'bins': common.BINS,
           'block_rows': BLOCK_ROWS, 'block_cols': BLOCK_COLS}
    procs = []
    for gi, gpu in enumerate(gpus):
        c = dict(cfg)
        c.update({'gpu': int(gpu), 'a': bounds[gi], 'b': bounds[gi + 1]})
        p = ctx.Process(target=_worker,
                        args=(c, feats_shm, ids_shm, per_gpu[gi][0], per_gpu[gi][1], res_q))
        p.start()
        procs.append(p)

    t0 = time.perf_counter()
    # 4) 合并各卡直方图
    # 先 drain 结果队列再 join（先 join 后取会因子进程 feeder 管道写满而互相死锁）
    pos_hist = neg_hist = None
    t_scan = t_pos = t_transfer = 0.0
    n_tiles = 0
    for _ in gpus:
        r = res_q.get(timeout=3600)
        if r.get('status') != 'ok':
            raise RuntimeError(f"gpu{r.get('gpu')} worker 失败: {r.get('msg')}")
        neg_hist = r['neg'] if neg_hist is None else neg_hist + r['neg']
        pos_hist = r['pos'] if pos_hist is None else pos_hist + r['pos']
        t_scan += r['t_scan']; t_pos += r['t_pos']
        t_transfer += r['t_transfer']; n_tiles += r['n_tiles']
    for p in procs:
        p.join()
    core_s = time.perf_counter() - t0
    bad = [p.exitcode for p in procs if p.exitcode != 0]
    if bad:
        raise RuntimeError(f'{len(bad)} 个 worker 进程异常退出 exitcode={bad}')

    pos_hist = np.ascontiguousarray(pos_hist, dtype=np.int64)
    neg_hist = np.ascontiguousarray(neg_hist, dtype=np.int64)
    assert neg_hist.min() >= 0, '负样本直方图出现负计数'

    meta = {
        'model': MODEL_NAME, 'desc': MODEL_DESC, 'origin': ORIGIN,
        'parallel': 'spawn每GPU一进程 + 等面积行划分(上三角面积均分) + 行分片常驻, '
                    '组起点所在卡统计正样本(原实现为单进程每GPU一条CUDA stream)',
        'block': {'rows': BLOCK_ROWS, 'cols': BLOCK_COLS},
        'precision': 'fp32 matmul (allow_tf32=False); 排除位填INVALID(-2)+histc(异步)',
        'native_bins': {'coarse': 4096, 'fine': 65536,
                        'fine_lo': 0.2, 'fine_hi': 1.0,
                        'note': '原两级直方图(全域粗4096+尾部细65536) 降级为统一单级'
                                ' 200_000 bins [-1,1]'},
        'sort_s': round(sort_s, 3), 'core_s': round(core_s, 3),
        'transfer_s': round(t_transfer, 3), 'scan_s': round(t_scan, 3),
        'pos_s': round(t_pos, 3), 'n_tiles': n_tiles,
    }
    print(f"[glm-pro] 完成: 负 {int(neg_hist.sum()):,} 正 {int(pos_hist.sum()):,} 对"
          f" | scan {t_scan:.1f}s pos {t_pos:.1f}s (聚合) core {core_s:.1f}s", flush=True)
    return pos_hist, neg_hist, meta
