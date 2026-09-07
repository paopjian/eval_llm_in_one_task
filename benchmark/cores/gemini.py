#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemini core —— 提炼自 gemini/face_eval_system.py

原实现要点（face_eval_system.py 的 compute_positive_pairs /
compute_negative_pairs_multi_gpu / _gpu_worker）:
    * 正负分离:
        - 正样本对: 主进程按身份分组（np.diff 定位连续身份区间），组内小矩阵点积
          （fp32），严格上三角提取全部同身份组合 -> 只进正样本直方图
        - 负样本对: GPU 分块扫描，同 ID 对用向量化掩码精确剔除（直方图绝对纯净），
          与正样本完全分离统计
    * 分块: 上三角分块任务 (bi,bj) 且 bi<=bj，chunk_size=4096；任务权重=块面积
      （对角块减半，因只统计严格上三角）
    * 负载均衡: 任务按权重降序排序，贪心分配给当前负载最小 GPU（LPT 静态分配），
      各 GPU 各自领固定任务列表
    * 精度/统计: FP16 Tensor Core GEMM（torch.mm，fp32 累加）；GPU 端
      torch.bincount 流式直方图，bin = ((sim+1)*bins/2).long().clamp(0,bins-1)
    * 掩码优化: 对角块 triu(1) & ~(id 等值)；非对角块仅当两块 ID 范围相交时做
      等值掩码，否则直接 view(-1) 全量统计（无 ID 重叠则不可能存在同身份对）
    * 原 bins=100,000 -> 统一网格 200,000（映射公式不变，仅 total_bins 参数适配）
    * 原用 mp.Manager().dict() 回传曾出现 KeyError -> 改为 spawn + mp.Queue 回传
      （可靠、无 Manager 竞态）；并行分块与正负分离思路保持不变
"""
import os
import time

import numpy as np
import torch
import torch.multiprocessing as mp

from .. import common

MODEL_NAME = 'gemini'
MODEL_DESC = ('Gemini 方法：主进程正样本分组点积 + spawn每GPU贪心静态分块(fp16 GEMM'
              '+bincount流式直方图, 同ID掩码剔除)')
ORIGIN = ('gemini/face_eval_system.py (compute_positive_pairs / '
          '_gpu_worker / compute_negative_pairs_multi_gpu；bins 100K -> 200K；'
          'Manager回传改Queue)')

BLOCK = 4096            # chunk_size（原实现默认 4096）


def _pos_hist_cpu(feats, ids):
    """
    正样本对直方图（主进程 CPU，忠实原 compute_positive_pairs）:
        按身份分组 -> 组内 fp32 点积 -> 严格上三角组合 -> 统一网格直方图。
    原实现假设 ids 已按身份单调连续排序（np.diff>0 分段）；这里先检测有序性，
    无序时用稳定排序构造连续分组（组内容不变，组内组合与顺序无关，结果一致），
    使 core 对任意 id 排列都正确。
    """
    t0 = time.time()
    bins = common.BINS
    scale = bins / 2.0
    N = len(ids)

    diff = np.diff(ids)
    if (diff >= 0).all():
        order = None                              # 已排序: 直接按原始顺序分段
        boundaries = np.flatnonzero(diff > 0) + 1
    else:
        order = np.argsort(ids, kind='stable')    # 无序: 稳定排序得到连续分组
        sids = ids[order]
        boundaries = np.flatnonzero(np.diff(sids) > 0) + 1

    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [N]))
    n_groups = len(starts)

    if order is not None:
        feats = feats[order]

    triu_cache = {}
    vals_list = []
    for s, e in zip(starts, ends):
        L = int(e - s)
        if L < 2:
            continue
        sub = feats[s:e]
        sim_mat = np.dot(sub, sub.T)              # fp32 组内全连接
        if L not in triu_cache:
            triu_cache[L] = np.triu_indices(L, k=1)
        r, c = triu_cache[L]
        vals_list.append(sim_mat[r, c])

    if vals_list:
        sims = np.concatenate(vals_list).astype(np.float32)
    else:
        sims = np.empty(0, dtype=np.float32)

    # 与原 evaluate_tpir_at_fpir 相同映射: clip(((s+1)*bins/2), 0, bins-1)
    # 用 float64 计算避免 fp32 在 1e5 量级上的舍入导致个别样本跨 bin
    idx = np.clip(((sims.astype(np.float64) + 1.0) * scale).astype(np.int64),
                  0, bins - 1)
    pos_hist = np.bincount(idx, minlength=bins).astype(np.int64)
    print(f'[gemini 正样本] 身份组数={n_groups:,} 正对数={int(pos_hist.sum()):,} '
          f'耗时 {time.time()-t0:.2f}s', flush=True)
    return pos_hist


def _neg_worker(cfg, feats_shm, ids_shm, tasks, res_q):
    """
    单卡 worker（spawn，忠实原 _gpu_worker）:
        feats fp32共享内存 -> 每卡转 fp16 常驻；领取本卡静态任务列表；
        每 tile: torch.mm(fp16) -> 同ID掩码/triu剔除 -> bincount 流式累加纯负样本直方图。
    """
    gpu = cfg['gpu']
    torch.cuda.set_device(gpu)
    dev = torch.device(f'cuda:{gpu}')
    B, bins, N = cfg['block'], cfg['bins'], cfg['N']
    cmin, cmax = cfg['cmin'], cfg['cmax']         # 每 chunk 的 id min/max（CPU 标量，免同步）

    t_start = time.time()
    with torch.no_grad():
        feats = feats_shm.to(device=dev, dtype=torch.float16)   # 常驻显存 fp16
        ids_gpu = ids_shm.to(device=dev, dtype=torch.int64)
        torch.cuda.synchronize()
        t_load = time.time() - t_start

        neg_hist = torch.zeros(bins, dtype=torch.int64, device=dev)
        scale = bins / 2.0
        n_done = 0
        n_pairs = 0
        nb_tasks = len(tasks)
        report_every = max(1, nb_tasks // 20)
        sync_every = 16                            # 周期同步防止提交队列积压爆显存

        for bi, bj in tasks:
            r0, r1 = bi * B, min((bi + 1) * B, N)
            c0, c1 = bj * B, min((bj + 1) * B, N)
            is_diag = bi == bj

            sim = torch.mm(feats[r0:r1], feats[c0:c1].t())     # fp16 TensorCore GEMM

            if is_diag:
                # 严格上三角 且 剔除同身份（正样本对）
                tri = torch.triu(torch.ones(r1 - r0, c1 - c0,
                                            dtype=torch.bool, device=dev), diagonal=1)
                keep = tri & ~(ids_gpu[r0:r1, None] == ids_gpu[None, c0:c1])
                vals = sim[keep]
            elif cmin[bi] > cmax[bj] or cmin[bj] > cmax[bi]:
                # 两分块 ID 范围完全不相交 => 不存在同身份对，直接全量统计
                vals = sim.view(-1)
            else:
                # ID 范围相交: 等值掩码精确剔除同身份对（非对角块全在严格上三角区）
                keep = ~(ids_gpu[r0:r1, None] == ids_gpu[None, c0:c1])
                vals = sim[keep]

            # GPU 流式直方图（原实现公式，bins 适配统一网格）
            bin_idx = ((vals.float() + 1.0) * scale).long().clamp_(0, bins - 1)
            neg_hist += torch.bincount(bin_idx, minlength=bins)

            n_pairs += int(vals.numel())
            n_done += 1
            if n_done % sync_every == 0:
                torch.cuda.synchronize()
            if n_done % report_every == 0 or n_done == nb_tasks:
                el = max(time.time() - t_start, 1e-9)
                print(f"  [gemini gpu{gpu}] tile {n_done}/{nb_tasks} "
                      f"已统计 {n_pairs:,} 对 ({el:.1f}s, "
                      f"{n_pairs/el/1e9:.3f} G对/s)", flush=True)

        torch.cuda.synchronize()
        t_loop = time.time() - t_start
        hist_np = neg_hist.cpu().numpy()
    res_q.put((hist_np, n_done, n_pairs, t_loop, t_load))
    print(f'[gemini gpu{gpu}] 完成 {n_done} tiles, 纯负样本 {n_pairs:,} 对, '
          f'加载 {t_load:.1f}s + 计算 {t_loop:.1f}s', flush=True)


def compute(feats, ids, gpus, workdir):
    feats = np.ascontiguousarray(feats, dtype=np.float32)
    ids = np.ascontiguousarray(ids, dtype=np.int64)
    N = int(len(ids))
    bins = common.BINS
    os.makedirs(workdir, exist_ok=True)

    total_pairs = N * (N - 1) // 2
    print(f'[gemini] N={N:,} gpus={len(gpus)} block={BLOCK} bins={bins} '
          f'总对数={total_pairs:,}', flush=True)

    # ---------------- 1) 上三角分块任务规划 + 贪心负载均衡（原实现 LPT） ----------------
    t_plan = time.time()
    nb = (N + BLOCK - 1) // BLOCK
    # 每 chunk 的 id 范围（精确的集合相交判定，不依赖数据排序；用于免掩码优化）
    cmin = [int(ids[i * BLOCK:min((i + 1) * BLOCK, N)].min()) for i in range(nb)]
    cmax = [int(ids[i * BLOCK:min((i + 1) * BLOCK, N)].max()) for i in range(nb)]

    tile_w = []                                    # (weight, bi, bj)
    for bi in range(nb):
        r0, r1 = bi * BLOCK, min((bi + 1) * BLOCK, N)
        for bj in range(bi, nb):
            c0, c1 = bj * BLOCK, min((bj + 1) * BLOCK, N)
            weight = (r1 - r0) * (c1 - c0) * (0.5 if bi == bj else 1.0)
            tile_w.append((weight, bi, bj))
    tile_w.sort(key=lambda x: -x[0])               # 大块先分配

    gpu_tasks = [[] for _ in gpus]
    gpu_loads = [0.0] * len(gpus)
    for weight, bi, bj in tile_w:
        g = int(np.argmin(gpu_loads))
        gpu_tasks[g].append((bi, bj))
        gpu_loads[g] += weight
    print(f'[gemini 规划] chunks={nb} tiles={len(tile_w):,} '
          f'负载均衡耗时 {time.time()-t_plan:.2f}s', flush=True)
    load_sum = sum(gpu_loads) or 1.0
    for g, (ts, ld) in enumerate(zip(gpu_tasks, gpu_loads)):
        print(f'  gpu{g}: {len(ts)} tiles, 负载 {ld/load_sum*100:.2f}%', flush=True)

    # ---------------- 2) spawn 每卡一进程（共享内存传 feats/ids） ----------------
    feats_shm = torch.from_numpy(feats).share_memory_()
    ids_shm = torch.from_numpy(ids).share_memory_()

    ctx = mp.get_context('spawn')
    res_q = ctx.Queue()
    cfg_base = {'block': BLOCK, 'bins': bins, 'N': N, 'cmin': cmin, 'cmax': cmax}

    t_neg0 = time.time()
    procs = []
    for g, gpu in enumerate(gpus):
        c = dict(cfg_base)
        c['gpu'] = int(gpu)
        p = ctx.Process(target=_neg_worker, args=(c, feats_shm, ids_shm,
                                                  gpu_tasks[g], res_q))
        p.start()
        procs.append(p)

    # ---------------- 3) 正样本对（主进程 CPU 分组点积，与 GPU 负样本并行） ----------------
    t_pos0 = time.perf_counter()
    pos_hist = _pos_hist_cpu(feats, ids)
    pos_cpu_s = time.perf_counter() - t_pos0
    pos_n = int(pos_hist.sum())

    # ---------------- 4) 聚合各卡纯负样本直方图 ----------------
    # 注意：先 drain 结果队列再 join —— 若先 join 后取结果，子进程退出时 feeder
    # 线程可能因管道写满而阻塞、父进程又等在 join 上，形成互相等待死锁。
    neg_hist = None
    n_pairs_total = 0
    t_loop_max = 0.0
    t_load_sum = 0.0
    n_done_total = 0
    for _ in gpus:
        hn, nd, np_, tl, tld = res_q.get(timeout=3600)
        neg_hist = hn if neg_hist is None else neg_hist + hn
        n_pairs_total += np_
        n_done_total += nd
        t_loop_max = max(t_loop_max, tl)
        t_load_sum += tld

    for p in procs:
        p.join()

    neg_n = int(neg_hist.sum())
    assert neg_n + pos_n == total_pairs, \
        f'计数不一致: neg {neg_n:,} + pos {pos_n:,} != total {total_pairs:,}'
    assert neg_hist.min() >= 0, '负样本直方图出现负计数'
    t_calc = time.time() - t_neg0
    print(f'[gemini] 负样本 {neg_n:,} 对完成 (wall {t_calc:.2f}s, '
          f'{neg_n/t_calc/1e9:.3f} G对/s), 正样本 {pos_n:,} 对 (CPU {pos_cpu_s:.2f}s), '
          f'校验一致', flush=True)

    meta = {
        'model': MODEL_NAME, 'desc': MODEL_DESC, 'origin': ORIGIN,
        'parallel': 'spawn 每GPU一进程 + 权重降序贪心静态分配(LPT, 对角块×0.5)',
        'block': BLOCK, 'precision': '负: fp16 GEMM(torch.mm)+bincount; 正: CPU fp32点积',
        'native_bins': 100_000, 'bins': bins,
        'plan_s': time.perf_counter() - t_plan, 'pos_cpu_s': round(pos_cpu_s, 3),
        'neg_s': round(t_calc, 3), 'matmul_s': round(t_loop_max, 3),
        'tiles': len(tile_w), 'gpus': len(gpus),
    }
    return pos_hist, neg_hist, meta
