#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GLM fp16 实验变体 —— 由 benchmark/cores/glm.py 自动生成，仅供 fp16 vs fp32 速度/精度对比，
不参与正式评估（与 glm core 唯一差异：matmul 走 fp16 Tensor Core）

原实现要点（见 glm/eval_similar.py 头部注释与 gpu_worker/run_pass）:
    * spawn 每 GPU 一个进程；tile (bi,bj) 共享队列动态领取，宽 tile 先入队 => 天然负载均衡
    * fp32 矩阵乘（关闭 TF32，数值精确），相似度 clamp 到 [-1,1]
    * 对角 tile 用 triu 掩码填范围外值，仅统计严格上三角
    * 同一轮扫描内用 id 等值掩码收集正样本对，histc 进 hist_pos
      负样本直方图 = 全体直方图 - 正样本直方图
    * 原 bins=2,000,000；统一网格下改为 200,000（其余逻辑不变）
"""
import math
import os
import time

import numpy as np
import torch
import torch.multiprocessing as mp

from .. import common

MODEL_NAME = 'glm-fp16'
MODEL_DESC = 'GLM-5.3-flash 方法(实验变体)：spawn + 动态tile队列 + fp16 matmul'
ORIGIN = 'benchmark/experiments/glm_fp16.py (自动生成自 cores/glm.py)'

BLOCK = 16384


def _worker(cfg, feats_shm, ids_shm, tile_q, res_q):
    """单卡 worker：动态取 tile，一次扫描得到全体上三角直方图 + 正样本直方图"""
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = f'cuda:{cfg["gpu"]}'
    torch.cuda.set_device(dev)
    B, bins, N = cfg['block'], cfg['bins'], cfg['N']
    feats = feats_shm.to(dev)          # (N,512) fp32 常驻显存
    ids_gpu = ids_shm.to(dev)          # (N,)
    torch.cuda.synchronize()

    hist_all = torch.zeros(bins, dtype=torch.int64, device=dev)
    hist_pos = torch.zeros(bins, dtype=torch.int64, device=dev)
    t_mm = t_hist = 0.0
    n_done = 0
    nb_tiles = cfg['nb_tiles']

    while True:
        tile = tile_q.get()
        if tile is None:
            break
        bi, bj = tile
        r0, r1 = bi * B, min((bi + 1) * B, N)
        c0, c1 = bj * B, min((bj + 1) * B, N)
        A, C = feats[r0:r1], feats[c0:c1]
        is_diag = bi == bj

        torch.cuda.synchronize(); t0 = time.perf_counter()
        simf = (A.half() @ C.t().half()).float()            # 实验: fp16 tensor-core A/B
        simf.clamp_(common.LO, common.HI)
        if is_diag:
            tri = torch.ones(simf.shape[0], simf.shape[1],
                             dtype=torch.bool, device=dev).triu_(1)
            simf.masked_fill_(~tri, common.LO - 2.0)       # 下三角/对角填范围外 -> histc 丢弃
        torch.cuda.synchronize(); t_mm += time.perf_counter() - t0

        t0 = time.perf_counter()
        h = torch.histc(simf, bins=bins, min=common.LO, max=common.HI)
        hist_all += h.round_().to(torch.int64)

        pos_mask = ids_gpu[r0:r1, None] == ids_gpu[None, c0:c1]
        if is_diag:
            pos_mask &= tri
        pv = simf[pos_mask]
        if pv.numel() > 0:
            hp = torch.histc(pv, bins=bins, min=common.LO, max=common.HI)
            hist_pos += hp.round_().to(torch.int64)
        t_hist += time.perf_counter() - t0

        n_done += 1
        if n_done % 20 == 0 or n_done == nb_tiles:
            print(f"  [gpu{cfg['gpu']}] tile {n_done}/{nb_tiles} "
                  f"matmul={t_mm/max(n_done,1)*1000:.0f}ms hist={t_hist/max(n_done,1)*1000:.0f}ms",
                  flush=True)

    torch.cuda.synchronize()
    res_q.put((hist_all.cpu().numpy(), hist_pos.cpu().numpy(),
               t_mm, t_hist, n_done))
    print(f"  [gpu{cfg['gpu']}] 完成 {n_done} tiles | matmul {t_mm:.1f}s hist {t_hist:.1f}s",
          flush=True)


def compute(feats, ids, gpus, workdir):
    feats = np.ascontiguousarray(feats, dtype=np.float32)
    ids = np.ascontiguousarray(ids, dtype=np.int64)
    N = len(ids)
    os.makedirs(workdir, exist_ok=True)

    # spawn 前先在主进程建立共享内存张量（不触碰 CUDA）
    feats_shm = torch.from_numpy(feats).share_memory_()
    ids_shm = torch.from_numpy(ids).share_memory_()

    nb = math.ceil(N / BLOCK)
    tiles = [(bi, bj) for bi in range(nb) for bj in range(bi, nb)]
    tiles.sort(key=lambda t: -(t[1] - t[0]))               # 宽 tile 先入队

    ctx = mp.get_context('spawn')
    tile_q = ctx.Queue()
    for t in tiles:
        tile_q.put(t)
    for _ in gpus:
        tile_q.put(None)
    res_q = ctx.Queue()

    cfg = {'N': N, 'block': BLOCK, 'bins': common.BINS, 'nb_tiles': len(tiles)}
    procs = []
    for gpu in gpus:
        c = dict(cfg); c['gpu'] = gpu
        p = ctx.Process(target=_worker, args=(c, feats_shm, ids_shm, tile_q, res_q))
        p.start(); procs.append(p)

    t0 = time.perf_counter()
    # 先取回全部结果再 join：若先 join，子进程退出时 feeder 线程可能因管道写满而阻塞（死锁）
    parts = [res_q.get(timeout=3600) for _ in gpus]
    for p in procs:
        p.join()
    core_s = time.perf_counter() - t0

    hist_all = None
    hist_pos = None
    t_mm = t_hist = 0.0
    for ha, hp, mm, hs, nd in parts:
        hist_all = ha if hist_all is None else hist_all + ha
        hist_pos = hp if hist_pos is None else hist_pos + hp
        t_mm += mm; t_hist += hs

    neg_hist = hist_all - hist_pos
    assert neg_hist.min() >= 0, '负样本直方图出现负计数'
    meta = {
        'model': MODEL_NAME, 'desc': MODEL_DESC, 'origin': ORIGIN,
        'parallel': 'spawn 每GPU一进程 + 动态tile队列(宽优先)', 'block': BLOCK,
        'precision': 'fp32 (allow_tf32=False)', 'native_bins': 2_000_000,
        'core_s': core_s, 'matmul_s': t_mm, 'hist_s': t_hist,
    }
    return hist_pos, neg_hist, meta
