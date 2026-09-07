#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GLM 多线程实验变体 —— 由 benchmark/cores/glm.py 派生，唯一差异：进程 -> 线程。

目的：回答“glm 的多进程架构改成多线程是否还能保持速度，同时能否省掉
共享内存 / 多进程带来的进程树 RSS 膨胀”。

与 cores/glm.py 逐行对齐的部分（算法语义完全一致）:
    * BLOCK=16384、fp32 矩阵乘（allow_tf32=False）、sim clamp 到 [-1,1]
    * tile (bi,bj) 共享队列动态领取、宽 tile 先入队 => 天然负载均衡
    * 对角 tile 用 triu 掩码填范围外值，仅统计严格上三角
    * 同一轮扫描内用 id 等值掩码收集正样本对，histc 进 hist_pos
    * 负样本直方图 = 全体直方图 - 正样本直方图（neg = all - pos）

与 cores/glm.py 的不同（只改并行载体，不改数值路径）:
    * multiprocessing.spawn + share_memory_() 共享内存  ->  threading 线程 + queue.Queue
    * 每 GPU 一个线程，线程内 set_device；feats/ids 直接用 numpy 零拷贝视图 H2D
      （无共享内存、无 7 次进程启动 / 重复 import / 重复 CUDA 初始化）
    * 结果回传：多进程用 res_q -> 线程用 out[slot] 列表槽位

额外可选项（仅用于 CPU 冒烟/实验，正式 GPU 跑时不用）:
    * compute(..., device='cpu', block=64) 可强制走 CPU、缩小分块，便于无 GPU 环境
      做正确性验证。

不参与正式评估，仅用于实验对比。
"""
import math
import os
import queue
import threading
import time

import numpy as np
import torch

from .. import common

MODEL_NAME = 'glm-threaded'
MODEL_DESC = 'GLM 方法(实验变体)：多线程每GPU一线程 + 动态tile队列(宽优先) + fp32'
ORIGIN = 'benchmark/experiments/glm_threaded.py (由 cores/glm.py 派生，进程改线程)'

BLOCK = 16384


def _worker(cfg, feats_np, ids_np, tile_q, out, slot):
    """单卡 worker（线程版）：动态取 tile，一次扫描得到全体上三角直方图 + 正样本直方图"""
    dev = cfg.get('device') or f'cuda:{cfg["gpu"]}'

    def _sync():
        if dev.startswith('cuda'):
            torch.cuda.synchronize()

    if dev.startswith('cuda'):
        torch.cuda.set_device(cfg['gpu'])
        torch.backends.cuda.matmul.allow_tf32 = False

    B, bins, N = cfg['block'], cfg['bins'], cfg['N']
    feats = torch.from_numpy(feats_np).to(dev)      # (N,512) fp32 常驻显存（零拷贝视图 H2D）
    ids_gpu = torch.from_numpy(ids_np).to(dev)      # (N,)
    _sync()

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

        _sync(); t0 = time.perf_counter()
        simf = A @ C.t()                                   # (b1,b2) fp32
        simf.clamp_(common.LO, common.HI)
        if is_diag:
            tri = torch.ones(simf.shape[0], simf.shape[1],
                             dtype=torch.bool, device=dev).triu_(1)
            simf.masked_fill_(~tri, common.LO - 2.0)       # 下三角/对角填范围外 -> histc 丢弃
        _sync(); t_mm += time.perf_counter() - t0

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

    _sync()
    out[slot] = (hist_all.cpu().numpy(), hist_pos.cpu().numpy(),
                 t_mm, t_hist, n_done)
    print(f"  [gpu{cfg['gpu']}] 完成 {n_done} tiles | matmul {t_mm:.1f}s hist {t_hist:.1f}s",
          flush=True)


def compute(feats, ids, gpus, workdir, device=None, block=None):
    """线程版 compute。与 cores/glm.compute 契约一致，额外 device/block 仅用于实验。

    gpus: GPU 编号列表（每 GPU 一线程）。device 强制为 'cpu' 时走 CPU（冒烟用）。
    """
    feats = np.ascontiguousarray(feats, dtype=np.float32)
    ids = np.ascontiguousarray(ids, dtype=np.int64)
    N = len(ids)
    os.makedirs(workdir, exist_ok=True)

    B = block or BLOCK
    nb = math.ceil(N / B)
    tiles = [(bi, bj) for bi in range(nb) for bj in range(bi, nb)]
    tiles.sort(key=lambda t: -(t[1] - t[0]))               # 宽 tile 先入队

    tile_q = queue.Queue()
    for t in tiles:
        tile_q.put(t)
    for _ in gpus:
        tile_q.put(None)

    cfg = {'N': N, 'block': B, 'bins': common.BINS, 'nb_tiles': len(tiles)}
    if device:
        cfg['device'] = device

    out = [None] * len(gpus)
    threads = []
    t0 = time.perf_counter()
    for slot, gpu in enumerate(gpus):
        c = dict(cfg); c['gpu'] = gpu
        th = threading.Thread(target=_worker,
                              args=(c, feats, ids, tile_q, out, slot),
                              name=f'gpu-{gpu}')
        th.start(); threads.append(th)
    for th in threads:
        th.join()
    core_s = time.perf_counter() - t0

    hist_all = None
    hist_pos = None
    t_mm = t_hist = 0.0
    for r in out:
        if r is None or isinstance(r, Exception):
            raise RuntimeError(f'worker 失败: {r!r}')
        ha, hp, mm, hs, nd = r
        hist_all = ha if hist_all is None else hist_all + ha
        hist_pos = hp if hist_pos is None else hist_pos + hp
        t_mm += mm; t_hist += hs

    neg_hist = hist_all - hist_pos
    assert neg_hist.min() >= 0, '负样本直方图出现负计数'
    meta = {
        'model': MODEL_NAME, 'desc': MODEL_DESC, 'origin': ORIGIN,
        'parallel': 'threading 每GPU一线程 + 动态tile队列(宽优先)', 'block': B,
        'precision': 'fp32 (allow_tf32=False)', 'native_bins': 2_000_000,
        'core_s': core_s, 'matmul_s': t_mm, 'hist_s': t_hist,
    }
    return hist_pos, neg_hist, meta
