#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek (v4-flash) core —— 提炼自 deepseek/run_eval.py + deepseek/faireval_lib.py + deepseek/gpu_worker.py

原实现要点（见 deepseek/faireval_lib.py 头部注释与评估报告）:
    * 每 GPU 一个独立子进程（原实现 subprocess 起 gpu_worker.py 自行载数据写 npz；
      本 core 改用统一模板 spawn + 共享内存张量 —— 并行结构/分卡方式等价，避免 fork+CUDA 死锁）
    * 多卡负载均衡: 行块按 GPU 数轮询静态分卡 —— assign_row_blocks 返回
      GPU_k 负责行块 k, k+G, k+2G, ...；行块 i 的属主承担全部上三角 tile (i, j>=i)，
      三角工作区（行块越靠前列块越多）被轮询摊平 => 近似天然负载均衡，无动态任务队列
    * 精度: fp32 矩阵乘；相似度 = 特征点积（特征已 L2 归一化）。
      原代码未显式关 TF32，依赖 torch matmul 默认 allow_tf32=False；此处显式设置以杜绝环境差异
    * 直方图不用 histc: 相似度 S 原地量化成 bin 序号 floor((S+1)*nbins/2)，clamp 到
      [0, nbins-1]，随后 torch.bincount —— 原生 nbins=200_000 恰为统一网格(宽1e-5)，
      与 common 边语义一致，无需重采样
    * 对角块 (i,i): 利用 S=A@A^T 逐位对称（原自检确认）—— 全块 bincount 含镜像双份与
      自相似对角线: posfull 同身份全计数、dh 对角线自相似、allh 全体；
      去对角线后 (posfull-dh)//2、(allh-posfull)//2 即严格上三角每对恰一次。
      奇偶守卫: 若某 bin 镜像计数为奇（fp32 非逐位对称 + 跨 bin 边界 => //2 会漏计），
      该块退化为严格上三角掩码整块重计 —— 保证"每对恰统计一次"
    * 非对角块 (i<j): 行块全部索引 < 列块全部索引，块内每个元素即一条唯一合法样本对
"""
import math
import os
import time

import numpy as np
import torch
import torch.multiprocessing as mp

from .. import common

MODEL_NAME = 'deepseek'
MODEL_DESC = ('DeepSeek-v4-flash 方法：spawn每GPU一进程 + 行块轮询静态分卡(三角区天然均衡) '
              '+ fp32精确(关TF32) + 量化bincount直方图(对角块对称除2, 奇偶守卫兜底)')
ORIGIN = ('deepseek/faireval_lib.py (_worker_hist_core/run_hist_pass/assign_row_blocks) + '
          'deepseek/gpu_worker.py + deepseek/run_eval.py；子进程 -> spawn+共享内存；'
          'nbins=200K 与统一网格一致，无重采样')

BLOCK = 8192


def _bounds(N, block):
    """每行块的 (r0, r1)（与 faireval_lib._mk_jobs 的 bounds 相同）"""
    return [(k * block, min((k + 1) * block, N))
            for k in range(math.ceil(N / block))]


def _assign_row_blocks(N, block, gpus):
    """行块轮询分配（忠实 faireval_lib.assign_row_blocks）: 第 g_i 个 GPU -> 行块 g_i, g_i+G, ..."""
    nblk = math.ceil(N / block)
    return {g: list(range(g_i, nblk, len(gpus))) for g_i, g in enumerate(gpus)}


def _worker_hist_core(cfg, feats_shm, ids_shm, res_q):
    """单 GPU: 上三角分块 matmul -> 量化 bincount -> 正/负直方图
    （忠实 deepseek/faireval_lib.py:_worker_hist_core；cfg['rows'] 为本卡行块列表）"""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_grad_enabled(False)
    dev = f'cuda:{cfg["gpu"]}'
    torch.cuda.set_device(dev)
    B, nbins, N = cfg['block'], cfg['bins'], cfg['N']
    F = feats_shm.to(dev)                # (N,512) fp32 常驻显存
    IDS = ids_shm.to(dev)                # (N,) int64
    torch.cuda.synchronize()

    pos_hist = torch.zeros(nbins, dtype=torch.int64, device=dev)
    neg_hist = torch.zeros(nbins, dtype=torch.int64, device=dev)
    scale = torch.tensor(0.5 * nbins, dtype=torch.float32, device=dev)   # 1/bin宽
    bounds = _bounds(N, B)
    nblk = len(bounds)
    t_calc = 0.0
    t_w0 = time.perf_counter()
    done = 0
    n_jobs = sum(nblk - i for i in cfg['rows'])

    for i in cfg['rows']:
        r0, r1 = bounds[i]
        for j in range(i, nblk):
            c0, c1 = bounds[j]
            A, C = F[r0:r1], F[c0:c1]
            t_s = time.perf_counter()
            S = A @ C.t()                                        # (m,n) fp32 精确
            S.add_(1.0).mul_(scale).clamp_(0, nbins - 1)         # 原地量化 -> bin序号(float)
            bins_v = S.reshape(-1).to(torch.int64)               # 全部元素所在 bin
            allh = torch.bincount(bins_v, minlength=nbins)
            eq2 = IDS[r0:r1, None] == IDS[c0:c1][None, :]        # (m,n) 同身份掩码
            eqf = eq2.reshape(-1)
            posfull = torch.bincount(bins_v[eqf], minlength=nbins)   # 同身份元素(含镜像/对角)
            if i == j:
                # ---- 对角块: S 逐位对称 => 非对角每对以镜像双份出现 ----
                dh = torch.bincount(S.diagonal().to(torch.int64), minlength=nbins)  # 自相似对角
                pe = posfull - dh                                # 同身份且非对角(双份)
                ne = allh - posfull                              # 异身份(双份)
                if bool((((pe | ne) & 1).any()).item()):
                    # 奇偶守卫: fp32 非逐位对称导致镜像双份跨 bin 时 //2 会漏计
                    # (原 deepseek 曾因直方图计数不一致失败)。退化为本块严格上三角掩码
                    # 重计 -> 每对恰一次、无对角、无镜像，计数严格精确。
                    tri = torch.ones_like(eq2).triu_(1)
                    sel = tri.reshape(-1)
                    pos_hist += torch.bincount(bins_v[sel & eqf], minlength=nbins)
                    neg_hist += torch.bincount(bins_v[sel & ~eqf], minlength=nbins)
                    print(f"  [gpu{cfg['gpu']}] tile({i},{j}) 奇偶守卫触发: "
                          f"对称假设失效, 掩码重计", flush=True)
                else:
                    pos_hist += pe // 2
                    neg_hist += ne // 2
            else:
                # ---- 非对角块 i<j: 本块行索引全部 < 列索引 => 每元素一条唯一合法对 ----
                pos_hist += posfull
                neg_hist += allh - posfull
            t_calc += time.perf_counter() - t_s

            done += 1
            if done % 20 == 0 or done == n_jobs:
                print(f"  [gpu{cfg['gpu']}] tile {done}/{n_jobs} calc={t_calc:.1f}s",
                      flush=True)

    torch.cuda.synchronize()
    res_q.put((cfg['gpu'], pos_hist.cpu().numpy(), neg_hist.cpu().numpy(),
               t_calc, time.perf_counter() - t_w0))
    print(f"  [gpu{cfg['gpu']}] 完成 {done} tiles | calc {t_calc:.1f}s", flush=True)


def compute(feats, ids, gpus, workdir):
    feats = np.ascontiguousarray(feats, dtype=np.float32)
    ids = np.ascontiguousarray(ids, dtype=np.int64)
    N = len(ids)
    os.makedirs(workdir, exist_ok=True)

    # spawn 前先在主进程建立共享内存张量（主进程不触碰 CUDA）
    feats_shm = torch.from_numpy(feats).share_memory_()
    ids_shm = torch.from_numpy(ids).share_memory_()

    rows_map = _assign_row_blocks(N, BLOCK, gpus)          # 静态轮询分卡，无共享任务队列
    ctx = mp.get_context('spawn')
    res_q = ctx.Queue()
    cfg_base = {'N': N, 'block': BLOCK, 'bins': common.BINS}
    procs = []
    for gpu in gpus:
        c = dict(cfg_base)
        c['gpu'] = gpu
        c['rows'] = rows_map[gpu]
        p = ctx.Process(target=_worker_hist_core, args=(c, feats_shm, ids_shm, res_q))
        p.start()
        procs.append(p)

    t0 = time.perf_counter()
    # 先 drain 结果队列再 join（先 join 后取会因子进程 feeder 管道写满而互相死锁）
    pos_hist = np.zeros(common.BINS, dtype=np.int64)
    neg_hist = np.zeros(common.BINS, dtype=np.int64)
    calc_sum = 0.0
    wall_max = 0.0
    per_gpu = []
    for _ in gpus:
        g, ph, nh, tc, tw = res_q.get(timeout=3600)
        pos_hist += ph
        neg_hist += nh
        calc_sum += tc
        wall_max = max(wall_max, tw)
        per_gpu.append({'gpu': g, 'calc_s': round(tc, 3), 'wall_s': round(tw, 3)})
    assert len(pos_hist) == common.BINS and len(neg_hist) == common.BINS

    for p in procs:
        p.join()
    core_s = time.perf_counter() - t0

    bad = [(p.pid, p.exitcode) for p in procs if p.exitcode != 0]
    if bad:
        raise RuntimeError(f'GPU worker 异常退出: {bad}')

    meta = {
        'model': MODEL_NAME, 'desc': MODEL_DESC, 'origin': ORIGIN,
        'parallel': 'spawn 每GPU一进程 + 行块轮询静态分卡(行块i属主承担全部上三角tile, 三角区天然均衡)',
        'block': BLOCK, 'precision': 'fp32 (allow_tf32=False)',
        'native_bins': 200_000,                     # 与统一网格一致，无重采样
        'core_s': round(core_s, 3),
        'gpu_calc_s': round(calc_sum, 3),
        'gpu_wall_max_s': round(wall_max, 3),
        'per_gpu': per_gpu,
    }
    return pos_hist, neg_hist, meta
