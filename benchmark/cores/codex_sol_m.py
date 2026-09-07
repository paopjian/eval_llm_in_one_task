#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Codex-Sol-M core —— 提炼自 codex-sol-m/face_similarity_eval.py

原实现要点（worker / main，见 codex-sol-m/任务文档.md 与脚本）:
    * spawn；每 GPU 领取固定的连续行区间 [n*k//G, n*(k+1)//G)
    * 行块循环（--block-size 默认 512）：x[i0:i1] @ x.T，fp32 且显式开启
      TF32（allow_tf32=True / float32_matmul_precision='high'，注释称对余弦
      相似度评估精度影响可忽略）
    * 每行 i 只取 j>i 的上三角段，按 ids[j0:] == ids[i] 切正/负样本，
      clip((v+1)*(NBINS/2)) + np.bincount 进 int64 直方图（NBINS=20000），
      npz 落盘后由 main 汇总
    * 大 N 时"一次 x[i0:i1] @ x.T 全宽结果"显存压力大（512×N fp32）、
      CPU 逐行 bincount 是主要耗时（203K 样本实测 ~90s）

适配（本 core，保持原思路）:
    * bins: 20,000 -> 统一 200,000（common.BINS）；网格语义与原公式一致
    * x[i0:i1] @ x.T 改为列分块的上三角分块矩阵乘：整块位于行块左侧
      (c1<=i0) 的列块整体跳过（该区 j<i，对已在行 j 端统计，不重复）；
      跨行块的列块用全局列>行掩码截 j>i —— 每对 (i<j) 恰统计一次，
      2M 规模单卡常驻 4GB(特征) + 分块结果 ~134MB，峰值显存 <<20GB
    * 正/负分流改为 GPU 端等价 2D 掩码（同原 ids 相等语义），负样本仍直接
      统计（不做"全体减正"），clamp 到 [-1,1] 后 torch.histc —— histc 与
      原 clip+bincount 网格逐格一致；clamp 保证格内每值恰入一格、求和严格
      守恒（原 clip 把越界值压入边 bin，等效）
    * 去掉 pkl 读取 / pos_values 收集 / 阈值样本对提取 / npz / 绘图；
      结果经 mp.Queue 直接回传
"""
import math
import os
import time

import numpy as np
import torch
import torch.multiprocessing as mp

from .. import common

MODEL_NAME = 'codex-sol-m'
MODEL_DESC = ('Codex-Sol-M 方法：spawn每GPU固定行区间 + 512行块×列分块上三角 '
              '+ fp32矩阵乘(TF32开启)')
ORIGIN = ('codex-sol-m/face_similarity_eval.py (worker/main；--block-size 512, '
          'NBINS 20000 -> 统一 200K，去读取/提取/绘图)')

BLOCK = 512          # 原 --block-size 默认值
COL_CHUNK = 65536    # 列分块宽（原一次乘完整 x.T；列分块避免 512×N fp32 结果 OOM）


def _worker(cfg, feats_shm, ids_shm, res_q):
    """单卡 worker：处理固定行区间 [start,end)，行块内按列分块扫上三角"""
    # 与原实现一致的精度设置：fp32 矩阵乘 + TF32（对余弦相似度评估影响可忽略）
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    gpu = cfg['gpu']
    torch.cuda.set_device(gpu)
    dev = f'cuda:{gpu}'
    B, C = cfg['block'], cfg['col_chunk']
    bins, n = cfg['bins'], cfg['n']
    start, end = cfg['start'], cfg['end']

    feats = feats_shm.to(dev)        # (N,512) fp32 常驻显存
    ids_g = ids_shm.to(dev)
    torch.cuda.synchronize()

    hist_pos = torch.zeros(bins, dtype=torch.int64, device=dev)
    hist_neg = torch.zeros(bins, dtype=torch.int64, device=dev)
    t_mm = t_hist = 0.0
    nb_chunk = 0

    if start < end:
        n_blocks = math.ceil((end - start) / B)
        stride = max(1, n_blocks // 20)
        with torch.inference_mode():
            for bi, i0 in enumerate(range(start, end, B)):
                i1 = min(i0 + B, end)
                for c0 in range(0, n, C):
                    c1 = min(c0 + C, n)
                    # 列块整体在行块左侧：所有 j < i0 <= i，已由行 j 端统计 -> 跳过
                    if c1 <= i0:
                        continue

                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    s = feats[i0:i1] @ feats[c0:c1].t()      # (nrow, ncol) fp32
                    torch.cuda.synchronize()
                    t_mm += time.perf_counter() - t0

                    t0 = time.perf_counter()
                    s.clamp_(common.LO, common.HI)           # 越界值贴边 -> histc 全覆盖
                    eq = ids_g[i0:i1, None] == ids_g[None, c0:c1]   # 同身份掩码
                    if c0 >= i1:
                        # 列块整体在行块右侧：块内所有 (i<j) 均有效
                        posv = s[eq]
                        s.masked_fill_(eq, common.LO - 2.0)  # 仅留负样本值，其余填范围外
                    else:
                        # 跨行块的列块：需全局 j>i 掩码（含块左 j<i 区段）
                        valid = (torch.arange(c0, c1, device=dev)[None, :]
                                 > torch.arange(i0, i1, device=dev)[:, None])
                        keep = valid & ~eq                    # 该列块内有效的负样本位
                        posv = s[valid & eq]
                        s.masked_fill_(~keep, common.LO - 2.0)
                    if posv.numel() > 0:
                        hp = torch.histc(posv, bins=bins, min=common.LO, max=common.HI)
                        hist_pos += hp.round_().to(torch.int64)
                    hn = torch.histc(s, bins=bins, min=common.LO, max=common.HI)
                    hist_neg += hn.round_().to(torch.int64)
                    torch.cuda.synchronize()
                    t_hist += time.perf_counter() - t0
                    nb_chunk += 1

                if (bi + 1) % stride == 0 or i1 == end:
                    print(f"  [gpu{gpu}] 行块 {bi + 1}/{n_blocks} "
                          f"(rows {i1:,}/{end:,}) chunk={nb_chunk} "
                          f"matmul={t_mm:.1f}s hist={t_hist:.1f}s", flush=True)

    torch.cuda.synchronize()
    res_q.put((hist_pos.cpu().numpy(), hist_neg.cpu().numpy(), t_mm, t_hist, nb_chunk))
    print(f"  [gpu{gpu}] 完成 rows [{start},{end}) chunks={nb_chunk} "
          f"matmul={t_mm:.1f}s hist={t_hist:.1f}s", flush=True)


def compute(feats, ids, gpus, workdir):
    """codex-sol-m 核心计算：spawn 每 GPU 固定行区间 + 512 行块列分块上三角"""
    gpus = list(gpus) or [0]            # 与原实现一致：空列表退回单卡
    feats = np.ascontiguousarray(feats, dtype=np.float32)
    ids = np.ascontiguousarray(ids, dtype=np.int64)
    n = len(ids)
    os.makedirs(workdir, exist_ok=True)   # 本 core 不落中间文件，仅保证目录存在

    # spawn 前在主进程建立共享内存张量（不触碰 CUDA）
    feats_shm = torch.from_numpy(feats).share_memory_()
    ids_shm = torch.from_numpy(ids).share_memory_()

    ctx = mp.get_context('spawn')
    res_q = ctx.Queue()
    cfg = {'n': n, 'block': BLOCK, 'col_chunk': COL_CHUNK, 'bins': common.BINS}
    procs = []
    for k, gpu in enumerate(gpus):        # 与原实现相同：按 GPU 序号切连续行区间
        c = dict(cfg)
        c.update({'gpu': gpu, 'start': n * k // len(gpus),
                  'end': n * (k + 1) // len(gpus)})
        p = ctx.Process(target=_worker, args=(c, feats_shm, ids_shm, res_q))
        p.start()
        procs.append(p)

    t0 = time.perf_counter()
    # 先 drain 结果队列再 join（先 join 后取会因子进程 feeder 管道写满而互相死锁）
    pos_hist = neg_hist = None
    t_mm = t_hist = 0.0
    n_chunk = 0
    for _ in gpus:
        hp, hn, mm, hs, nb = res_q.get(timeout=3600)
        pos_hist = hp if pos_hist is None else pos_hist + hp
        neg_hist = hn if neg_hist is None else neg_hist + hn
        t_mm += mm
        t_hist += hs
        n_chunk += nb
    for p in procs:
        p.join()
    core_s = time.perf_counter() - t0
    for p in procs:                        # worker 异常时尽早报错，避免队列阻塞挂死
        if p.exitcode != 0:
            raise RuntimeError(f'codex-sol-m worker 异常退出 exitcode={p.exitcode}')

    assert pos_hist is not None and len(pos_hist) == common.BINS
    assert neg_hist is not None and len(neg_hist) == common.BINS
    meta = {
        'model': MODEL_NAME, 'desc': MODEL_DESC, 'origin': ORIGIN,
        'parallel': 'spawn 每GPU固定行区间连续分片(原实现)；行块内列分块上三角',
        'block': BLOCK, 'col_chunk': COL_CHUNK,
        'precision': 'fp32 矩阵乘，TF32 开启(allow_tf32=True, '
                     'float32_matmul_precision=high，同原实现)',
        'native_bins': 20_000,
        'core_s': core_s, 'matmul_s': t_mm, 'hist_s': t_hist,
        'chunks': n_chunk,
    }
    return pos_hist, neg_hist, meta
