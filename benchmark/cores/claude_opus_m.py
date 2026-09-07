#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
claude-opus-m core —— 提炼自 claude-opus-m/eval_similarity_final.py
（其计算逻辑与 eval_similarity_v6.py 相同，final 只加可视化；见 claude-opus-m/README.md 演进表）

原实现要点（process_row_range / compute_similarity_multigpu，行主分配）:
    * 静态行区间分片: rows_per_gpu = N // n_gpu；GPU k 只负责自己行区间的行 i，
      对每个 i 只算与全局 j>i 的相似度 —— 每对 (i<j) 恰由行 i 的属主卡统计一次
      （无任务队列、无 tile 动态调度，与本仓 glm 的 tile 队列方案形成对照）
    * 行块双段处理（块长 chunk_size）:
        ① 块内上三角: 逐行 i 与块内后续行 (i, i_end) 的 strip 矩阵乘
        ② 块外全部列: 行块 × [i_end, N) 的列块矩阵乘
    * 直方图在 CPU 侧: bin = int((sim+1)/2 * bins)（截断）并 clip 到 [0, bins-1]，
      正/负掩码分开累加（np.add.at）=> 每对恰入一桶，正/负计数与理论严格一致
    * 原 bins=10_000；统一网格下改为 200_000（映射公式不变）
    * 结果原走 manager.dict；按框架要求改为结果 Queue 回传，不落临时文件

本 core 保留上述结构；以下为 bit 级等价/纯提速改动（逐项注释）:
    * np.add.at(hist, idx[mask], 1) -> np.bincount(idx[mask], minlength=bins)
      （每元素恰好 +1，两者计数逐桶一致）
    * 大块的负桶 = bincount(全块) - bincount(正子集)：每块内正/负掩码互补，逐桶相等
    * 大块的桶号算术改在 GPU 上做、只回传 int32 索引（同一 fp32 公式，同一截断/clip）
    * allow_tf32=False：原代码未显式设置；统一参考/smoke 均关 TF32，
      此处显式关闭以保证可复现并与参考一致
"""
import time

import numpy as np
import torch
import torch.multiprocessing as mp

from .. import common

MODEL_NAME = 'claude-opus-m'
MODEL_DESC = ('Claude Opus-M：静态行区间分片（行主、与全局右侧列配对、每对恰一次）'
              '+ 块内逐行 strip / 块外列块 矩阵乘 + CPU int32 分桶直方图')
ORIGIN = ('claude-opus-m/eval_similarity_final.py: process_row_range / '
          'compute_similarity_multigpu（10k bins -> 统一 200k；manager.dict -> Queue；'
          'np.add.at -> 等价的 np.bincount）')
BLOCK = 8192


def _bin_row_strip(sv, pos_mask, pos_hist, neg_hist, bins):
    """块内逐行 strip 的分桶（沿用原代码 CPU 公式与 np.add.at 语义，块很小）"""
    idx = ((sv + 1) / 2 * bins).astype(np.int32)      # 原公式: fp32 运算后截断
    np.clip(idx, 0, bins - 1, out=idx)                # 原代码: clip 到 [0, bins-1]
    if pos_mask.any():
        np.add.at(pos_hist, idx[pos_mask], 1)
    np.add.at(neg_hist, idx[~pos_mask], 1)


def _worker(cfg, feats_shm, ids_shm, res_q):
    """单卡 worker：负责行区间 [row_start, row_end)，每行只算与全局 j>i 的对"""
    gpu, B, bins, N = cfg['gpu'], cfg['block'], cfg['bins'], cfg['N']
    rs, re = cfg['row_start'], cfg['row_end']
    t0 = time.perf_counter()
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        dev = f'cuda:{gpu}'
        torch.cuda.set_device(dev)
        feats = feats_shm.to(dev)                     # (N,512) fp32 常驻显存
        ids_g = ids_shm.to(dev)                       # (N,) int64（GPU 等值掩码）
        ids_np = ids_shm.numpy()                      # CPU 零拷贝视图（逐行小掩码用）
        torch.cuda.synchronize()

        pos_hist = np.zeros(bins, dtype=np.int64)
        neg_hist = np.zeros(bins, dtype=np.int64)
        t_mm = t_bin = 0.0
        n_blocks = 0
        t_last = time.perf_counter()

        with torch.no_grad():
            for i0 in range(rs, re, B):
                i1 = min(i0 + B, re)

                # ---- ① 块内上三角: 逐行 rr 与块内后续行 (rr, i1) 的 strip ----
                for rr in range(i0, i1 - 1):          # 末行无后续行，跳过
                    s0 = time.perf_counter()
                    sim = feats[rr:rr + 1] @ feats[rr + 1:i1].t()   # (1, i1-rr-1)
                    t_mm += time.perf_counter() - s0
                    s0 = time.perf_counter()
                    sv = sim.reshape(-1).cpu().numpy()
                    p = ids_np[rr] == ids_np[rr + 1:i1]  # 小掩码 CPU 比较（与 GPU 等值等价）
                    _bin_row_strip(sv, p, pos_hist, neg_hist, bins)
                    t_bin += time.perf_counter() - s0
                    n_blocks += 1

                # ---- ② 块外全部列: 行块 × [i1, N) 的列块（含其它卡行区间的行）----
                A = feats[i0:i1]
                ia = ids_g[i0:i1]
                for j0 in range(i1, N, B):
                    j1 = min(j0 + B, N)
                    s0 = time.perf_counter()
                    sim = A @ feats[j0:j1].t()                      # (nrow, nj)
                    # 同一原公式的 GPU 版: fp32 -> int32 截断 -> clip
                    idx32 = torch.clamp((((sim + 1) / 2) * bins).to(torch.int32),
                                        0, bins - 1)
                    posm = ia[:, None] == ids_g[j0:j1][None, :]     # (nrow, nj) bool
                    t_mm += time.perf_counter() - s0
                    s0 = time.perf_counter()
                    idx_np = idx32.cpu().numpy()
                    nz = posm.nonzero()
                    if nz.numel():
                        pidx = idx32[nz[:, 0], nz[:, 1]].cpu().numpy()
                        bc_pos = np.bincount(pidx, minlength=bins)
                    else:
                        bc_pos = np.zeros(bins, dtype=np.int64)
                    bc_all = np.bincount(idx_np.ravel(), minlength=bins)
                    pos_hist += bc_pos
                    neg_hist += bc_all - bc_pos       # 块内 pos/neg 互补 => 无重复无遗漏
                    t_bin += time.perf_counter() - s0
                    n_blocks += 1

                    if time.perf_counter() - t_last > 12.0:
                        print(f'  [gpu{gpu}] 行[{rs},{re}) 进度 {n_blocks} 块 '
                              f'已算 {pos_hist.sum() + neg_hist.sum():,} 对 耗时 '
                              f'{time.perf_counter() - t0:.0f}s', flush=True)
                        t_last = time.perf_counter()

        torch.cuda.synchronize()
        res_q.put(('ok', pos_hist, neg_hist, t_mm, t_bin, n_blocks))
        print(f'  [gpu{gpu}] 完成: 正={pos_hist.sum():,} 负={neg_hist.sum():,} '
              f'({n_blocks} 块, matmul {t_mm:.1f}s hist {t_bin:.1f}s)',
              flush=True)
    except Exception as e:                            # noqa: BLE001
        import traceback
        res_q.put(('err', f'[gpu{gpu}] {type(e).__name__}: {e}\n'
                          f'{traceback.format_exc()[-1500:]}'))
        print(f'  [gpu{gpu}] 失败: {type(e).__name__}: {e}', flush=True)


def compute(feats, ids, gpus, workdir):
    feats = np.ascontiguousarray(feats, dtype=np.float32)
    ids = np.ascontiguousarray(ids, dtype=np.int64)
    N = int(ids.shape[0])
    ng = len(gpus)
    if ng < 1 or N < 2:
        raise ValueError(f'需要 >=1 张卡且 N>=2 (N={N}, gpus={gpus})')

    # 静态行区间切分（原实现: 每卡 N//n_gpu 行，末卡拿余数）
    rp = N // ng
    ranges = [(k * rp, N if k == ng - 1 else (k + 1) * rp) for k in range(ng)]
    for k, (rs, re) in enumerate(ranges):
        print(f'  [规划] gpu{gpus[k]}: 行 {rs} - {re} ({re - rs} 行)', flush=True)

    # spawn 前先在主进程建立共享内存张量（不触碰 CUDA；worker 只按需映射）
    feats_shm = torch.from_numpy(feats).share_memory_()
    ids_shm = torch.from_numpy(ids).share_memory_()

    ctx = mp.get_context('spawn')
    res_q = ctx.Queue()
    procs = []
    for gpu, (rs, re) in zip(gpus, ranges):
        cfg = {'gpu': gpu, 'row_start': rs, 'row_end': re,
               'block': BLOCK, 'bins': common.BINS, 'N': N}
        p = ctx.Process(target=_worker, args=(cfg, feats_shm, ids_shm, res_q))
        p.start()
        procs.append(p)

    # 边收边等：避免结果积压把管道写满导致 join 死锁
    got = []
    remaining = ng
    while remaining > 0:
        try:
            item = res_q.get(timeout=30)
        except Exception:                             # noqa: BLE001
            if not any(p.is_alive() for p in procs):
                raise RuntimeError(f'claude-opus-m worker 提前退出，只收到 '
                                   f'{len(got)}/{ng} 份结果')
            continue
        remaining -= 1
        got.append(item)
    for p in procs:
        p.join()

    errs = [it[1] for it in got if it[0] == 'err']
    if errs:
        raise RuntimeError('worker 出错: ' + '; '.join(errs))

    t0 = time.perf_counter()
    pos_hist = neg_hist = None
    t_mm = t_bin = 0.0
    for tag, hp, hn, mm, hs, nb in got:               # noqa: B007
        pos_hist = hp if pos_hist is None else pos_hist + hp
        neg_hist = hn if neg_hist is None else neg_hist + hn
        t_mm += mm
        t_bin += hs
    core_s = time.perf_counter() - t0

    # 计数自检（与 run_one 的统一校验同一标准，提前暴露分桶/覆盖 bug）
    total, exp_pos, exp_neg = common.pair_stats(ids)
    got_pos, got_neg = int(pos_hist.sum()), int(neg_hist.sum())
    if got_pos != exp_pos or got_neg != exp_neg or got_pos + got_neg != total:
        raise RuntimeError(f'计数校验失败: pos={got_pos:,} vs {exp_pos:,} '
                           f'neg={got_neg:,} vs {exp_neg:,} '
                           f'(合计 {got_pos + got_neg:,} vs {total:,})')

    meta = {
        'model': MODEL_NAME, 'desc': MODEL_DESC, 'origin': ORIGIN,
        'parallel': 'spawn 每GPU一进程 + 静态行区间切分(行主, 每行只算右上全局列)',
        'block': BLOCK, 'precision': 'fp32 (allow_tf32=False)',
        'native_bins': 10_000,
        'core_s': round(core_s, 2), 'matmul_s': round(t_mm, 1),
        'hist_s': round(t_bin, 1),
    }
    print(f'[汇总] claude-opus-m 完成: 正={got_pos:,} 负={got_neg:,} '
          f'(合计 {got_pos + got_neg:,}) matmul={t_mm:.1f}s hist={t_bin:.1f}s',
          flush=True)
    return pos_hist, neg_hist, meta
