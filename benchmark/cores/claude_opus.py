#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
claude-opus core —— 提炼自 claude-opus/eval_v5_final.py（“版本5: 最终优化版本”）。
选它作方法来源的事实依据：该文件是目录中最后修改/实际执行的文件（时间戳最新），
且 02-测试报告/FINAL_REPORT_200W.md 实测 claude-opus 报出的
/tmp/.../gpu_0.pkl 临时文件缺失，正是 v5 的 tempfile+gpu_i.pkl 设计 ——
v6–v9 及其配套宣传文档与其自述运行日志不符，仅作历史参考，不构成本 core 的方法来源。

原实现要点（compute_row_range_worker / compute_similarity_multigpu）:
    * 并行: 静态行区间分片 rows_per_gpu = N // n_gpu（末卡拿余数）；
      GPU k 负责自己行区间内的每个行块，列块只扫全局 j>i 的右上方
      —— 每对 (i<j) 恰由“行 i 的属主卡”统计一次；无任务队列/动态调度，
      前卡重、后卡闲是该方法的固有负载不均衡（予以保留，作为方法本色）
    * 分块: chunk_size=5000；对行块起点 i（全局索引），列块 j 自 i 起步进
      5000 扫到 N；对角块 (i==j，行列同起点，行块尾部短于 5000 时亦然)
      用 triu(diagonal=1) 只保留严格上三角，非对角块整体有效
      （其列起点 ≥ i+5000，必在行块所有行之后）=> 每对恰统计一次
    * 精度: fp32 matmul（本 core 显式 allow_tf32=False，与统一参考一致）
    * 正负划分: ids 等值掩码 —— 正样本=同身份、负样本=异身份
    * 二次扫描部分去掉: v5 把相似度值存下来后用 numpy.partition 求阈值/
      提取样本对；框架指标只吃直方图 => 一趟直出直方图即可

框架适配改动（原版会内存泄漏、依赖临时文件，统一契约下必须修）:
    ① 结果回传: 原 v5 每 GPU 写 tempfile 临时文件 gpu_k.pkl 再合并
       （200W 实测 FileNotFoundError 且 RSS 暴增 >450GB）=> 直方图经
       mp.Queue 回传，worker 不写任何文件（workdir 因此未使用）
    ② 数据传参: 原 v5 spawn 时把整个 feats/ids pickle 进每个 worker
       （每 worker 一份 GB 级拷贝）=> torch 共享内存张量，spawn 只传句柄
    ③ 不累积原始相似度: 原 v5 在 worker 内 append 全部 pos/neg 数值
       （206 亿对 ≈ 82GB，其内存泄漏根源）=> 分块直方图 int64 增量累加，
       内存 O(bins) 与样本对数无关
    ④ 网格: 原 v5 对原始数值直接阈值化（无网格）=> 统一 200,000 bins
       [-1,1]；sim clamp 到 [-1,1] 后入桶（浮点归一化残差越界不再丢对，
       histc 含边界: v==1.0 入末桶、v==-1.0 入首桶）
    ⑤ 计数严格精确: torch.histc 内部 fp32 累计，须单次调用内单桶计数
       < 2^24 才逐桶精确；故直方图按列切成 ≤3000 列的小段
       （每段元素 ≤ 5000×3000 = 1500 万 < 2^24），int64 累计
       => 正/负计数与理论值严格一致，可过框架零容忍校验
"""
import time

import numpy as np
import torch
import torch.multiprocessing as mp

from .. import common

MODEL_NAME = 'claude-opus'
MODEL_DESC = ('Claude-Opus 方法：静态行区间分片(无动态队列,行i属主卡算全局j>i)'
              '+ chunk5000 三角分块 fp32 matmul + id等值分正负 + 趟内直方图')
ORIGIN = ('claude-opus/eval_v5_final.py: compute_row_range_worker / '
          'compute_similarity_multigpu (tempfile gpu_i.pkl -> mp.Queue；'
          '累积原始相似度 -> 趟内直方图；原无网格 -> 统一 200K bins)')

BLOCK = 5000            # v5 默认 chunk_size=5000（实际运行值）
COLSEG = 3000           # 直方图按列分段上限：5000×3000=1500万 < 2^24（见文档⑤）
NATIVE_BINS = 0         # v5 无原生分箱（直接对原始相似度阈值化）


def _worker(cfg, feats_shm, ids_shm, res_q):
    """单卡 worker：负责行区间 [row_start, row_end)，只统计全局 (i<j) 的样本对。

    结构忠实 eval_v5_final.py 的 compute_row_range_worker：
    for i in range(row_start, row_end, chunk_size):
        chunk_i = feats[i:i_end]
        for j in range(i, N, chunk_size):          # 列块自行块起点扫到 N
            sim = chunk_i @ chunk_j.T
            if i == j: triu(diagonal=1) 只保留严格上三角
            pos = ids_i == ids_j；pos/neg 分开
    差异仅为：掩码值不再 .cpu() 攒列表，而是就地按段直方图累计（改动③⑤）。
    """
    gpu, B, bins, N = cfg['gpu'], cfg['block'], cfg['bins'], cfg['N']
    rs, re = cfg['row_start'], cfg['row_end']
    t0 = time.perf_counter()
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        dev = f'cuda:{gpu}'
        torch.cuda.set_device(dev)
        feats = feats_shm.to(dev)          # (N,512) fp32 常驻显存（v5 同样整表上卡）
        ids_g = ids_shm.to(dev)            # (N,) int64
        torch.cuda.synchronize()

        hist_all = torch.zeros(bins, dtype=torch.int64, device=dev)  # 全部有效对(正+负)
        hist_pos = torch.zeros(bins, dtype=torch.int64, device=dev)  # 其中同身份对
        t_mm = t_hist = 0.0
        n_tiles = 0
        # 预计 tile 数（行块 i 的列块数为 ceil((N-i)/B)），用于进度打印节奏
        expect = 0
        for i0 in range(rs, re, B):
            expect += (N - i0 + B - 1) // B

        with torch.no_grad():
            for i in range(rs, re, B):
                i_end = min(i + B, re)
                A = feats[i:i_end]                 # (b1, 512) 行块
                ia = ids_g[i:i_end]                # (b1,)
                for j in range(i, N, B):           # v5: 列块 j 自 i 扫到 N
                    j_end = min(j + B, N)
                    torch.cuda.synchronize()
                    s0 = time.perf_counter()
                    sim = A @ feats[j:j_end].t()   # (b1,b2) fp32
                    sim.clamp_(common.LO, common.HI)   # 网格 [-1,1]；越界残差收回
                    diag = (i == j)                # 对角块：行列同起点
                    if diag:
                        tri = torch.ones(sim.shape[0], sim.shape[1],
                                         dtype=torch.bool, device=dev).triu_(1)
                        sim.masked_fill_(~tri, common.LO - 2.0)  # 下三角/对角填范围外
                    torch.cuda.synchronize()
                    t_mm += time.perf_counter() - s0

                    s0 = time.perf_counter()
                    if diag:
                        posm = (ia[:, None] == ids_g[j:j_end][None, :]) & tri
                    else:
                        posm = ia[:, None] == ids_g[j:j_end][None, :]
                    # 按列分段直方图：单次 histc 内单桶计数 < 2^24 => fp32 累计精确
                    for cs in range(0, sim.shape[1], COLSEG):
                        ce = min(cs + COLSEG, sim.shape[1])
                        seg = sim[:, cs:ce]
                        if not seg.is_contiguous():
                            seg = seg.contiguous()
                        h = torch.histc(seg, bins=bins, min=common.LO, max=common.HI)
                        hist_all += h.round_().to(torch.int64)
                        pv = seg[posm[:, cs:ce]]
                        if pv.numel():
                            hp = torch.histc(pv, bins=bins, min=common.LO,
                                             max=common.HI)
                            hist_pos += hp.round_().to(torch.int64)
                    t_hist += time.perf_counter() - s0

                    n_tiles += 1
                    if n_tiles % 50 == 0 or n_tiles == expect:
                        n_pairs = int(hist_all.sum().item())
                        print(f'  [gpu{gpu}] 行[{rs},{re}) tile {n_tiles}/{expect} '
                              f'已算 {n_pairs:,} 对 matmul={t_mm/n_tiles*1000:.0f}ms '
                              f'hist={t_hist/n_tiles*1000:.0f}ms',
                              flush=True)

        torch.cuda.synchronize()
        hist_all -= hist_pos                       # 负样本直方图 = 全量 - 正样本
        assert hist_all.min().item() >= 0, '负样本直方图出现负计数'
        pos_hist = hist_pos.cpu().numpy()
        neg_hist = hist_all.cpu().numpy()
        res_q.put(('ok', pos_hist, neg_hist, t_mm, t_hist, n_tiles))
        print(f'  [gpu{gpu}] 完成: 行[{rs},{re}) {n_tiles} tiles '
              f'正={int(pos_hist.sum()):,} 负={int(neg_hist.sum()):,} '
              f'(matmul {t_mm:.1f}s hist {t_hist:.1f}s)', flush=True)
    except Exception as e:                         # noqa: BLE001
        res_q.put(('err', f'[gpu{gpu}] {type(e).__name__}: {e}'))
        print(f'  [gpu{gpu}] 失败: {type(e).__name__}: {e}', flush=True)


def compute(feats, ids, gpus, workdir):
    feats = np.ascontiguousarray(feats, dtype=np.float32)
    ids = np.ascontiguousarray(ids, dtype=np.int64)
    N = int(ids.shape[0])
    ng = len(gpus)
    if ng < 1 or N < 2:
        raise ValueError(f'需要 >=1 张卡且 N>=2 (N={N}, gpus={gpus})')

    # 静态行区间切分（v5: 每卡 N//n_gpu 行，末卡拿余数）
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
        if re <= rs:                               # 空区间（N<n 时）不启进程
            print(f'  [规划] gpu{gpu} 行区间为空，跳过', flush=True)
            continue
        cfg = {'gpu': gpu, 'row_start': rs, 'row_end': re,
               'block': BLOCK, 'bins': common.BINS, 'N': N}
        p = ctx.Process(target=_worker, args=(cfg, feats_shm, ids_shm, res_q))
        p.start()
        procs.append(p)

    t0 = time.perf_counter()
    # 边收边等：避免结果积压把管道写满导致 join 死锁
    got = []
    remaining = len(procs)
    while remaining > 0:
        try:
            item = res_q.get(timeout=30)
        except Exception:                          # noqa: BLE001
            if not any(p.is_alive() for p in procs):
                raise RuntimeError(f'claude-opus worker 提前退出，只收到 '
                                   f'{len(got)}/{len(procs)} 份结果')
            continue
        remaining -= 1
        got.append(item)
    for p in procs:
        p.join()

    errs = [it[1] for it in got if it[0] == 'err']
    if errs:
        raise RuntimeError('worker 出错: ' + '; '.join(errs))

    pos_hist = neg_hist = None
    t_mm = t_hist = 0.0
    n_tiles = 0
    for tag, hp, hn, mm, hs, nt in got:            # noqa: B007
        pos_hist = hp if pos_hist is None else pos_hist + hp
        neg_hist = hn if neg_hist is None else neg_hist + hn
        t_mm += mm
        t_hist += hs
        n_tiles += nt
    core_s = time.perf_counter() - t0

    if len(pos_hist) != common.BINS or len(neg_hist) != common.BINS:
        raise RuntimeError(f'直方图长度 {len(pos_hist)}/{len(neg_hist)} != {common.BINS}')

    # 计数自检（与 run_one 的统一校验同一标准，提前暴露分桶/覆盖 bug）
    total, exp_pos, exp_neg = common.pair_stats(ids)
    got_pos, got_neg = int(pos_hist.sum()), int(neg_hist.sum())
    if got_pos != exp_pos or got_neg != exp_neg or got_pos + got_neg != total:
        raise RuntimeError(f'计数校验失败: pos={got_pos:,} vs 理论 {exp_pos:,} '
                           f'neg={got_neg:,} vs 理论 {exp_neg:,} '
                           f'(合计 {got_pos + got_neg:,} vs 理论 {total:,})')

    meta = {
        'model': MODEL_NAME, 'desc': MODEL_DESC, 'origin': ORIGIN,
        'parallel': 'spawn 每GPU一进程 + 静态行区间切分(行i属主卡算全局j>i, 无动态队列)',
        'block': BLOCK, 'precision': 'fp32 (allow_tf32=False)',
        'native_bins': NATIVE_BINS,
        'core_s': round(core_s, 2), 'matmul_s': round(t_mm, 1),
        'hist_s': round(t_hist, 1),
    }
    print(f'[汇总] claude-opus 完成: 正={got_pos:,} 负={got_neg:,} '
          f'(合计 {got_pos + got_neg:,}) tiles={n_tiles} matmul={t_mm:.1f}s '
          f'hist={t_hist:.1f}s', flush=True)
    return pos_hist, neg_hist, meta
