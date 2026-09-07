#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
codex-sol-2 core —— 提炼自 codex-sol-2/face_similarity_evaluator.py

原实现要点（见原文件头部注释与 evaluate_worker/main）:
    * 只生成上三角分块任务（make_tasks），Linux fork 每 GPU 一个进程，
      按块乘法元素数贪心均衡静态分配任务（balance_tasks，宽块优先）
    * 预先建立“正样本稀疏坐标索引”（build_positive_pair_index）：
      按身份分组，组内/组间按块划分，生成每个非空 tile 内的
      (行,列) 局部 int32 坐标（只含 i<j 的正样本对）
    * worker 每 tile: torch.mm -> clamp[-1,1] -> 对角 tile 用 row>=col 掩码
      填 -2.0（范围外）；相似度用“分块 bincount + int64”精确直方图
      （_histogram，4M/chunk，避免 torch.histc float32 超过 2^24 失精）
    * 正样本直方图仅对索引坐标 gather 后统计；负样本直方图 = 全体直方图 - 正
      样本直方图（每 tile 内减法，两侧 bin 语义一致）
    * 默认精度 tf32（allow_tf32=True + float32_matmul_precision=high）
    * GPU 特征缓存 auto：特征 ≤ VRAM/4 时整表驻留显存，否则每 tile 拷贝
    * 原 bins 默认即 200,000，与统一网格一致（无需 rebin）

映射: make_tasks->_make_tasks, balance_tasks->_balance_tasks,
      build_positive_pair_index->_build_positive_index,
      evaluate_worker->_evaluate_worker, worker_entry->_worker_entry,
      run_workers/并行编排 -> compute() 内联。
去掉（框架统一承担）: 读 pkl/归一化校验/设备发现/正负样本对提取(top-k)/
      Parquet 输出/绘图/TPIR 插值指标/命令行参数解析。
"""
import math
import os
import time

import numpy as np
import torch
import torch.multiprocessing as mp

from .. import common

MODEL_NAME = 'codex-sol-2'
MODEL_DESC = ('Codex-sol-2 方法：Linux fork 每GPU一进程 + 上三角分块按元素数贪心'
              '均衡静态分配 + 正样本稀疏坐标索引(neg=all-pos) + int64 bincount 精确直方图')
ORIGIN = ('codex-sol-2/face_similarity_evaluator.py (make_tasks / balance_tasks / '
          'build_positive_pair_index / evaluate_worker / _histogram；块8192、默认tf32)')

BLOCK = 8192            # 原 --block-size 默认
_CHUNK = 4_000_000      # 原 _histogram 分块大小


# ---------------------------------------------------------------- 任务划分与均衡分配
def _make_tasks(sample_count: int, block: int):
    """生成上三角块任务 (bi, bj, r0, r1, c0, c1)；bj>=bi，对角块内部再取 i<j。"""
    nb = math.ceil(sample_count / block)
    tasks = []
    for bi in range(nb):
        r0, r1 = bi * block, min((bi + 1) * block, sample_count)
        for bj in range(bi, nb):
            c0, c1 = bj * block, min((bj + 1) * block, sample_count)
            tasks.append((bi, bj, r0, r1, c0, c1))
    return tasks


def _task_pairs(t):
    """块内样本对数（对角块为严格上三角 C(rows,2)）"""
    bi, bj, r0, r1, c0, c1 = t
    if bi == bj:
        n = r1 - r0
        return n * (n - 1) // 2
    return (r1 - r0) * (c1 - c0)


def _balance_tasks(tasks, worker_count: int):
    """按块元素数(样本对数)从大到小贪心分配给当前累计最少的 worker（原 balance_tasks）"""
    assignments = [[] for _ in range(worker_count)]
    assigned_pairs = [0] * worker_count
    for task in sorted(tasks, key=_task_pairs, reverse=True):
        wi = min(range(worker_count), key=assigned_pairs.__getitem__)
        assignments[wi].append(task)
        assigned_pairs[wi] += _task_pairs(task)
    return assignments


# ---------------------------------------------------------------- 正样本稀疏坐标索引
def _build_positive_index(codes: np.ndarray, block: int):
    """原 build_positive_pair_index：返回 {(bi,bj): (rows int32, cols int32)}。

    行/列为 tile 内局部坐标，只含全局 i<j 的同身份对：对角块内取 triu，
    跨块对按 (左块, 右块) 全量展开（同身份内排序保证左块索引 < 右块索引）。
    """
    parts = {}
    order = np.argsort(codes, kind='stable')
    sorted_codes = codes[order]
    # 身份分组边界（等值即同身份，不依赖标签取值/连续编码）
    group_starts = np.r_[0, np.flatnonzero(sorted_codes[1:] != sorted_codes[:-1]) + 1,
                         len(order)]
    for begin, end in zip(group_starts[:-1], group_starts[1:]):
        indices = order[begin:end]
        if len(indices) < 2:
            continue
        block_numbers = indices // block
        unique_blocks, starts = np.unique(block_numbers, return_index=True)
        starts = np.r_[starts, len(indices)]
        block_positions = []
        for pos, bn in enumerate(unique_blocks):
            local = (indices[starts[pos]:starts[pos + 1]]
                     - int(bn) * block).astype(np.int32, copy=False)
            block_positions.append((int(bn), local))
        for lp, (lb, left_idx) in enumerate(block_positions):
            if len(left_idx) >= 2:
                rows, cols = np.triu_indices(len(left_idx), k=1)
                parts.setdefault((lb, lb), []).append((left_idx[rows], left_idx[cols]))
            for rb, right_idx in block_positions[lp + 1:]:
                rows = np.repeat(left_idx, len(right_idx))
                cols = np.tile(right_idx, len(left_idx))
                parts.setdefault((lb, rb), []).append((rows, cols))

    index = {}
    for key, coord_parts in parts.items():
        index[key] = (
            np.concatenate([r for r, _ in coord_parts]).astype(np.int32, copy=False),
            np.concatenate([c for _, c in coord_parts]).astype(np.int32, copy=False),
        )
    return index


# ---------------------------------------------------------------- 直方图（int64 精确）
def _histogram(scores: torch.Tensor, bins: int) -> torch.Tensor:
    """原 _histogram：分块 bincount 的 int64 精确直方图。

    bin k 覆盖 [-1 + k*w, -1 + (k+1)*w)（w=2/bins；恰为 1.0 钳到最后桶，
    与框架 torch.histc / np.histogram 语义一致）。不用 float32 累加的
    torch.histc，避免单桶计数超过 2^24 时失精。
    """
    flat_scores = scores.reshape(-1)
    histogram = torch.zeros(bins, dtype=torch.int64, device=scores.device)
    scale = bins / 2.0
    for start in range(0, flat_scores.numel(), _CHUNK):
        chunk = flat_scores[start:start + _CHUNK]
        valid = (chunk >= common.LO) & (chunk <= common.HI)   # -2.0 填充/越界值丢弃
        bin_index = torch.floor((chunk[valid] + 1.0) * scale).to(torch.int64)
        bin_index.clamp_(min=0, max=bins - 1)
        histogram.add_(torch.bincount(bin_index, minlength=bins))
    return histogram


# ---------------------------------------------------------------- GPU worker
def _evaluate_worker(device_name: str, tasks, cfg, feats: np.ndarray, pos_index: dict):
    """原 evaluate_worker（去掉 top-k 样本对提取）"""
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    if not torch.cuda.is_available():
        raise RuntimeError(f'CUDA 不可用，无法使用 {device_name}')
    torch.backends.cuda.matmul.allow_tf32 = True               # 原默认精度 tf32
    torch.set_float32_matmul_precision('high')

    started_at = time.perf_counter()
    bins = cfg['bins']
    feature_cpu = torch.from_numpy(feats)                      # fork COW 共享，零拷贝
    # 原 should_cache_features(auto)：特征不超 VRAM 的 1/4 时整表驻留显存
    feature_bytes = feats.nbytes
    total_memory = torch.cuda.get_device_properties(device).total_memory
    cache_enabled = feature_bytes <= total_memory // 4
    if cache_enabled:
        feature_source = feature_cpu.to(device, non_blocking=False)
    else:
        feature_source = feature_cpu

    hist_pos = torch.zeros(bins, dtype=torch.int64, device=device)
    hist_neg = torch.zeros(bins, dtype=torch.int64, device=device)
    n_pos = 0
    done = 0
    total_tasks = len(tasks)
    report_step = max(1, total_tasks // 10)
    mm_pairs, hist_pairs = [], []        # (CUDA 事件对)，结束后一次性统计 GPU 耗时

    for task in tasks:
        bi, bj, r0, r1, c0, c1 = task
        if cache_enabled:
            row_feats = feature_source[r0:r1]
            col_feats = feature_source[c0:c1]
        else:
            row_feats = feature_source[r0:r1].to(device, non_blocking=False)
            col_feats = feature_source[c0:c1].to(device, non_blocking=False)

        e_mm0 = torch.cuda.Event(enable_timing=True); e_mm0.record()
        scores = torch.mm(row_feats, col_feats.t())
        scores.clamp_(min=common.LO, max=common.HI)
        if bi == bj:
            diag_idx = torch.arange(scores.shape[0], device=device)
            invalid = diag_idx[:, None] >= diag_idx[None, :]   # i>=j 填范围外 -> 丢弃
            scores.masked_fill_(invalid, common.LO - 2.0)
        e_hist = torch.cuda.Event(enable_timing=True); e_hist.record()
        mm_pairs.append((e_mm0, e_hist))

        all_hist = _histogram(scores, bins)                    # 本 tile 全体样本
        coords = pos_index.get((bi, bj))
        if coords is not None:
            p_rows = torch.as_tensor(coords[0], dtype=torch.long, device=device)
            p_cols = torch.as_tensor(coords[1], dtype=torch.long, device=device)
            pos_scores = scores[p_rows, p_cols]                # 仅统计稀疏正样本
            pos_hist_tile = _histogram(pos_scores, bins)
            hist_pos.add_(pos_hist_tile)
            hist_neg.sub_(pos_hist_tile)                       # 负样本 = 全体 - 正样本
            n_pos += int(pos_scores.numel())
        hist_neg.add_(all_hist)
        e_end = torch.cuda.Event(enable_timing=True); e_end.record()
        hist_pairs.append((e_hist, e_end))

        del scores
        done += 1
        if report_step and done % report_step == 0:
            print(f'  [codex-sol-2 {device_name}] 进度 {done}/{total_tasks} tiles',
                  flush=True)

    torch.cuda.synchronize(device)
    mm_s = sum(a.elapsed_time(b) for a, b in mm_pairs) / 1000.0
    hist_s = sum(a.elapsed_time(b) for a, b in hist_pairs) / 1000.0
    return {
        'device': device_name,
        'task_count': total_tasks,
        'pair_count': sum(_task_pairs(t) for t in tasks),
        'n_pos': n_pos,
        'pos_hist': hist_pos.cpu().numpy(),
        'neg_hist': hist_neg.cpu().numpy(),
        'cache_enabled': cache_enabled,
        'mm_s': mm_s, 'hist_s': hist_s,
        'wall_s': time.perf_counter() - started_at,
    }


def _worker_entry(device_name, tasks, cfg, feats, pos_index, res_q):
    """模块级顶层入口（fork/spawn 均安全）：worker 异常以结果消息返回主进程。"""
    try:
        result = _evaluate_worker(device_name, tasks, cfg, feats, pos_index)
        result['ok'] = True
    except BaseException as e:  # noqa: BLE001
        result = {'ok': False, 'device': device_name,
                  'error': f'{type(e).__name__}: {e}'}
    res_q.put(result)


# ---------------------------------------------------------------- 统一契约入口
def compute(feats, ids, gpus, workdir):
    feats = np.ascontiguousarray(feats, dtype=np.float32)
    ids = np.ascontiguousarray(ids, dtype=np.int64)
    N = len(ids)
    if feats.ndim != 2 or feats.shape[0] != N:
        raise ValueError(f'feats 形状 {feats.shape} 与 ids N={N} 不一致')
    if N < 2:
        raise ValueError('样本数 < 2，无法组成样本对')
    if not gpus:
        raise ValueError('codex-sol-2 core 要求至少 1 张 GPU')
    os.makedirs(workdir, exist_ok=True)

    # 身份编码为 0..K-1（仅用于分组等值判断；正样本统计不依赖编码取值）
    _, codes = np.unique(ids, return_inverse=True)
    codes = np.ascontiguousarray(codes.astype(np.int64, copy=False))
    _, counts = np.unique(codes, return_counts=True)
    exp_pos = int((counts.astype(np.int64) * (counts - 1) // 2).sum())
    exp_total = N * (N - 1) // 2
    exp_neg = exp_total - exp_pos

    # 正样本稀疏索引（CPU，fork 后子进程 COW 只读共享）
    t0 = time.perf_counter()
    pos_index = _build_positive_index(codes, BLOCK)
    index_s = time.perf_counter() - t0
    indexed_pos = int(sum(len(r) for r, _ in pos_index.values()))
    if indexed_pos != exp_pos:
        raise RuntimeError(f'正样本索引自校验失败: {indexed_pos:,} != 理论 {exp_pos:,}')

    tasks = _make_tasks(N, BLOCK)
    assignments = _balance_tasks(tasks, len(gpus))
    print(f'[codex-sol-2] N={N:,} d={feats.shape[1]} 块={BLOCK} tiles={len(tasks):,} '
          f'gpus={len(gpus)} 正样本索引 {len(pos_index):,} 块/{indexed_pos:,} 对 '
          f'索引耗时 {index_s:.2f}s', flush=True)

    # 原实现为 Linux fork：fork 前父进程不触碰 CUDA（本 core 全程纯 CPU 直到 worker）
    ctx = mp.get_context('fork')
    res_q = ctx.Queue()
    cfg = {'bins': common.BINS}
    procs = [ctx.Process(target=_worker_entry,
                         args=(f'cuda:{g}', assignments[i], cfg, feats, pos_index, res_q))
             for i, g in enumerate(gpus)]
    t0 = time.perf_counter()
    for p in procs:
        p.start()
    # 边跑边取结果（原 run_workers 模式）：持续排空队列避免 feeder 管道写满死锁；
    # worker 异常或崩溃（exitcode!=0）时终止其余进程并报错
    results = []
    try:
        while len(results) < len(procs):
            try:
                r = res_q.get(timeout=1)
            except Exception:                                 # noqa: BLE001
                dead = [p.exitcode for p in procs if p.exitcode not in (None, 0)]
                if dead:
                    raise RuntimeError(f'GPU worker 异常退出 exitcode={dead}')
                continue
            results.append(r)
            if not r.get('ok'):
                raise RuntimeError(f"{r['device']} 执行失败: {r['error']}")
    except BaseException:                                     # noqa: BLE001
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join()
        raise
    for p in procs:
        p.join()
    core_s = time.perf_counter() - t0

    pos_hist = np.zeros(common.BINS, dtype=np.int64)
    neg_hist = np.zeros(common.BINS, dtype=np.int64)
    got_pos = got_neg = 0
    n_pairs = 0
    mm_s = hist_s = 0.0
    for r in results:
        if not r['ok']:
            raise RuntimeError(f"{r['device']} 执行失败: {r['error']}")
        pos_hist += np.asarray(r['pos_hist'], dtype=np.int64)
        neg_hist += np.asarray(r['neg_hist'], dtype=np.int64)
        got_pos += int(r['n_pos'])
        n_pairs += int(r['pair_count'])
        mm_s += float(r['mm_s']); hist_s += float(r['hist_s'])
        print(f"  [codex-sol-2 {r['device']}] 完成 {r['task_count']} tiles / "
              f"{r['pair_count']:,} 对 / 正 {r['n_pos']:,} / 耗时 {r['wall_s']:.1f}s "
              f"(matmul {r['mm_s']:.1f}s hist {r['hist_s']:.1f}s "
              f"cache={r['cache_enabled']})", flush=True)

    got_neg_total = int(neg_hist.sum())
    if got_pos != exp_pos or got_neg_total != exp_neg:
        raise RuntimeError(f'样本对统计自校验失败: 正 {got_pos:,}/{exp_pos:,} '
                           f'负 {got_neg_total:,}/{exp_neg:,} 总 {n_pairs:,}/{exp_total:,}')
    if int(neg_hist.min()) < 0:
        raise RuntimeError('负样本直方图出现负计数')

    meta = {
        'model': MODEL_NAME, 'desc': MODEL_DESC, 'origin': ORIGIN,
        'parallel': 'Linux fork 每GPU一进程 + 上三角分块按元素数贪心均衡静态分配(宽块优先)',
        'block': BLOCK,
        'precision': 'tf32 (allow_tf32=True, float32_matmul_precision=high)',
        'native_bins': int(common.BINS),
        'hist': 'chunked int64 bincount (4e6/chunk)；正样本稀疏坐标索引 gather，'
                '负样本=全体-正样本（逐tile，bin语义一致）',
        'gpu_cache': 'auto（特征<=VRAM/4 时整表驻留显存）',
        'n_tiles': len(tasks), 'pos_index_blocks': len(pos_index),
        'index_s': round(index_s, 3), 'core_s': round(core_s, 3),
        'matmul_s': round(mm_s, 3), 'hist_s': round(hist_s, 3),
    }
    print(f'[codex-sol-2] 完成: 正 {got_pos:,} 负 {got_neg_total:,} '
          f'core {core_s:.1f}s (matmul合计 {mm_s:.1f}s hist合计 {hist_s:.1f}s)',
          flush=True)
    return pos_hist, neg_hist, meta
